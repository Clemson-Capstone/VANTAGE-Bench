"""Adapter: run the unchanged Astro2DDetectionDataset evaluator against a
submission JSONL and a private GT JSONL.

The adapter:
1. Loads + validates the submission.
2. Loads private GT.
3. Constructs an Astro2DDetectionDataset instance via __new__ (no __init__).
4. Materializes synthetic KITTI label files in a temp dir, faithfully
   reproducing the shape that parse_kitti_label expects.
5. Sets self.labels_dir = <tmp dir>, self.min_bbox_area = <from canonical GT>,
   self._gt_cache = {}.
6. Builds a synthetic prediction DataFrame with image_path / image_filename /
   prediction columns. image_path points at the REAL on-disk image so the
   evaluator's PIL.Image.open succeeds (this is identical to the legacy
   path's image-file requirement).
7. Calls the existing, UNCHANGED dataset.evaluate(temp_xlsx, **judge_kwargs).

CRITICAL preservations:
  * The Gemini `box_2d` axis-swap branch is preserved BY DEFAULT because the
    adapter passes assistant.value as raw text through dataset.evaluate(),
    which internally calls parse_bbox_2d_from_text. We do nothing to that.
  * min_bbox_area is set explicitly on the instance from canonical GT
    metadata. The runtime substring-on-data_root quirk inside __init__ is
    bypassed (we use __new__) — but the resolved value is identical because
    the converter applies the same rule at convert time.
  * The PIL.Image.open silent fallback to 640x480 is preserved unchanged.
  * map_label_to_person is preserved unchanged inside evaluate().
"""

import json
import os
import os.path as osp
import shutil
import tempfile

import pandas as pd

from vlmeval.smp import dump

from .io import read_jsonl
from .validate import SubmissionValidationError, validate_submission


def _extract_assistant_value(record):
    for turn in record.get('conversations', []):
        if isinstance(turn, dict) and turn.get('from') == 'assistant':
            v = turn.get('value')
            if isinstance(v, str):
                return v
    return ''


def _extract_gt_value(record):
    for turn in record.get('conversations', []):
        if isinstance(turn, dict) and turn.get('from') == 'gpt':
            v = turn.get('value')
            if isinstance(v, str):
                return v
    return ''


def _format_kitti_line(label, bbox):
    """Format a single KITTI label line. Only label + 4 bbox floats are
    semantically meaningful to parse_kitti_label; the 11 placeholder floats
    are the standard KITTI fields the evaluator ignores.
    """
    x1, y1, x2, y2 = bbox[:4]
    return (
        f"{label} 0.00 0 0.00 {x1} {y1} {x2} {y2} "
        "0.00 0.00 0.00 0.00 0.00 0.00 0.00\n"
    )


def _materialize_synthetic_kitti(private_gt, labels_dir):
    """For each GT record, write a synthetic KITTI .txt at labels_dir/<stem>.txt
    using the JSON-string list of {label, bbox} stored in gpt.value.
    """
    for r in private_gt:
        extra = (r.get('metadata') or {}).get('extra', {}) or {}
        image_filename = extra.get('image_filename') or r.get('media', '')
        if not image_filename:
            continue
        base_name = osp.splitext(image_filename)[0]
        kitti_path = osp.join(labels_dir, base_name + '.txt')
        try:
            objects = json.loads(_extract_gt_value(r))
        except (json.JSONDecodeError, TypeError):
            objects = []
        with open(kitti_path, 'w') as f:
            for obj in objects:
                if 'label' not in obj or 'bbox' not in obj:
                    continue
                if not isinstance(obj['bbox'], (list, tuple)) or len(obj['bbox']) < 4:
                    continue
                f.write(_format_kitti_line(obj['label'], obj['bbox']))


def _build_synthetic_pred_df(submission, private_gt):
    """Synthetic pred DataFrame carrying the columns evaluate() reads from
    each row.

    Columns:
      - index          : canonical id (string)
      - prediction     : verbatim assistant.value (raw string)
      - image_path     : absolute on-disk path (for PIL.Image.open)
      - image_filename : basename (for _load_ground_truth lookup)
    """
    gt_by_id = {r['id']: r for r in private_gt}
    rows = []
    for s in submission:
        rid = s['id']
        gt = gt_by_id.get(rid, {})
        extra = (gt.get('metadata') or {}).get('extra', {}) or {}
        rows.append({
            'index': rid,
            'prediction': _extract_assistant_value(s),
            'image_path': extra.get('image_path', ''),
            'image_filename': extra.get('image_filename', gt.get('media', '')),
        })
    return pd.DataFrame(rows)


def _resolve_min_bbox_area_from_gt(private_gt):
    """Pull the per-dataset min_bbox_area from canonical GT. All records in a
    single Astro2D dataset share the same value; we read it from the first.
    """
    if not private_gt:
        return 0
    extra = (private_gt[0].get('metadata') or {}).get('extra', {}) or {}
    return int(extra.get('min_bbox_area', 0))


def evaluate_astro_submission(
    submission_path,
    private_gt_path,
    work_dir=None,
    **judge_kwargs,
):
    """End-to-end Astro2D evaluation from submission + private GT JSONL.

    Parameters
    ----------
    submission_path : str
        Path to a submission JSONL.
    private_gt_path : str
        Path to a private GT JSONL.
    work_dir : str or None
        Directory in which to write the temporary synthetic prediction file
        and synthetic KITTI labels dir. Defaults to a fresh tempfile.mkdtemp().
    **judge_kwargs
        Forwarded verbatim to Astro2DDetectionDataset.evaluate().

    Returns
    -------
    dict
        Whatever the existing Astro2DDetectionDataset.evaluate() returns.
        Today that is a flat dict with keys precision/recall/f1/f1_0.95/
        f1_mIOU/total_predictions/valid_bbox_predictions/valid_rate/
        total_gt_objects/total_pred_objects/true_positives/false_positives/
        false_negatives/gt_filtered_small/pred_filtered_small.
    """
    # Local import to avoid module-load-time side effects on non-astro runs.
    from vlmeval.dataset.vantage2d.astro_2d_dataset import Astro2DDetectionDataset

    submission = read_jsonl(submission_path)
    private_gt = read_jsonl(private_gt_path)

    validate_submission(submission, private_gt, task='astro')

    cleanup_dir = None
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='vantage_ol_eval_')
        cleanup_dir = work_dir
    os.makedirs(work_dir, exist_ok=True)

    # Materialize synthetic KITTI labels in a sub-directory of work_dir.
    synth_labels_dir = osp.join(work_dir, '_ol_synth_labels')
    os.makedirs(synth_labels_dir, exist_ok=True)
    _materialize_synthetic_kitti(private_gt, synth_labels_dir)

    # __new__ bypasses __init__ entirely.
    ds = Astro2DDetectionDataset.__new__(Astro2DDetectionDataset)
    ds.labels_dir = synth_labels_dir
    ds.min_bbox_area = _resolve_min_bbox_area_from_gt(private_gt)
    ds._gt_cache = {}
    # Set ds.data to an empty placeholder to avoid AttributeError if any
    # inherited helper probes it. The evaluator itself never indexes self.data.
    ds.data = pd.DataFrame([])

    pred_df = _build_synthetic_pred_df(submission, private_gt)
    tmp_pred_path = osp.join(work_dir, '_ol_submission_pred.xlsx')
    dump(pred_df, tmp_pred_path)

    try:
        result = ds.evaluate(tmp_pred_path, **judge_kwargs)
    finally:
        # Best-effort cleanup of the synthetic labels dir.
        try:
            shutil.rmtree(synth_labels_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            if osp.exists(tmp_pred_path):
                os.remove(tmp_pred_path)
        except OSError:
            pass
        if cleanup_dir is not None:
            try:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            except Exception:
                pass

    return result
