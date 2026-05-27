"""Adapter: run the unchanged VANTAGE_2DGroundingDataset evaluator against a
submission JSONL and a private GT JSONL.

The adapter:
1. Loads + validates the submission.
2. Loads private GT.
3. Constructs a VANTAGE_2DGroundingDataset instance via __new__ (no __init__),
   because the class's __init__ mandates data_root/annotation_file/disk-image
   filtering and does not accept a test_mode kwarg. __new__ is verified safe
   because evaluate() reads only self.data — and even that read is a
   conditional fallback that never fires when the prediction row already
   carries gt_bboxes/image_width/image_height (which our synthetic row does).
4. Assigns a synthetic self.data carrying the columns the fallback path
   would consult (for safety).
5. Builds a synthetic prediction DataFrame carrying exactly the columns the
   evaluator reads from each pred row: index, prediction, gt_bboxes,
   image_width, image_height, image_path.
6. Calls the existing, UNCHANGED dataset.evaluate(temp_xlsx, **judge_kwargs).

Coordinate convention preservation: the adapter performs ZERO coordinate
transformation. Predictions are 0-1000 normalized (passed verbatim as a raw
string); GT is pixel-coord JSON-string (passed verbatim). The existing
evaluator does all scaling (`p * w / 1000`) and clipping.
"""

import json
import os
import os.path as osp
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


def _build_synthetic_ds_data(private_gt):
    """Return a DataFrame for the fallback lookup path in evaluate().

    The evaluator only consults self.data when row['gt_bboxes'] is None. Our
    synthetic pred_df always carries gt_bboxes, so this fallback should not
    fire — but we populate self.data with the right shape for safety.

    Columns:
      - index       : canonical id (string, matches legacy str-type)
      - gt_bboxes   : verbatim JSON-string of pixel-coord GT list-of-lists
      - image_width : int (or None if absent)
      - image_height: int (or None if absent)
      - image_path  : '' (the image-open sub-fallback should not fire either)
    """
    rows = []
    for r in private_gt:
        extra = (r.get('metadata') or {}).get('extra', {}) or {}
        rows.append({
            'index': r['id'],
            'gt_bboxes': _extract_gt_value(r),
            'image_width': extra.get('image_width'),
            'image_height': extra.get('image_height'),
            'image_path': '',
        })
    return pd.DataFrame(rows)


def _build_synthetic_pred_df(submission, private_gt):
    """Return a DataFrame with every column evaluate() reads from each pred row.

    The submission carries only id + raw prediction. We join to private GT by
    id to populate the rest (gt_bboxes, image dims, image_path).

    Columns:
      - index       : canonical id
      - prediction  : raw assistant.value
      - gt_bboxes   : verbatim JSON-string from private GT gpt.value
      - image_width : int from private GT metadata.extra.image_width
      - image_height: int from private GT metadata.extra.image_height
      - image_path  : '' (image-open fallback never fires when dims are present)
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
            'gt_bboxes': _extract_gt_value(gt),
            'image_width': extra.get('image_width'),
            'image_height': extra.get('image_height'),
            'image_path': '',
        })
    return pd.DataFrame(rows)


def evaluate_grounding_submission(
    submission_path,
    private_gt_path,
    work_dir=None,
    **judge_kwargs,
):
    """End-to-end 2DGrounding evaluation from submission + private GT JSONL.

    Parameters
    ----------
    submission_path : str
        Path to a submission JSONL (one record per line).
    private_gt_path : str
        Path to a private GT JSONL.
    work_dir : str or None
        Directory in which to write the temporary synthetic prediction file.
        Defaults to a fresh tempfile.mkdtemp() directory.
    **judge_kwargs
        Forwarded verbatim to VANTAGE_2DGroundingDataset.evaluate().

    Returns
    -------
    dict
        Whatever the existing VANTAGE_2DGroundingDataset.evaluate() returns.
        Today that is a dict with keys
        Acc@0.25, Acc@0.5, Acc@0.75, Mean_IoU, total_samples, valid_predictions,
        valid_rate, parse_failures, invalid_boxes_filtered.
    """
    # Local import to avoid module-load-time side effects on non-grounding runs.
    from vlmeval.dataset.vantage2d.grounding_2d_dataset import VANTAGE_2DGroundingDataset

    submission = read_jsonl(submission_path)
    private_gt = read_jsonl(private_gt_path)

    validate_submission(submission, private_gt, task='grounding')

    # __new__ bypasses __init__ entirely — no annotations / images disk read.
    ds = VANTAGE_2DGroundingDataset.__new__(VANTAGE_2DGroundingDataset)
    ds.data = _build_synthetic_ds_data(private_gt)

    pred_df = _build_synthetic_pred_df(submission, private_gt)

    cleanup_dir = None
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='vantage_rx_eval_')
        cleanup_dir = work_dir
    os.makedirs(work_dir, exist_ok=True)
    tmp_pred_path = osp.join(work_dir, '_rx_submission_pred.xlsx')
    dump(pred_df, tmp_pred_path)

    try:
        result = ds.evaluate(tmp_pred_path, **judge_kwargs)
    finally:
        try:
            if osp.exists(tmp_pred_path):
                os.remove(tmp_pred_path)
        except OSError:
            pass
        if cleanup_dir is not None:
            for fname in os.listdir(cleanup_dir):
                try:
                    os.remove(osp.join(cleanup_dir, fname))
                except OSError:
                    pass
            try:
                os.rmdir(cleanup_dir)
            except OSError:
                pass

    return result
