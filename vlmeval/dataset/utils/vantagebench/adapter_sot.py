"""Adapter: run the unchanged VANTAGE_SOT evaluator against a submission
JSONL and a private GT JSONL.

The adapter:
1. Loads + validates the submission.
2. Loads private GT.
3. Constructs a VANTAGE_SOT instance via __new__ (no __init__).
4. Sets ds.verbose = False explicitly (the evaluator reads it).
5. Builds a synthetic _gt_cache KEYED BY CANONICAL ID (seq_dir.name).
   This is the central Phase 8 design change: cache is keyed by stable
   semantic id, not by enumeration int. The evaluator's lookup
   `self._gt_cache.get(idx)` works for any hashable key.
6. Builds a synthetic prediction DataFrame with index = canonical id,
   prediction = raw assistant.value.
7. Calls the existing, UNCHANGED dataset.evaluate(temp_xlsx, **judge_kwargs).

CRITICAL preservations:
  * The evaluator's result dict keys by `cache['label']` (slash-form
    label like "Warehouse_000/Camera_0003/5648/obj37"), NOT by the
    canonical id. Both paths produce identical result-dict key sets.
  * compute_sot_metrics, compute_seq_tags, iou_2d, parse_sot_response
    are all unchanged.
  * No file materialization is needed (the evaluator works purely on
    in-memory cache; frame PNGs are inference-input only).
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


def _build_synthetic_gt_cache(private_gt):
    """Build _gt_cache keyed by canonical id (seq_dir.name).

    Mirrors the cache shape the evaluator reads from. Frame PNG paths and
    crop_path are passed through but never opened at eval time.
    """
    cache = {}
    for r in private_gt:
        rid = r['id']
        extra = (r.get('metadata') or {}).get('extra', {}) or {}
        # gpt.value is a JSON-string of {str(int): bbox} from gt.json.
        try:
            gpt_dict = json.loads(_extract_gt_value(r))
        except (json.JSONDecodeError, TypeError):
            gpt_dict = {}
        # Coerce keys back to int for cache parity with legacy _prepare_data_from_dir.
        gt_bboxes = {}
        for k, v in gpt_dict.items():
            try:
                gt_bboxes[int(k)] = v
            except (ValueError, TypeError):
                continue

        cache[rid] = {
            'frame_ids':   list(extra.get('frame_ids', [])),
            'gt_bboxes':   gt_bboxes,
            'occluded':    extra.get('occluded', {}) or {},
            'init_bbox':   list(extra.get('init_bbox', [])),
            'object_type': str(extra.get('object_type', 'Person')),
            'label':       str(extra.get('label', rid)),
            'video_path':  str(extra.get('video_path', '')),
            # Frame paths and crop path are forensic only; evaluator does
            # not open them at eval time.
            'frame_paths': [],
            'crop_path':   '',
            'seq_dir':     str(extra.get('seq_dir_abspath', '')),
        }
    return cache


def _build_synthetic_pred_df(submission):
    """Synthetic pred DataFrame with the two columns evaluate() reads."""
    rows = []
    for r in submission:
        rows.append({
            'index': r['id'],
            'prediction': _extract_assistant_value(r),
        })
    return pd.DataFrame(rows)


def evaluate_sot_submission(
    submission_path,
    private_gt_path,
    work_dir=None,
    **judge_kwargs,
):
    """End-to-end SOT evaluation from submission + private GT JSONL.

    Parameters
    ----------
    submission_path : str
        Path to a submission JSONL.
    private_gt_path : str
        Path to a private GT JSONL.
    work_dir : str or None
        Directory in which to write the temporary synthetic prediction
        file. Defaults to a fresh tempfile.mkdtemp().
    **judge_kwargs
        Forwarded verbatim to VANTAGE_SOT.evaluate().

    Returns
    -------
    dict
        Whatever the existing VANTAGE_SOT.evaluate() returns: a dict
        keyed by cache['label'] (slash-form) plus 'Overall'.
    """
    # Local import to avoid module-load-time side effects on non-SOT runs.
    from vlmeval.dataset.vantage_sot import VANTAGE_SOT

    submission = read_jsonl(submission_path)
    private_gt = read_jsonl(private_gt_path)

    validate_submission(submission, private_gt, task='sot')

    # __new__ bypasses __init__ entirely. No prepared_data_dir resolution,
    # no _prepare_data_from_dir call, no filesystem scan.
    ds = VANTAGE_SOT.__new__(VANTAGE_SOT)
    ds.verbose = False
    ds._gt_cache = _build_synthetic_gt_cache(private_gt)
    # data is not consulted by evaluate(); placeholder for inherited helpers.
    ds.data = pd.DataFrame([])

    pred_df = _build_synthetic_pred_df(submission)

    cleanup_dir = None
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='vantage_sot_eval_')
        cleanup_dir = work_dir
    os.makedirs(work_dir, exist_ok=True)
    tmp_pred_path = osp.join(work_dir, '_sot_submission_pred.xlsx')
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
