"""Adapter: run the unchanged VANTAGE_Temporal evaluator against a submission
JSONL and a private GT JSONL.

The adapter:
1. Loads + validates the submission.
2. Loads private GT.
3. Constructs a VANTAGE_Temporal dataset instance via test_mode=True (an
   existing supported parameter that skips super().__init__() and the TSV
   disk read).
4. Assigns a synthetic self.data carrying exactly the columns the existing
   evaluate() / _compute_metrics read (index, answer, duration, video, plus
   passenger columns).
5. Builds a synthetic prediction DataFrame, dumps it as xlsx to a temp path.
6. Calls the existing, UNCHANGED dataset.evaluate(temp_xlsx, **judge_kwargs).

Score parity with the legacy xlsx path is structural: the parser sees the
same raw bytes (the assistant.value string), the evaluator sees the same
DataFrame columns it reads today, and the metric functions (parse_timestamps,
iou, np.mean) are deterministic for the same inputs.
"""

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
    """Return a DataFrame carrying exactly the columns VANTAGE_Temporal's
    evaluate() / _compute_metrics read.

    Columns provided:
      - index    : canonical id (string)
      - answer   : verbatim JSON-string from conversations[from="gpt"].value
      - duration : float (from metadata.extra.duration, with 30.0 fallback)
      - video    : per-vid aggregation key (canonical media)
      - category : passenger (evaluator uses self.get_category(video) instead)
      - question : passenger (verbose-print only)
    """
    rows = []
    for r in private_gt:
        extra = (r.get('metadata') or {}).get('extra', {}) or {}
        # Default to 30.0 to preserve legacy evaluator behavior when
        # duration is missing.
        dur = extra.get('duration')
        if dur is None:
            dur = 30.0
        rows.append({
            'index': r['id'],
            'answer': _extract_gt_value(r),
            'duration': float(dur),
            'video': r.get('media', ''),
            'category': r.get('category', ''),
            'question': extra.get('question', ''),
        })
    return pd.DataFrame(rows)


def _build_synthetic_pred_df(submission):
    """Return a DataFrame with the two columns evaluate() reads from the pred file."""
    rows = []
    for r in submission:
        rows.append({
            'index': r['id'],
            'prediction': _extract_assistant_value(r),
        })
    return pd.DataFrame(rows)


def evaluate_temporal_submission(
    submission_path,
    private_gt_path,
    work_dir=None,
    **judge_kwargs,
):
    """End-to-end Temporal evaluation from submission + private GT JSONL.

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
        Forwarded verbatim to VANTAGE_Temporal.evaluate().

    Returns
    -------
    dict
        Whatever the existing VANTAGE_Temporal.evaluate() returns. Today that
        is a dict with keys 'overall' and 'category_metrics'.
    """
    # Local import to avoid any module-load-time side effects on non-Temporal runs.
    from vlmeval.dataset.vantage_temporal import VANTAGE_Temporal

    submission = read_jsonl(submission_path)
    private_gt = read_jsonl(private_gt_path)

    validate_submission(submission, private_gt, task='temporal')

    # test_mode=True bypasses super().__init__() and the TSV disk read.
    ds = VANTAGE_Temporal(test_mode=True)
    ds.data = _build_synthetic_ds_data(private_gt)

    pred_df = _build_synthetic_pred_df(submission)

    cleanup_dir = None
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='vantage_tg_eval_')
        cleanup_dir = work_dir
    os.makedirs(work_dir, exist_ok=True)
    tmp_pred_path = osp.join(work_dir, '_tg_submission_pred.xlsx')
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
