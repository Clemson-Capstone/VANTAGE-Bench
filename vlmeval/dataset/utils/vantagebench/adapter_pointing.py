"""Adapter: run the unchanged VANTAGE_2DPointing evaluator against a submission
JSONL and a private GT JSONL.

The adapter:
1. Loads + validates the submission.
2. Loads private GT.
3. Constructs a VANTAGE_2DPointing instance via __new__ (no __init__).
4. Sets ds.verbose = False explicitly (evaluator at line 3005 reads it).
5. Assigns a synthetic self.data carrying exactly the columns the evaluator
   reads: index, answer, question_id. Critically, question_id is required
   because the evaluator derives the bucket category from it.
6. Builds a synthetic prediction DataFrame, dumps it as xlsx to a temp path.
7. Calls the existing, UNCHANGED dataset.evaluate(temp_xlsx, **judge_kwargs).

The MCQ semantics are preserved: gpt.value (the GT) is a single letter
A/B/C/D; assistant.value (the prediction) is raw text from which
extract_answer() recovers a letter. The target_point coordinate is forensic
only and is NEVER consulted at scoring time.
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
    """Return a DataFrame carrying exactly the columns VANTAGE_2DPointing's
    evaluate() reads from self.data.

    Columns provided:
      - index       : canonical id (string)
      - answer      : single GT letter A/B/C/D from conversations[from="gpt"].value
      - question_id : required for the evaluator's category-derivation path
                      (line 3041-3043 in image_mcq.py)
    """
    rows = []
    for r in private_gt:
        extra = (r.get('metadata') or {}).get('extra', {}) or {}
        rows.append({
            'index': r['id'],
            'answer': _extract_gt_value(r),
            'question_id': extra.get('question_id', ''),
        })
    return pd.DataFrame(rows)


def _build_synthetic_pred_df(submission):
    rows = []
    for r in submission:
        rows.append({
            'index': r['id'],
            'prediction': _extract_assistant_value(r),
        })
    return pd.DataFrame(rows)


def evaluate_pointing_submission(
    submission_path,
    private_gt_path,
    work_dir=None,
    **judge_kwargs,
):
    """End-to-end 2DPointing evaluation from submission + private GT JSONL.

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
        Forwarded verbatim to VANTAGE_2DPointing.evaluate().

    Returns
    -------
    dict
        Whatever the existing VANTAGE_2DPointing.evaluate() returns. Today
        that is a flat dict: {'Overall': <float>, '<category>': <float>, ...}.
    """
    # Local import to avoid module-load-time side effects on non-pointing runs.
    from vlmeval.dataset.image_mcq import VANTAGE_2DPointing

    submission = read_jsonl(submission_path)
    private_gt = read_jsonl(private_gt_path)

    validate_submission(submission, private_gt, task='pointing')

    # __new__ bypasses __init__ entirely — no TSV disk read.
    ds = VANTAGE_2DPointing.__new__(VANTAGE_2DPointing)
    # evaluate() at line 3005 reads self.verbose; set it explicitly.
    ds.verbose = False
    ds.data = _build_synthetic_ds_data(private_gt)

    pred_df = _build_synthetic_pred_df(submission)

    cleanup_dir = None
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='vantage_sp_eval_')
        cleanup_dir = work_dir
    os.makedirs(work_dir, exist_ok=True)
    tmp_pred_path = osp.join(work_dir, '_sp_submission_pred.xlsx')
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
