"""Shared canonical-submission evaluator entrypoint.

Single public dispatch over the 8 per-task adapter strategies. The adapters
keep doing what they have always done — each is a task-specific strategy that
knows how to bridge a (submission JSONL, private GT JSONL) pair into a call
to the corresponding legacy ``dataset.evaluate(...)``. This module adds one
registry-based wrapper so callers can speak a single API:

    from vlmeval.dataset.utils.vantagebench.evaluator import evaluate_submission

    result = evaluate_submission(
        submission_path,
        private_gt_path,
        task='vqa',
        **judge_kwargs,
    )

Behavior is byte-identical to calling the per-task ``evaluate_<task>_submission``
function directly — this module only forwards.

Task-specific strategies that remain in their own modules (per the migration
plan's "small task strategies only where needed"):

- adapter_astro: synthesizes an on-disk KITTI labels directory before evaluate
- adapter_sot:   keys ``_gt_cache`` by canonical id (seq_dir.name)
- adapter_dvc / adapter_temporal: per-task synthetic ``self.data`` shape
- adapter_grounding: pred_df joins GT bboxes + image dims for the no-fallback path
- adapter_pointing: requires ``question_id`` column for category derivation
- adapter_vqa / adapter_event_verification: minimal column contracts

This module deliberately does not flatten any of the above into one
implementation. Doing so would couple eight independent legacy class
constructors into a single file and create score-drift risk for negligible
LOC savings.
"""

from __future__ import annotations

from typing import Any, Callable

from .adapter_astro import evaluate_astro_submission
from .adapter_dvc import evaluate_dvc_submission
from .adapter_event_verification import evaluate_event_verification_submission
from .adapter_grounding import evaluate_grounding_submission
from .adapter_pointing import evaluate_pointing_submission
from .adapter_sot import evaluate_sot_submission
from .adapter_temporal import evaluate_temporal_submission
from .adapter_vqa import evaluate_vqa_submission
from .validate import SubmissionValidationError

__all__ = [
    "SubmissionValidationError",
    "TASKS",
    "evaluate_submission",
]


# Short-key -> task-specific adapter strategy. Same short keys established in
# M2 (validate._TASK_REGISTRY) and M3 (emit._TASK_SPECS).
_TASK_REGISTRY: dict[str, Callable[..., Any]] = {
    'vqa': evaluate_vqa_submission,
    'event_verification': evaluate_event_verification_submission,
    'temporal': evaluate_temporal_submission,
    'dvc': evaluate_dvc_submission,
    'grounding': evaluate_grounding_submission,
    'pointing': evaluate_pointing_submission,
    'astro': evaluate_astro_submission,
    'sot': evaluate_sot_submission,
}


TASKS: tuple[str, ...] = tuple(sorted(_TASK_REGISTRY.keys()))


def evaluate_submission(
    submission_path: str,
    private_gt_path: str,
    *,
    task: str,
    work_dir: str | None = None,
    **judge_kwargs,
):
    """Run the legacy evaluator for ``task`` against a submission + private GT.

    Thin dispatcher: looks up the per-task adapter strategy in the registry
    and forwards every argument through.

    Parameters
    ----------
    submission_path : str
        Path to a submission JSONL (one canonical record per line).
    private_gt_path : str
        Path to a private GT JSONL.
    task : str
        Short task key. One of: 'vqa', 'event_verification', 'temporal',
        'dvc', 'grounding', 'pointing', 'astro', 'sot'.
    work_dir : str or None
        Working directory for the temporary synthetic prediction xlsx (and,
        for Astro, the synthetic KITTI labels directory). Defaults to a
        fresh ``tempfile.mkdtemp()`` directory the adapter cleans up itself.
    **judge_kwargs
        Forwarded verbatim to the underlying legacy ``dataset.evaluate(...)``.

    Returns
    -------
    The exact object the underlying ``evaluate_<task>_submission`` returns —
    typically a dict (VQA / Temporal / DVC / Grounding / Pointing / Astro /
    SOT) or DataFrame (EventVerification).

    Raises
    ------
    KeyError
        If ``task`` is not a known short key.
    SubmissionValidationError
        Propagated from the per-task validator if the submission is invalid.
    """
    if task not in _TASK_REGISTRY:
        raise KeyError(
            f"unknown task '{task}'; known tasks: {list(TASKS)}"
        )
    return _TASK_REGISTRY[task](
        submission_path,
        private_gt_path,
        work_dir=work_dir,
        **judge_kwargs,
    )
