"""Shared canonical-submission validator.

Single implementation of the validation policy used for every VANTAGE task.
Callers pin the task via the ``task`` short-key (e.g. ``task='vqa'``); the
report structure and error messages are identical across tasks, only the
``expected_task`` string in the record-shape check varies.

Policy (identical across tasks):
- missing prediction for any GT id   -> invalid
- duplicate id within submission     -> invalid
- unknown id (not in GT)             -> invalid
- task mismatch                      -> invalid
- missing assistant value            -> invalid
- malformed record                   -> invalid

The first failure class encountered raises ``SubmissionValidationError`` with
a structured report; the report enumerates every offending id at that level
so participants can fix all of them in one revision.
"""


__all__ = ["SubmissionValidationError", "validate_submission"]


class SubmissionValidationError(ValueError):
    """Raised when a submission cannot be scored.

    Single shared exception class across all canonical task validators.
    """

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


# Short-key -> canonical task string the submission/GT records carry.
# Matches what each id_rules.make_<task>_id and emit_<task> module emits.
_TASK_REGISTRY = {
    'vqa': 'video_qa',
    'event_verification': 'event_verification',
    'temporal': 'temporal_grounding',
    'dvc': 'dense_video_captioning',
    'grounding': 'referring_expressions',
    'pointing': 'spatial_pointing',
    'astro': 'object_localization',
    'sot': 'single_object_tracking',
}


def _assistant_value(record):
    """Return the assistant turn value or None if missing/malformed."""
    convs = record.get('conversations')
    if not isinstance(convs, list):
        return None
    for turn in convs:
        if not isinstance(turn, dict):
            continue
        if turn.get('from') == 'assistant':
            v = turn.get('value')
            if isinstance(v, str):
                return v
            return None
    return None


def validate_submission(submission, private_gt, *, task=None, expected_task=None):
    """Validate a canonical submission against private GT.

    Parameters
    ----------
    submission : list[dict]
        Records loaded from a submission JSONL.
    private_gt : list[dict]
        Records loaded from a private GT JSONL.
    task : str, optional
        Short task key (e.g. 'vqa'); resolved via the task registry to the
        canonical ``expected_task`` string the records must carry.
    expected_task : str, optional
        Canonical task string (e.g. 'video_qa'). Provide either ``task`` OR
        ``expected_task`` — not both.

    Returns
    -------
    dict
        Validation report on success.

    Raises
    ------
    SubmissionValidationError
        On any policy violation.
    """
    if task is not None and expected_task is not None:
        raise TypeError(
            "validate_submission: provide either 'task' or 'expected_task', not both"
        )
    if task is not None:
        if task not in _TASK_REGISTRY:
            raise KeyError(
                f"unknown task '{task}'; known tasks: {sorted(_TASK_REGISTRY.keys())}"
            )
        expected_task = _TASK_REGISTRY[task]
    if expected_task is None:
        raise TypeError(
            "validate_submission: must provide either 'task' or 'expected_task'"
        )

    report = {
        'task': expected_task,
        'submission_records': len(submission),
        'private_gt_records': len(private_gt),
        'malformed_records': [],
        'task_mismatches': [],
        'duplicates': [],
        'missing_from_submission': [],
        'extra_in_submission': [],
        'missing_assistant_value': [],
    }

    seen = {}
    for i, r in enumerate(submission):
        if not isinstance(r, dict):
            report['malformed_records'].append(
                {'position': i, 'reason': 'record is not a JSON object'}
            )
            continue
        rid = r.get('id')
        if not isinstance(rid, str) or not rid:
            report['malformed_records'].append(
                {'position': i, 'reason': 'missing or non-string id'}
            )
            continue
        task_value = r.get('task')
        if task_value != expected_task:
            report['task_mismatches'].append({'id': rid, 'task': task_value})
            continue
        if _assistant_value(r) is None:
            report['missing_assistant_value'].append(rid)
        if rid in seen:
            report['duplicates'].append(rid)
        else:
            seen[rid] = r

    if report['malformed_records']:
        raise SubmissionValidationError(
            f"Submission has {len(report['malformed_records'])} malformed record(s).",
            report,
        )
    if report['task_mismatches']:
        raise SubmissionValidationError(
            f"Submission has {len(report['task_mismatches'])} record(s) with "
            f"task != '{expected_task}'.",
            report,
        )
    if report['duplicates']:
        raise SubmissionValidationError(
            f"Submission has {len(report['duplicates'])} duplicate id(s).",
            report,
        )
    if report['missing_assistant_value']:
        raise SubmissionValidationError(
            f"Submission has {len(report['missing_assistant_value'])} record(s) "
            f"missing an assistant turn value.",
            report,
        )

    sub_ids = set(seen)
    gt_ids = set()
    for r in private_gt:
        if not isinstance(r, dict):
            continue
        rid = r.get('id')
        if isinstance(rid, str) and rid:
            gt_ids.add(rid)

    missing = sorted(gt_ids - sub_ids)
    extra = sorted(sub_ids - gt_ids)
    report['missing_from_submission'] = missing
    report['extra_in_submission'] = extra

    if missing:
        raise SubmissionValidationError(
            f"Submission is missing predictions for {len(missing)} id(s).",
            report,
        )
    if extra:
        raise SubmissionValidationError(
            f"Submission contains {len(extra)} id(s) not present in private GT.",
            report,
        )

    report['matched'] = sorted(gt_ids & sub_ids)
    return report
