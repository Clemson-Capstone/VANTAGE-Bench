"""Shared canonical-submission emitter.

One implementation of the submission-JSONL emission policy that every VANTAGE
task needed. The 8 per-task ``emit_<task>.py`` modules now thin-wrap this
entrypoint; they only pin the task short-key and (for SOT) pass through the
``dataset`` kwarg.

Policy (identical across tasks except where noted):
- never raise — any exception is converted to a ``warnings.warn`` and the
  function returns silently. The legacy xlsx remains the authoritative output.
- missing required columns -> warn and return.
- iterate ``meta_df.iterrows()`` in order (no sort).
- build a record per row: ``{id, task, conversations: [assistant], metadata}``.
- write to ``out_path`` via ``write_jsonl``.

Task-specific bits captured by the registry:
- ``display_name``: human-readable label used in warning prefixes.
- ``task``: canonical task string written into each record.
- ``required_cols``: minimum DataFrame columns needed.
- ``id_resolver(row, dataset) -> Optional[str]``: returns the canonical id,
  or ``None`` to skip this row (only SOT uses skip-on-None).
- ``needs_dataset``: SOT only — emit is aborted if ``dataset`` is None.
"""

from __future__ import annotations

import os.path as osp
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

from .id_rules import (
    make_astro_id,
    make_dvc_id,
    make_event_verification_id,
    make_grounding_id,
    make_pointing_id,
    make_sot_id,
    make_temporal_id,
    make_vqa_id,
)
from .io import write_jsonl


__all__ = ["emit_submission"]


@dataclass(frozen=True)
class EmitSpec:
    """Static description of one task's emission policy."""

    short_key: str
    display_name: str
    task: str
    required_cols: frozenset
    id_resolver: Callable[[object, object], Optional[str]]
    needs_dataset: bool = False


# ---------------------------------------------------------------------------
# Per-task id resolvers
# ---------------------------------------------------------------------------

def _resolve_sot_id(row, dataset) -> Optional[str]:
    """Map a SOT row's legacy enumeration-int index -> seq_dir.name.

    The dataset's _gt_cache is keyed by the same enum int the row carries
    in its 'index' column. cache['seq_dir'] is the absolute path; we take
    basename() to get seq_dir.name.

    Returns None if the lookup fails (the caller skips the row).
    """
    if dataset is None:
        return None
    cache = getattr(dataset, '_gt_cache', None)
    if cache is None:
        return None
    legacy_index = row['index']
    entry = cache.get(legacy_index)
    if entry is None:
        try:
            entry = cache.get(int(legacy_index))
        except (TypeError, ValueError):
            entry = None
    if entry is None:
        return None
    seq_dir = entry.get('seq_dir', '')
    if not seq_dir:
        return None
    return make_sot_id(osp.basename(str(seq_dir).rstrip('/')))


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

_TASK_SPECS: dict[str, EmitSpec] = {
    'vqa': EmitSpec(
        short_key='vqa',
        display_name='VANTAGE_VQA',
        task='video_qa',
        required_cols=frozenset({'index', 'video', 'prediction'}),
        id_resolver=lambda row, _ds: make_vqa_id(row['video'], row['index']),
    ),
    'event_verification': EmitSpec(
        short_key='event_verification',
        display_name='VANTAGE_EventVerification',
        task='event_verification',
        required_cols=frozenset({'index', 'video', 'prediction'}),
        id_resolver=lambda row, _ds: make_event_verification_id(
            row['video'], row['index']
        ),
    ),
    'temporal': EmitSpec(
        short_key='temporal',
        display_name='VANTAGE_Temporal',
        task='temporal_grounding',
        required_cols=frozenset({'index', 'video', 'prediction'}),
        id_resolver=lambda row, _ds: make_temporal_id(row['video'], row['index']),
    ),
    'dvc': EmitSpec(
        short_key='dvc',
        display_name='VANTAGE_DVC',
        task='dense_video_captioning',
        required_cols=frozenset({'index', 'video', 'prediction'}),
        id_resolver=lambda row, _ds: make_dvc_id(row['video'], row['index']),
    ),
    'grounding': EmitSpec(
        short_key='grounding',
        display_name='VANTAGE_2DGrounding',
        task='referring_expressions',
        required_cols=frozenset({'index', 'image_filename', 'prediction'}),
        id_resolver=lambda row, _ds: make_grounding_id(
            row['image_filename'], row['index']
        ),
    ),
    'pointing': EmitSpec(
        short_key='pointing',
        display_name='VANTAGE_2DPointing',
        task='spatial_pointing',
        required_cols=frozenset({'index', 'image_path', 'prediction'}),
        id_resolver=lambda row, _ds: make_pointing_id(
            row['image_path'], row['index']
        ),
    ),
    'astro': EmitSpec(
        short_key='astro',
        display_name='Astro2D',
        task='object_localization',
        required_cols=frozenset({'index', 'image_filename', 'prediction'}),
        id_resolver=lambda row, _ds: make_astro_id(
            row['image_filename'], row['index']
        ),
    ),
    'sot': EmitSpec(
        short_key='sot',
        display_name='VANTAGE_SOT',
        task='single_object_tracking',
        required_cols=frozenset({'index', 'prediction'}),
        id_resolver=_resolve_sot_id,
        needs_dataset=True,
    ),
}


# ---------------------------------------------------------------------------
# Shared entrypoint
# ---------------------------------------------------------------------------

def emit_submission(meta_df, model_name, out_path, *, task, dataset=None):
    """Emit a canonical submission JSONL for ``task`` from a meta DataFrame.

    Parameters
    ----------
    meta_df : pandas.DataFrame
        The post-inference DataFrame. Required columns depend on ``task``
        (see ``EmitSpec.required_cols``).
    model_name : str
        Model name passed through from the inference driver.
    out_path : str
        Absolute output path for the submission JSONL.
    task : str
        Short task key (one of: 'vqa', 'event_verification', 'temporal',
        'dvc', 'grounding', 'pointing', 'astro', 'sot').
    dataset : object, optional
        Required only for SOT (the canonical id requires lookup into
        ``dataset._gt_cache``). Ignored for all other tasks.

    On any exception this function warns and returns; it never raises.
    """
    if task not in _TASK_SPECS:
        warnings.warn(
            f"canonical emit: unknown task '{task}'. "
            f"Known: {sorted(_TASK_SPECS.keys())}. Legacy xlsx is unaffected."
        )
        return

    spec = _TASK_SPECS[task]

    try:
        missing = spec.required_cols - set(meta_df.columns)
        if missing:
            warnings.warn(
                f"{spec.display_name} submission JSONL emit skipped: "
                f"DataFrame is missing columns {sorted(missing)}. "
                f"Legacy xlsx is unaffected."
            )
            return

        if spec.needs_dataset and dataset is None:
            warnings.warn(
                f"{spec.display_name} submission JSONL emit skipped: dataset "
                f"parameter not provided. The canonical id (seq_dir.name) "
                f"must be resolved via the dataset's _gt_cache; pass "
                f"dataset=... to fix. Legacy xlsx is unaffected."
            )
            return

        records = []
        skipped = 0
        for _, row in meta_df.iterrows():
            canonical_id = spec.id_resolver(row, dataset)
            if canonical_id is None:
                skipped += 1
                continue
            records.append({
                "id": canonical_id,
                "task": spec.task,
                "conversations": [
                    {"from": "assistant", "value": str(row['prediction'])},
                ],
                "metadata": {
                    "model": model_name,
                    "extra": {},
                },
            })

        if spec.needs_dataset and skipped:
            warnings.warn(
                f"{spec.display_name} submission JSONL emit: skipped "
                f"{skipped} row(s) with unresolvable canonical id "
                f"(no matching _gt_cache entry)."
            )

        if spec.needs_dataset and not records:
            warnings.warn(
                f"{spec.display_name} submission JSONL emit: no records "
                f"produced (empty meta or all rows unresolvable). Legacy "
                f"xlsx is unaffected."
            )
            return

        write_jsonl(records, out_path)
    except Exception as e:  # noqa: BLE001 - intentional broad catch; never propagate
        warnings.warn(
            f"{spec.display_name} submission JSONL emit failed: "
            f"{type(e).__name__}: {e}. Legacy xlsx is unaffected and "
            f"remains the authoritative artifact."
        )
