"""VANTAGE canonical schema layer.

Runtime canonical interface for all 8 VANTAGE benchmark tasks:
``vqa``, ``event_verification``, ``temporal``, ``dvc``, ``grounding``,
``pointing``, ``astro``, ``sot``.

Three shared public entrypoints — one per concern — drive every task
through a registry on a small short-key:

- ``vlmeval.dataset.utils.vantagebench.validate.validate_submission``  — schema validation
- ``vlmeval.dataset.utils.vantagebench.emit.emit_submission``          — submission JSONL emission
- ``vlmeval.dataset.utils.vantagebench.evaluator.evaluate_submission`` — legacy-evaluator dispatch

Module layout:

- ``id_rules``        — canonical id synthesis (shared across all tasks)
- ``io``              — JSONL read/write helpers
- ``validate``        — shared validation policy + ``SubmissionValidationError``
- ``emit``            — shared emitter policy + per-task registry
- ``evaluator``       — registry-based dispatch over per-task adapters
- ``adapter_<task>``  — task-specific evaluator strategies (kept separate by
                       design; do not consolidate into ``evaluator.py``
                       without explicit approval — adapters encode legacy
                       dataset-class constructor quirks and synthetic-shape
                       knowledge that is direct score-drift risk if edited)

Offline private GT artifact generation lives outside this package, under
``tools/build_private_gt/``; it never imports from this module beyond
``id_rules`` and ``io``.

This ``__init__`` is deliberately import-side-effect free: the public
entrypoints are reachable via their submodules (e.g.
``from vlmeval.dataset.utils.vantagebench.evaluator import evaluate_submission``).
"""
