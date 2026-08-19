---
name: vantage-add-model
description: Use when the user wants to run a model that isn't in supported_VLM yet — write a wrapper class, register it, and smoke-test it.
---

Bring-your-own-model walkthrough: local (`vlmeval/vlm/`) or API-style (`vlmeval/api/`)
wrapper, registered in `vlmeval/config.py`, verified with a 1-sample run before trusting it.

## State detection

Check the model isn't already registered before writing anything:

```bash
python -c "from vlmeval.config import supported_VLM; print('<Name>' in supported_VLM)"
```

If `True`, this skill isn't needed — use `../vantage-model-config/SKILL.md` instead.

## Decision: local vs API wrapper — different base classes, different contracts

**Local (`vlmeval/vlm/`, subclass `BaseModel` from `vlmeval/vlm/base.py`)**:
- `generate_inner(self, message, dataset=None) -> str` — returns the raw answer string
  directly.
- Optionally override `use_custom_prompt(dataset) -> bool` + `build_prompt(line, dataset)` for
  dataset-specific prompt formatting; default `use_custom_prompt` returns `False`.
- Class attrs: `INTERLEAVE` (bool, whether the model accepts interleaved image/text turns),
  `allowed_types` (default `['text', 'image', 'video']`).
- Simple reference example: `vlmeval/vlm/molmo.py` — clean `__init__` (HF `from_pretrained` +
  processor), `use_custom_prompt`/`build_prompt` override pattern. Skim it, don't copy it
  verbatim — VANTAGE video tasks need `allowed_types` to include `'video'` and typically use
  `message_to_promptvideo`/`message_to_promptvideo_withrole` helpers already on `BaseModel`
  (see `vlmeval/vlm/base.py`) rather than the image-only helpers `molmo.py` uses.

**API-style (`vlmeval/api/`, subclass `BaseAPI` from `vlmeval/api/base.py`)** — a different
contract from `BaseModel`, easy to get wrong by copying the local pattern:
- `generate_inner(self, inputs, **kwargs) -> tuple[int, str, str]` — returns
  `(ret_code, answer, log)`, **not** a bare string. `ret_code == 0` means success.
- `__init__` should accept and forward `retry`, `wait`, `system_prompt`, `verbose`,
  `fail_msg` to `BaseAPI.__init__` via `super().__init__(**kwargs)`.
- Reference examples already in this repo, both real VANTAGE-relevant patterns: 
  `vlmeval/api/gpt.py` (`GPT4V` subclasses `OpenAIWrapper`, which implements the
  OpenAI-compatible `generate_inner` once — `GPT4V` itself only overrides `generate`) and
  `vlmeval/api/cosmos_reason.py` (`CosmosReason2` — closest existing template if the new
  model is also an OpenAI-compatible hosted endpoint; its `generate_inner` at line ~246
  returns the `(ret_code, answer, response)` tuple directly).

## Register it

1. Import the new class in the package `__init__.py`:
   - Local: add to `vlmeval/vlm/__init__.py` (plain `from .yourmodule import YourClass` — no
     `__all__` list to update there).
   - API: add to `vlmeval/api/__init__.py` **and** append the class name to its `__all__`
     list — unlike `vlm/__init__.py`, this file defines one and omitting the class from it can
     leave `from vlmeval.api import *` unable to see it.
2. **Gate: confirm before editing `vlmeval/config.py`** — it's a shared file every model
   entry lives in. Add one entry to the `supported_VLM` dict (or one of the dicts merged into
   it near the bottom of the file), following the existing `partial(...)` pattern:
   ```python
   'YourModel-Name': partial(YourClass, model_path='org/checkpoint', use_vllm=True),
   ```
   Match the style of neighboring entries (e.g. lines near `'Cosmos-Reason2-8B'`) — kwargs
   passed to `partial` become the class's default constructor args.

## Smoke test before trusting it

Use the smallest available dataset for a fast check —
`VANTAGE_VQA_8frame_200` (200-sample subset, seed 42) — or, for a true 1-sample check, run
with `--data VANTAGE_VQA_8frame_200 --mode infer` and inspect the first row of the output
`.xlsx` rather than waiting for the full 200:

```bash
python run.py --data VANTAGE_VQA_8frame_200 --model YourModel-Name --mode infer --verbose
```

Confirm: the process doesn't crash, `outputs/YourModel-Name/<eval_id>/*.xlsx` is written, and
the `prediction` column contains real text (not empty strings or a `fail_msg` placeholder like
`'Failed to obtain answer via API.'`). Only after this passes, move to a real run via
`../vantage-run/SKILL.md`.

## Failure handling

| Symptom | Fix |
|---|---|
| `assert 0, 'generate_inner not defined'` | Abstract method not overridden — implement `generate_inner` with the right signature for your base class. |
| API wrapper returns a bare string instead of failing loudly | You returned a `str` from `generate_inner` instead of the `(ret_code, answer, log)` tuple `BaseAPI` expects — check callers upstream break silently on this. |
| `ImportError` on `from vlmeval.api import *` | Class added to `vlmeval/api/__init__.py` imports but missing from its `__all__` list. |
| Model runs but every prediction is the API `fail_msg` | `ret_code != 0` on every call — check retry/auth/timeout kwargs reached the constructor. |

## Further reading

`vlmeval/vlm/base.py` and `vlmeval/api/base.py` in full for every helper method (image/video
message formatting, retry loop in `BaseModel.chat`); `README_VANTAGE.md` §1 for where VANTAGE
dataset code lives if the new model also needs dataset-side changes (out of scope here).
