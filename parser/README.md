# parser/

Organizer-only tools for compiling benchmark outputs into the leaderboard artifact.

**Participants do not need anything in this folder.**

---

## Files

### `get_all_outputs.py`

Walks the `outputs/` directory tree, finds all VANTAGE metric JSON files, and compiles them into a single `vlmevalkit_outputs.json` keyed by model name.

**Recognized metric files:**

| Filename pattern | Leaderboard key |
|-----------------|-----------------|
| `*Astro2D_metrics.json` | `astro` |
| `*VANTAGE_2DGrounding_metrics.json` | `grounding` |
| `*VANTAGE_2DPointing_results.json` | `pointing` |
| `*VANTAGE_DVC_metrics.json` | `dvc` |
| `*VANTAGE_EventVerification_acc.json` | `event_verification` |
| `*VANTAGE_Temporal_metrics.json` | `temporal` |
| `*VANTAGE_VQA_results.json` | `vqa` |
| `*model_config.json` | `config` |

```bash
cd parser
python get_all_outputs.py
# writes: parser/vlmevalkit_outputs.json
```

### `prepare_outputs_hf.py`

Reads `vlmevalkit_outputs.json` (from above) and an existing `hf/leaderboard.json`, extracts the primary metrics defined in the leaderboard schema, computes an overall score, and writes the updated `hf/leaderboard.json`.

```bash
cd parser
python prepare_outputs_hf.py
# updates: hf/leaderboard.json
```

After this, run `hf/up.py` to push the updated leaderboard to the HF Space.

---

## Full pipeline

```
outputs/                     ← run.py output directory
    │
    ▼
get_all_outputs.py
    │  vlmevalkit_outputs.json
    ▼
prepare_outputs_hf.py
    │  hf/leaderboard.json
    ▼
hf/up.py  →  HF Space (hf/app.py)
```
