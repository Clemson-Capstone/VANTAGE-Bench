"""
MMSB Leaderboard — Multi-Modal Scene Benchmark
================================================
UI-only HF Space. Reads a prepared leaderboard artifact (data/leaderboard.json)
produced by the Eval Repo. Does NOT recompute scores, validate submissions,
or define any benchmark logic. All benchmark truth lives in the Eval Repo.
"""

import json
from datetime import datetime

import gradio as gr
import pandas as pd

# ── Data contract ──────────────────────────────────────────────────────────────
# leaderboard.json is the single input artifact this Space consumes.
# The Eval Repo is responsible for generating and updating this file.
DATA_FILE = "data/leaderboard.json"

MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}

# Domain selector — present on all leaderboard tabs.
# In the Overall tab, selecting a domain re-ranks models by their pre-computed domain score.
# In task-specific tabs the selector is present for UI consistency; task ranking uses the
# task's primary metric regardless of domain selection.
DOMAIN_CHOICES = ["All", "Public Safety", "Transportation", "Warehouse"]
DOMAIN_KEY_MAP = {
    "All":            None,
    "Public Safety":  "public_safety",
    "Transportation": "transportation",
    "Warehouse":      "warehouse",
}

# ── Task definitions ──────────────────────────────────────────────────────────
TASKS = {
    "event_verification": {
        "tab_name":       "Event Verification",
        "description":    "Identify and classify events occurring in video footage across public safety and industrial contexts.",
        "primary_key":    "macro avg--f1-score",
        "primary_metric": "Macro F1",
        "columns": [
            ("macro avg--f1-score", "Macro F1"),
            ("accuracy",            "Accuracy"),
            ("macro avg--precision","Macro Prec."),
            ("macro avg--recall",   "Macro Recall"),
            ("weighted avg--f1-score","Weighted F1"),
        ],
    },
    "vqa": {
        "tab_name":       "Video Question Answering",
        "description":    "Answer natural language questions grounded in video content, requiring visual and temporal reasoning.",
        "primary_key":    "acc",
        "primary_metric": "Accuracy",
        "columns": [
            ("acc", "Accuracy"),
        ],
    },
    "temporal": {
        "tab_name":       "Temporal",
        "description":    "Localize and order events over time, including action duration and sequence comprehension.",
        "primary_key":    "iou",
        "primary_metric": "IoU",
        "columns": [
            ("iou",           "IoU"),
            ("precision@0.5", "Prec@0.5"),
        ],
    },
    "dvc": {
        "tab_name":       "Dense Video Captioning",
        "description":    "Generate dense, accurate textual descriptions of all events within a video segment.",
        "primary_key":    "SODA_c",
        "primary_metric": "SODA-c",
        "columns": [
            ("SODA_c",       "SODA-c"),
            ("BertScore_F1", "BERTScore"),
            ("IoU_F1",       "IoU F1"),
            ("mIoU",         "mIoU"),
        ],
    },
    "astro": {
        "tab_name":       "Astro2D",
        "description":    "Detecting and localizing humans in 2D scenes with high-definition bounding boxes.",
        "primary_key":    "f1_mIOU",
        "primary_metric": "F1 mIoU",
        "columns": [
            ("f1_mIOU",   "F1 mIoU"),
            ("f1",        "F1"),
            ("precision", "Precision"),
            ("recall",    "Recall"),
            ("valid_rate","Valid Rate"),
        ],
    },
    "grounding": {
        "tab_name":       "2D Referring Expressions",
        "description":    "Grounding natural language referring expressions to the correct object in 2D imagery.",
        "primary_key":    "Mean_IoU",
        "primary_metric": "Mean IoU",
        "columns": [
            ("Mean_IoU", "Mean IoU"),
            ("Acc@0.5",  "Acc@0.5"),
            ("Acc@0.25", "Acc@0.25"),
            ("valid_rate","Valid Rate"),
        ],
    },
    "pointing": {
        "tab_name":       "2D Spatial Pointing",
        "description":    "Pointing to the specific object referenced by language in 2D imagery.",
        "primary_key":    "acc",
        "primary_metric": "Accuracy",
        "columns": [
            ("acc", "Accuracy"),
        ],
    },
}

TASK_KEYS = list(TASKS.keys())
MAJORITY_THRESHOLD = len(TASK_KEYS) // 2 + 1 

# Overall table: task columns (key → header label)
OVERALL_TASK_COLS = [
    ("event_verification", "EV"),
    ("vqa",                "VQA"),
    ("temporal",           "Temp"),
    ("dvc",                "DVC"),
    ("astro",              "Astro"),
    ("grounding",          "Ground"),
    ("pointing",           "Point"),
]

# ── Loader ─────────────────────────────────────────────────────────────────────

def load_leaderboard() -> dict:
    """Load the leaderboard artifact. No processing or scoring happens here."""
    with open(DATA_FILE) as f:
        return json.load(f)


# ── Eligibility ────────────────────────────────────────────────────────────────

def is_overall_eligible(model: dict) -> bool:
    """True if model has results for enough tasks to appear in the Overall view."""
    covered = sum(1 for k in TASK_KEYS if k in model.get("task_results", {}))
    return covered >= MAJORITY_THRESHOLD


# ── Table builders ─────────────────────────────────────────────────────────────

def _fmt(val) -> str:
    """Format a numeric score for display."""
    if isinstance(val, (int, float)):
        return f"{val:.1f}"
    return "—"


def build_overall_df(
    models: list[dict], search: str = "", domain: str = "All"
) -> pd.DataFrame:
    """
    Build the Overall leaderboard table.
    Only includes models that pass the majority-task eligibility check.
    `domain` changes which pre-computed domain score is used for ranking
    (presentation only — scores come directly from the data file).
    """
    domain_key = DOMAIN_KEY_MAP.get(domain)

    def sort_key(m: dict) -> float:
        if domain_key is None:
            return m.get("overall_score") or 0.0
        return m.get("domain_scores", {}).get(domain_key, {}).get("score", 0.0)

    eligible = [m for m in models if is_overall_eligible(m)]
    ranked   = sorted(eligible, key=sort_key, reverse=True)

    rows = []
    for rank, m in enumerate(ranked, start=1):
        name = m["model_name"]
        if search and search.lower() not in name.lower():
            continue
        medal       = MEDALS.get(rank, "")
        rank_cell   = f"{rank} {medal}" if medal else str(rank)
        display     = f"{name} ☆" if m.get("is_baseline") else name
        score       = sort_key(m)
        tr          = m.get("task_results", {})

        row: dict = {
            "#":          rank_cell,
            "Model":      display,
            "Overall":    _fmt(score),
        }
        for task_key, col_name in OVERALL_TASK_COLS:
            pk  = TASKS[task_key]["primary_key"]
            val = tr.get(task_key, {}).get(pk)
            row[col_name] = _fmt(val)
        rows.append(row)

    return pd.DataFrame(rows)


def build_task_df(
    models: list[dict], task_key: str, search: str = ""
) -> pd.DataFrame:
    """
    Build a per-task leaderboard table.
    Includes every model that has results for `task_key`, ranked by
    that task's primary metric. Models without this task are excluded.
    """
    columns     = TASKS[task_key]["columns"]
    primary_key = TASKS[task_key]["primary_key"]
    task_models = [m for m in models if task_key in m.get("task_results", {})]
    ranked      = sorted(task_models,
                         key=lambda m: m["task_results"][task_key].get(primary_key, 0.0),
                         reverse=True)

    rows = []
    for rank, m in enumerate(ranked, start=1):
        name = m["model_name"]
        if search and search.lower() not in name.lower():
            continue
        medal     = MEDALS.get(rank, "")
        rank_cell = f"{rank} {medal}" if medal else str(rank)
        display   = f"{name} ☆" if m.get("is_baseline") else name
        task_r    = m["task_results"][task_key]

        row: dict = {"#": rank_cell, "Model": display}
        for i, (metric_key, col_name) in enumerate(columns):
            # Mark the first column (primary ranking metric) with ▲
            header = f"{col_name} ▲" if i == 0 else col_name
            row[header] = _fmt(task_r.get(metric_key))
        rows.append(row)

    return pd.DataFrame(rows)


def make_status_html(shown: int, total: int, last_updated: str = "") -> str:
    filtered = " (filtered)" if shown < total else ""
    noun     = "model" if shown == 1 else "models"
    left     = f"{shown} {noun} shown{filtered} · Click any row to view details"
    right    = f"Last updated: {last_updated}" if last_updated else ""
    return (
        f'<div class="lb-status" style="display:flex;justify-content:space-between;align-items:center;">'
        f"<span>{left}</span><span>{right}</span>"
        f"</div>"
    )


def build_task_subheader_html(task_key: str, n_models: int) -> str:
    task = TASKS[task_key]
    return (
        f'<div class="lb-task-subheader">'
        f'<div class="lb-task-title">'
        f'Task: <strong>{task["tab_name"]}</strong>'
        f'&nbsp;&nbsp;—&nbsp;&nbsp;'
        f'Ranked by <strong>{task["primary_metric"]} ▲</strong>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;'
        f'<span class="lb-task-count">{n_models} models</span>'
        f'</div>'
        f'<div class="lb-task-desc">{task["description"]}</div>'
        f"</div>"
    )


def build_overall_eligibility_html(n_eligible: int, n_total: int) -> str:
    n_partial = n_total - n_eligible
    partial_note = (
        f"<strong>{n_partial}</strong> model{'s' if n_partial != 1 else ''} "
        f"evaluated on fewer tasks appear only in the task-specific tabs."
    )
    return (
        f'<div class="lb-eligibility-note">'
        f'Showing <strong>{n_eligible}</strong> models with results for '
        f'<strong>≥{MAJORITY_THRESHOLD} of {len(TASK_KEYS)} tasks</strong>.'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;{partial_note}'
        f"</div>"
    )


def build_overall_context_html(n_eligible: int, n_total: int, n_baselines: int) -> str:
    """Three-line context block shown below the Overall status bar."""
    lines = [
        f"Overall scoring = macro-average of primary metrics across all {len(TASK_KEYS)} tasks.",
        "☆ = internal baseline model.",
        f"Overall-eligible = models with results for ≥{MAJORITY_THRESHOLD} of {len(TASK_KEYS)} tasks."
        f" (while models evaluated on <{MAJORITY_THRESHOLD} tasks appear only in the task-specific tabs).",
    ]
    items = "".join(f'<div class="lb-context-line">{l}</div>' for l in lines)
    return f'<div class="lb-context-block">{items}</div>'


# ── Model detail ───────────────────────────────────────────────────────────────

def format_model_detail(
    model: dict, rank: int, total: int, context_task: str | None = None
) -> str:
    """
    Render model detail as Markdown.
    context_task: if set, highlights that task's metrics at the top
                  (used when the detail is opened from a task-specific tab).
    """
    meta    = model.get("metadata", {})
    links   = meta.get("links", {})
    name    = model["model_name"]
    params  = model.get("params", "—")
    medal   = MEDALS.get(rank, "")
    btag    = "  `☆ baseline`" if model.get("is_baseline") else ""
    tr      = model.get("task_results", {})
    n_tasks = len(tr)

    lines = [
        f"### {medal} {name}{btag}",
        f"Rank **#{rank}** of {total} · {params} params",
        "",
    ]

    # External links
    link_parts = []
    if links.get("model_repo"):
        link_parts.append(f"[Model Repository ↗]({links['model_repo']})")
    if links.get("results_file"):
        link_parts.append(f"[Download Results ↗]({links['results_file']})")
    if link_parts:
        lines.append("  ".join(link_parts))
        lines.append("")

    # Coverage indicator
    eligible_str = (
        f"✓ Overall-eligible ({n_tasks}/{len(TASK_KEYS)} tasks)"
        if is_overall_eligible(model)
        else f"Task-only — {n_tasks}/{len(TASK_KEYS)} tasks covered (below eligibility threshold)"
    )
    lines += [f"*{eligible_str}*", ""]

    if context_task and context_task in tr:
        # Lead with the task the user clicked from
        task_info   = TASKS[context_task]
        task_r      = tr[context_task]
        primary_key = task_info["primary_key"]
        primary     = task_r.get(primary_key)
        lines += [
            f"**{task_info['tab_name']}  —  {task_info['primary_metric']}: {_fmt(primary)}**",
            "",
        ]
        for metric_key, col_name in task_info["columns"][1:]:
            lines.append(f"- {col_name}: **{_fmt(task_r.get(metric_key))}**")
        lines.append("")

        # Any other tasks the model has
        other = [(k, v) for k, v in tr.items() if k != context_task]
        if other:
            lines.append("**Other Available Task Results**")
            for tk, tr2 in other:
                t   = TASKS[tk]
                val = tr2.get(t["primary_key"])
                lines.append(f"- {t['tab_name']}: **{_fmt(val)}** *({t['primary_metric']})*")
            lines.append("")

    else:
        # Overall detail view
        overall = model.get("overall_score")
        if overall is not None:
            lines += [
                f"**Overall Score: {_fmt(overall)}**",
                f"*Macro-average of {n_tasks} task primary metrics*",
                "",
            ]

        lines.append("**Per-Task Primary Metrics**")
        for tk in TASK_KEYS:
            t = TASKS[tk]
            if tk in tr:
                val = tr[tk].get(t["primary_key"])
                lines.append(f"- {t['tab_name']}: **{_fmt(val)}** *({t['primary_metric']})*")
            else:
                lines.append(f"- {t['tab_name']}: —")
        lines.append("")

        # Domain scores (only for Overall-eligible models)
        ds = model.get("domain_scores", {})
        if ds:
            lines.append("**Domain Scores (Overall)**")
            domain_labels = {
                "public_safety":  "Public Safety",
                "transportation": "Transportation",
                "warehouse":      "Warehouse",
            }
            for dk, dlabel in domain_labels.items():
                val = ds.get(dk, {}).get("score")
                lines.append(f"- {dlabel}: **{_fmt(val)}**")
            lines.append("")

    # Evaluation metadata
    if meta:
        lines.append("**Evaluation Metadata**")
        meta_fields = [
            ("evaluated_at",  "Evaluated",  False),
            ("eval_hardware", "Hardware",   True),
        ]
        for key, label, mono in meta_fields:
            if key in meta:
                val = f"`{meta[key]}`" if mono else meta[key]
                lines.append(f"- {label}: {val}")

    if meta.get("notes"):
        lines += ["", "**Notes**", meta["notes"]]

    return "\n".join(lines)


# ── Header / footer / infobar HTML ─────────────────────────────────────────────

def build_header_html(benchmark: dict, meta_line: str = "") -> str:
    lk = benchmark.get("links", {})

    def nav_btn(key, label):
        url = lk.get(key) or "#"
        return (
            f'<a href="{url}" target="_blank" class="lb-nav-link">'
            f'{label} <span style="font-size:0.7em">↗</span></a>'
        )

    nav = "".join([
        nav_btn("eval_repo",    "GitHub Repo"),
        nav_btn("dataset_repo", "Dataset"),
        nav_btn("docs",         "Docs"),
    ])

    meta_html = (
        f'<p style="margin:0 0 8px 0; font-size:0.75rem; '
        f'color:var(--body-text-color-subdued); font-family:ui-monospace,monospace;">'
        f'{meta_line}</p>'
    ) if meta_line else ""

    return f"""
<div style="text-align:center; padding-bottom:14px;
            border-bottom:1px solid var(--border-color-primary); margin-bottom:4px;">
  <h1 style="margin:0 0 4px 0; font-size:2rem; font-weight:700; line-height:1.3;
             background:linear-gradient(90deg,#22c55e,#16a34a);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;
             color:#22c55e;">
    Multi-Modal Scene Bench Leaderboard
  </h1>
  <p style="margin:0 0 10px 0; font-size:0.88rem; color:var(--body-text-color-subdued);">
    Video Understanding Evaluation for Vision-Language Models
  </p>
  {meta_html}<div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center; justify-content:center;">
    {nav}
  </div>
</div>
"""


def build_infobar_html(benchmark: dict, artifact_generated_at: str, n_eligible: int, n_total: int, n_baselines: int) -> str:
    dv  = benchmark.get("dataset_version", "—")
    ev  = benchmark.get("evaluator_version", "—")
    ver = benchmark.get("version", "v0.1")
    return (
        f'<div class="lb-infobar">'
        f'Tasks ({ver}): <strong>{len(TASK_KEYS)}</strong>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;Models: <strong>{n_total}</strong>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;Overall-eligible: <strong>{n_eligible}</strong>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;Baselines: <strong>{n_baselines}</strong>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;Dataset: <strong>{dv}</strong>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;Evaluator: <strong>{ev}</strong>'
        f'<span style="float:right;">Generated: <strong>{artifact_generated_at}</strong></span>'
        f'</div>'
    )


def build_footer_html(benchmark: dict) -> str:
    lk      = benchmark.get("links", {})
    version = benchmark.get("version", "v0.1")
    lic     = benchmark.get("license", "MIT")

    def lnk(key, label):
        url = lk.get(key) or "#"
        return f'<a href="{url}" target="_blank">{label} ↗</a>'

    parts = " &nbsp;|&nbsp; ".join([
        lnk("eval_repo",    "GitHub Repository"),
        lnk("dataset_repo", "Dataset"),
    ])
    return (
        f'<div class="lb-footer">'
        f'MMSB — Multi-Modal Scene Benchmark {version}'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;License: {lic}'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;{parts}'
        f'</div>'
    )


# ── About page ─────────────────────────────────────────────────────────────────

def build_about_md(benchmark: dict) -> str:
    lk = benchmark.get("links", {})

    def lnk(key, label):
        url = lk.get(key) or "#"
        return f"[{label} ↗]({url})"

    cite = benchmark.get("citation", (
        "@inproceedings{mmsb2026,\n"
        "  title={MMSB: Multi-Modal Scene Benchmark for Video Understanding\n"
        "         in Vision-Language Models},\n"
        "  author={Author et al.},\n"
        "  booktitle={Conference},\n"
        "  year={2026}\n}"
    ))

    dv       = benchmark.get("dataset_version", "—")
    ev_ver   = benchmark.get("evaluator_version", "—")
    protocol = benchmark.get("evaluation_protocol", "—")
    metric   = benchmark.get("primary_metric", "—")
    hw       = benchmark.get("hardware_baseline", "—")
    license_ = benchmark.get("license", "—")
    eval_url = lk.get("eval_repo") or "#"

    return f"""## Benchmark Description
<div style="max-width:700px;">
**MMSB (Multi-Modal Scene Benchmark)** is designed to evaluate vision-language models for
real-world video understanding in intelligent infrastructure systems.
Inspired by real-world intelligent video analytics deployments
across smart cities, transportation networks, and industrial environments, MMSB measures
how effectively modern multimodal models can interpret and reason over video surveillance data.
The goal is to provide a standardized evaluation suite for video reasoning tasks relevant to
real-world AI applications, enabling rigorous comparison across models on a common set of tasks
and datasets.
</div>
---
## Tasks
| Task | Primary Metric | Description |
|---|---|---|
| **Event Verification** | F1 Score | Identifying and classifying events in video footage across public safety and industrial contexts |
| **Video Question Answering** | Accuracy | Answering natural language questions grounded in video content |
| **Temporal** | mAP@tIoU | Localizing and ordering events over time |
| **Dense Video Captioning** | SODA-c | Generating dense textual descriptions of all events in a video segment |
| **2D Object Localization (ITS OD-style)** | mAP@IoU | Detecting and localizing objects in 2D scenes with bounding boxes |
| **2D Referring Expressions (RefDrone-style)** | Acc@IoU | Grounding natural language referring expressions to the correct object in 2D imagery |
| **2D Spatial Pointing (BLINK-style)** | Pointing Accuracy | Pointing to the specific object referenced by language in 2D imagery |
---
## Overall Score and Eligibility
The **Overall leaderboard** includes only models evaluated on ≥{MAJORITY_THRESHOLD} of {len(TASK_KEYS)} tasks.
The Overall score is the macro-average of each task's primary metric (all normalized to a 0–100 scale).
Models with partial coverage appear only in the task-specific leaderboards where they have results.
---
## Domains
Each evaluation covers three infrastructure domains:
| Domain | Description |
|---|---|
| **Public Safety** | Urban surveillance, crowd monitoring, incident detection |
| **Transportation** | Traffic management, vehicle tracking, road safety |
| **Warehouse** | Industrial automation, inventory, safety compliance |
The **Domain** filter in the Overall leaderboard re-ranks models using their pre-computed domain-specific scores.
---
## Reproducibility
To evaluate your model on MMSB, visit our [GitHub repository ↗]({eval_url}).
| Field | Value |
|---|---|
| Dataset Version | {dv} |
| Evaluator Version | {ev_ver} |
| Evaluation Protocol | {protocol} |
| Primary Metric | {metric} |
| Hardware Baseline | {hw} |
| License | {license_} |
---
## Resources
- {lnk("eval_repo", "GitHub Repository")} — Official evaluation scripts and scoring pipeline
- {lnk("dataset_repo", "Dataset")} — Download and explore the MMSB dataset
- {lnk("docs", "Documentation")} — Task definitions, data format specs, and submission guidelines
---
## Citation
If you use MMSB in your research, please cite:
```bibtex
{cite}
```
"""


# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
/* ── Layout ── */
.gradio-container {
    width: 96vw !important;
    max-width: 1800px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 4px !important;
    padding-right: 4px !important;
    box-sizing: border-box !important;
}
.gradio-container > .main,
.gradio-container > div > .main {
    max-width: 100% !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}
:root {
    --block-padding: 6px !important;
}
/* ── Nav link buttons ── */
.lb-nav-link {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 5px 11px;
    border: 1px solid var(--border-color-primary);
    border-radius: 6px;
    font-size: 0.8rem;
    text-decoration: none;
    color: var(--body-text-color);
    background: var(--background-fill-primary);
    white-space: nowrap;
}
.lb-nav-link:hover {
    background: var(--background-fill-secondary);
}
/* ── Metadata info bar ── */
.lb-infobar {
    font-size: 0.78rem;
    font-family: ui-monospace, monospace;
    color: var(--body-text-color-subdued);
    padding: 8px 2px;
    border-bottom: 1px solid var(--border-color-primary);
    margin-bottom: 10px;
    overflow: hidden;
}
/* ── Status line ── */
.lb-status {
    font-size: 0.8rem;
    color: var(--body-text-color-subdued);
    margin: 2px 0 0 2px;
    font-style: italic;
}
/* ── Overall eligibility note ── */
.lb-eligibility-note {
    font-size: 0.82rem;
    color: var(--body-text-color-subdued);
    margin: 6px 0 8px 0;
    padding: 7px 14px;
    border-left: 3px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
    border-radius: 0 6px 6px 0;
    line-height: 1.5;
}
/* ── Overall score computation note (sits between eligibility note and table) ── */
.lb-score-note {
    font-size: 0.8rem;
    color: var(--body-text-color-subdued);
    margin: 0 0 10px 0;
    padding: 5px 14px;
    font-family: ui-monospace, monospace;
    border-left: 3px solid transparent;  /* aligns visually with eligibility note */
}
/* ── Per-task sub-header ── */
.lb-task-subheader {
    font-size: 0.82rem;
    color: var(--body-text-color-subdued);
    margin: 6px 0 12px 0;
    padding: 8px 14px;
    border-left: 3px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
    border-radius: 0 6px 6px 0;
    line-height: 1.5;
}
/* Identity line: task name + ranking metric — slightly more prominent */
.lb-task-title {
    font-size: 0.84rem;
    color: var(--body-text-color);
}
.lb-task-count {
    color: var(--body-text-color-subdued);
    font-weight: normal;
}
/* Description line: clearly secondary to the identity line */
.lb-task-desc {
    margin-top: 3px;
    font-style: italic;
    font-size: 0.79rem;
}
/* ── Model detail modal overlay ── */
#detail-modal {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    width: min(620px, 92vw) !important;
    max-height: 82vh !important;
    overflow-y: auto !important;
    background: var(--background-fill-primary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.22) !important;
    padding: 24px !important;
    z-index: 1000 !important;
}
.detail-header {
    border-bottom: 1px solid var(--border-color-primary);
    padding-bottom: 8px;
    margin-bottom: 8px;
}
/* ── Overall context block (below status bar) ── */
.lb-context-block {
    margin: 8px 0 0 2px;
}
.lb-context-line {
    font-size: 0.78rem;
    color: var(--body-text-color-subdued);
    font-style: italic;
    line-height: 1.7;
}
/* ── Search & Domain form controls (Overall + all per-task tabs) ── */
[id^="search-task-"],
[id^="domain-task-"] {
    border: 1.5px solid rgba(0, 0, 0, 0.18) !important;
    border-radius: 7px !important;
    padding: 10px 14px 12px 14px !important;
    background: var(--background-fill-primary) !important;
    box-shadow: none !important;
}
[id^="search-task-"] .block,
[id^="domain-task-"] .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}
[id^="search-task-"] label,
[id^="domain-task-"] label {
    display: block !important;
    margin-bottom: 6px !important;
}
[id^="search-task-"] input,
[id^="search-task-"] textarea,
[id^="domain-task-"] input,
[id^="domain-task-"] .wrap {
    background: var(--background-fill-primary) !important;
    border: 1px solid rgba(0, 0, 0, 0.12) !important;
    border-radius: 5px !important;
    box-shadow: none !important;
}
/* ── Footer ── */
.lb-footer {
    font-size: 0.78rem;
    color: var(--body-text-color-subdued);
    border-top: 1px solid var(--border-color-primary);
    padding-top: 10px;
    margin-top: 24px;
    overflow: hidden;
}
.lb-footer a {
    color: var(--body-text-color-subdued);
    text-decoration: none;
}
.lb-footer a:hover {
    color: var(--body-text-color);
    text-decoration: underline;
}
"""

_BACKDROP_HTML = (
    '<div style="position:fixed;inset:0;background:rgba(0,0,0,0.35);'
    'z-index:999;pointer-events:none;"></div>'
)


# ── Load data once at startup ──────────────────────────────────────────────────

_data          = load_leaderboard()
benchmark_meta = _data["benchmark"]
models_raw     = _data["results"]    # artifact key: "results" (list of eval result entries)
generated_at   = _data.get("generated_at", "")

model_map      = {m["model_name"]: m for m in models_raw}
total_models   = len(models_raw)
n_eligible     = sum(1 for m in models_raw if is_overall_eligible(m))
n_baselines    = sum(1 for m in models_raw if m.get("is_baseline"))

# Parse ISO 8601 timestamp or plain date from the artifact's generated_at field.
LAST_UPDATED = generated_at
for _fmt_str in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
    try:
        LAST_UPDATED = datetime.strptime(generated_at, _fmt_str).strftime("%b %-d, %Y")
        break
    except ValueError:
        pass

# Pre-build all static content and DataFrames once at startup.
_meta_line = (
    f"Models: {total_models}"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;Overall-eligible: {n_eligible}"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;Baselines: {n_baselines}"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;Last Updated: {LAST_UPDATED}"
)
HEADER_HTML           = build_header_html(benchmark_meta, meta_line=_meta_line)
OVERALL_CONTEXT_HTML  = build_overall_context_html(n_eligible, total_models, n_baselines)
FOOTER_HTML           = build_footer_html(benchmark_meta)
ABOUT_MD              = build_about_md(benchmark_meta)

base_overall_df  = build_overall_df(models_raw)

task_base_dfs: dict[str, pd.DataFrame] = {
    tk: build_task_df(models_raw, tk) for tk in TASK_KEYS
}
task_n_models: dict[str, int] = {
    tk: sum(1 for m in models_raw if tk in m.get("task_results", {})) for tk in TASK_KEYS
}


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="MMSB Leaderboard") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── Overall tab ────────────────────────────────────────────────────────
        with gr.Tab("Overall"):
            with gr.Row():
                search_overall = gr.Textbox(
                    placeholder="Search by model name…",
                    label="Model Search",
                    show_label=True,
                    max_lines=1,
                    scale=1,
                    min_width=200,
                    elem_id="search-task-overall",
                )
                domain_dd = gr.Dropdown(
                    choices=DOMAIN_CHOICES,
                    value="All",
                    label="Domain",
                    scale=1,
                    min_width=180,
                    elem_id="domain-task-overall",
                )
                gr.HTML("", scale=3)

            overall_cur_df = gr.State(value=base_overall_df)
            overall_table  = gr.Dataframe(value=base_overall_df, interactive=False, wrap=False)
            overall_status = gr.HTML(
                make_status_html(n_eligible, n_eligible, LAST_UPDATED)
            )
            gr.HTML(OVERALL_CONTEXT_HTML)

        # ── Per-task tabs ──────────────────────────────────────────────────────
        # Each task tab is self-contained: search box + task-specific table.
        task_components: dict[str, tuple] = {}

        for _task_key in TASK_KEYS:
            _task = TASKS[_task_key]
            with gr.Tab(_task["tab_name"]):
                gr.HTML(build_task_subheader_html(_task_key, task_n_models[_task_key]))
                with gr.Row():
                    _search = gr.Textbox(
                        placeholder="Search by model name…",
                        label="Model Search",
                        show_label=True,
                        max_lines=1,
                        scale=1,
                        min_width=200,
                        elem_id=f"search-task-{_task_key}",
                    )
                    _domain_dd = gr.Dropdown(
                        choices=DOMAIN_CHOICES,
                        value="All",
                        label="Domain",
                        scale=1,
                        min_width=180,
                        elem_id=f"domain-task-{_task_key}",
                    )
                    gr.HTML("", scale=3)

                _cur_df = gr.State(value=task_base_dfs[_task_key])
                _table  = gr.Dataframe(
                    value=task_base_dfs[_task_key], interactive=False, wrap=False
                )
                _status = gr.HTML(
                    make_status_html(task_n_models[_task_key], total_models, LAST_UPDATED)
                )
                task_components[_task_key] = (_search, _domain_dd, _cur_df, _table, _status)

        # ── About tab ──────────────────────────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown(ABOUT_MD)

    # Shared modal — fixed-position overlay, works across all tabs
    backdrop_html = gr.HTML(value=_BACKDROP_HTML, visible=False)
    with gr.Column(visible=False, elem_id="detail-modal") as detail_col:
        with gr.Row(elem_classes="detail-header"):
            gr.Markdown("#### Model Detail")
            close_btn = gr.Button("✕", size="sm", scale=0, min_width=36)
        detail_md = gr.Markdown(value="")

    gr.HTML(FOOTER_HTML)

    # ── Event handlers ────────────────────────────────────────────────────────

    # Overall tab
    def on_overall_filter(query: str, domain: str):
        df     = build_overall_df(models_raw, search=query, domain=domain)
        status = make_status_html(len(df), n_eligible, LAST_UPDATED)
        return df, df, status, gr.update(visible=False), gr.update(visible=False)

    search_overall.input(
        fn=on_overall_filter,
        inputs=[search_overall, domain_dd],
        outputs=[overall_table, overall_cur_df, overall_status, backdrop_html, detail_col],
    )
    domain_dd.change(
        fn=on_overall_filter,
        inputs=[search_overall, domain_dd],
        outputs=[overall_table, overall_cur_df, overall_status, backdrop_html, detail_col],
    )

    def on_overall_row_select(evt: gr.SelectData, cur_df: pd.DataFrame, domain: str):
        row_idx    = evt.index[0]
        raw_name   = cur_df.iloc[row_idx]["Model"]
        clean_name = raw_name.replace(" ☆", "").strip()
        model      = model_map.get(clean_name)
        if model is None:
            return gr.update(visible=True), gr.update(visible=True), "*(Model not found.)*"
        rank_raw = str(cur_df.iloc[row_idx]["#"]).split()[0]
        rank     = int(rank_raw) if rank_raw.isdigit() else 0
        detail   = format_model_detail(model, rank, n_eligible, context_task=None)
        return gr.update(visible=True), gr.update(visible=True), detail

    overall_table.select(
        fn=on_overall_row_select,
        inputs=[overall_cur_df, domain_dd],
        outputs=[backdrop_html, detail_col, detail_md],
    )

    # Per-task tabs — wire handlers via closures
    def _make_task_filter(tk: str):
        n = task_n_models[tk]
        def handler(query: str, domain: str):  # domain accepted for UI consistency; ignored in ranking
            df     = build_task_df(models_raw, tk, search=query)
            status = make_status_html(len(df), n, LAST_UPDATED)
            return df, df, status, gr.update(visible=False), gr.update(visible=False)
        return handler

    def _make_task_row_select(tk: str):
        n = task_n_models[tk]
        def handler(evt: gr.SelectData, cur_df: pd.DataFrame):
            row_idx    = evt.index[0]
            raw_name   = cur_df.iloc[row_idx]["Model"]
            clean_name = raw_name.replace(" ☆", "").strip()
            model      = model_map.get(clean_name)
            if model is None:
                return gr.update(visible=True), gr.update(visible=True), "*(Model not found.)*"
            rank_raw = str(cur_df.iloc[row_idx]["#"]).split()[0]
            rank     = int(rank_raw) if rank_raw.isdigit() else 0
            detail   = format_model_detail(model, rank, n, context_task=tk)
            return gr.update(visible=True), gr.update(visible=True), detail
        return handler

    for _task_key, (_search, _domain_dd, _cur_df, _table, _status) in task_components.items():
        _search.input(
            fn=_make_task_filter(_task_key),
            inputs=[_search, _domain_dd],
            outputs=[_table, _cur_df, _status, backdrop_html, detail_col],
        )
        _domain_dd.change(
            fn=_make_task_filter(_task_key),
            inputs=[_search, _domain_dd],
            outputs=[_table, _cur_df, _status, backdrop_html, detail_col],
        )
        _table.select(
            fn=_make_task_row_select(_task_key),
            inputs=[_cur_df],
            outputs=[backdrop_html, detail_col, detail_md],
        )

    # Close modal
    close_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=False)),
        outputs=[backdrop_html, detail_col],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
