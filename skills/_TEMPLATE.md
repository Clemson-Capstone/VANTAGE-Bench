---
name: vantage-<slug>
description: <One line, <=25 words, phrased as "Use when ...". This is the only text loaded unconditionally into every agent's context — every word must earn its place.>
---

<!--
House style for VANTAGE-Bench skills. Delete this comment block in real skills.

HARD LIMITS
- SKILL.md body (below the frontmatter): <= 120 lines, <= ~900 words.
- No prose that restates a repo README/doc. Link to it instead.
- No content duplicated across skills or between SKILL.md and reference.md.
  Shared facts (dataset keys, pillar map, id formats) live ONLY in
  skills/reference/tasks-and-pillars.md — link, don't copy.
- Every command must be traceable to an existing repo file (run.py,
  run_lmudata.py, package_submission.py, scripts/*.py). Never invent a flag.
- Machine-checkable work is a script call (`python scripts/x.py --json`), not
  a checklist the agent reasons through by hand.
- If the long tail (full flag reference, troubleshooting matrix, per-task
  schema) would blow the line budget, split it into ./reference.md and link
  it with a one-line "read this if <specific case>" pointer. Do not inline it.

SECTION SHAPE (adapt headers to the skill; keep the order)
1. One-sentence purpose statement.
2. State detection — how the agent figures out where things stand from the
   filesystem (no state file). E.g. "if outputs/<model>/<eval_id>/*.xlsx
   exists, skip to validation."
3. Decision logic — the branches this skill actually has to make, as a short
   table or tight prose. Not a tutorial.
4. Exact commands for the common path, in order, each with what success/
   failure looks like.
5. Gates — explicit "STOP and confirm with the user before:" list, matching
   this skill's gate policy (see table below). State the gate inline at the
   point it applies, not as a separate section nobody reads.
6. Failure handling — the 3-5 failures this skill actually hits, one line
   each: symptom -> fix. Anything longer goes in reference.md.
7. Pointer to reference.md (if present) and to the one or two upstream repo
   docs worth reading for edge cases this skill doesn't cover.

GATE POLICY (apply per the skill's row in the plan)
- Large downloads (SOT ~16GB, --copy duplication): confirm before starting.
- Full benchmark run (multi-hour / multi-GPU / metered API, across all
  chosen tasks): confirm before launching; smoke tests do NOT need this gate.
- Environment mutation (pip/conda installs, editing config.py, writing env
  vars into shell profiles): confirm before applying.
- Destructive flags (--force-clean, overwriting an existing submission
  archive): confirm before running, name exactly what will be lost.
- Portal submit: this repo's skills NEVER click submit. Stop after fill.

FRONTMATTER RULES
- name: vantage-<slug>, kebab-case, matches the directory name.
- description: single line, third person, states the trigger condition
  ("Use when the user wants to ..."), no marketing language, no restating
  the whole skill in the description.
-->
