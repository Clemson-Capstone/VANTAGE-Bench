# VANTAGE-Bench skills

Plain markdown + YAML frontmatter, no vendor lock-in — every skill here is
readable and executable by any coding agent that can run shell commands and
read files, not just one particular tool. Each `SKILL.md` states its
trigger condition, the exact commands it wraps, and where it hands off
next; shared facts (dataset keys, pillar map, id formats) live once in
[`reference/tasks-and-pillars.md`](reference/tasks-and-pillars.md).

## Index

| Skill | Purpose | Typical trigger |
|---|---|---|
| [`vantage-bench`](vantage-bench/SKILL.md) | Top-level orchestrator across the whole flow | "Run VANTAGE-Bench for me end to end" |
| [`vantage-preflight`](vantage-preflight/SKILL.md) | Environment/dependency check before anything else | "Am I set up to run VANTAGE-Bench?" |
| [`vantage-data-prep`](vantage-data-prep/SKILL.md) | Fetch/link the public dataset into `$LMUData` | "Download the VANTAGE-Bench data" |
| [`vantage-model-config`](vantage-model-config/SKILL.md) | Write/choose a model config (CLI shortcut vs. config JSON) | "Set up my model to run against VANTAGE" |
| [`vantage-add-model`](vantage-add-model/SKILL.md) | Register a new model class not already in `supported_VLM` | "Add support for my custom model" |
| [`vantage-run`](vantage-run/SKILL.md) | Smoke test → gated full inference/eval run, with monitoring and resume | "Run VQA and Temporal for my model" |
| [`vantage-cluster-launch`](vantage-cluster-launch/SKILL.md) | Same runs on SLURM or a cloud/rented GPU VM | "Launch this on the cluster" |
| [`vantage-validate`](vantage-validate/SKILL.md) | Blocker/warning check on submission JSONLs before packaging | "Is my submission ready to package?" |
| [`vantage-package-submit`](vantage-package-submit/SKILL.md) | Bundle into `submission.tar.gz`, draft the portal form | "Package my submission" |
| [`vantage-portal-submit`](vantage-portal-submit/SKILL.md) | Fill the live portal form; never clicks submit | "Fill out the submission form" |
| [`vantage-troubleshoot`](vantage-troubleshoot/SKILL.md) | Symptom → fix lookup table across the whole flow | "Why did my run/validation/submission fail?" |

`vantage-bench`, `vantage-preflight`, `vantage-data-prep`,
`vantage-model-config`, and `vantage-add-model` cover setup through a
working inference command; `vantage-run` onward covers everything from a
working setup through a submission-ready archive and filled-in portal form.
