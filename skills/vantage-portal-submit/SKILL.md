---
name: vantage-portal-submit
description: Use when a packaged submission.tar.gz and form_metadata.md are ready to fill into the VANTAGE-Bench portal form, without submitting.
---

Fill the live VANTAGE-Bench submission form from `form_metadata.md`, using
whatever browser-automation capability is available (navigate, read page
content, fill fields, upload a file). **This skill never clicks submit** —
it fills every field, attaches the archive, then stops and hands control to
the human.

## Precondition

Only run this after `../vantage-package-submit/SKILL.md` has produced both
`submission.tar.gz` and `form_metadata.md`.

## 1. Re-read the live form first

Navigate to https://vantage-bench.org/submit and read its current field list
**before** filling anything from `form_metadata.md`. The field list has
already drifted once since this skill's sections below were written — do
not assume the live form matches this document. If a field, section, or
control type doesn't match what's described here, follow what's actually on
the page and flag the mismatch to the user.

## 2. Sections to fill (per `form_metadata.md`, cross-checked against the live page)

| # | Section | Notes |
|---|---|---|
| 01 | Identity | Leaderboard name, organization, model card/paper URL, contact email |
| 02 | Submission type | Pipeline type, system access type, **and an Evaluation track radio (Public / Preview)** |
| 03 | Model configuration | Checkpoint, parameter count, precision, zero-shot vs. fine-tuned |
| 04 | Inference setup | Infrastructure, official harness used, additional hyperparameters |
| 05 | Pillars submitted | Checkboxes — check only the pillars actually packaged (per `tasks-and-pillars.md`) |
| 06 | Predictions file | Attach `submission.tar.gz` |
| 07 | Acknowledgements | Checkboxes — see rule below |

## 3. Acknowledgements — do not check these

Section 07's checkboxes are human attestations. **Do not check any of
them.** List them out for the human to review and check themselves. Treat
any field marked `<FILL IN>` in `form_metadata.md` the same way — surface it
by name rather than guessing a value (this typically includes email,
organization name, and anything requiring subjective judgment).

## 4. Budget awareness

Submissions are capped at **2/day and 30/lifetime per email**. Because
mistakes are expensive under that budget, do a full field-by-field review
against `form_metadata.md` and the attached archive's contents before
handing back — better to catch a wrong pillar checkbox now than after using
a submission slot.

## 5. Hand back to the human

**Hard rule: never click submit.** After every field is filled and the
archive is attached, stop and summarize for the human:
- what was filled, section by section
- every field that still needs their judgment (email, org name, any
  `<FILL IN>` placeholder, all of section 07)
- any mismatch found between this skill's field list and the live page

## Failure handling

| Symptom | Fix |
|---|---|
| Live page has a section/field not listed above | fill what you can identify from `form_metadata.md`/context, flag the new field to the user, do not guess at required-but-unknown values |
| File upload doesn't accept `.tar.gz` | re-check the archive extension and the field's accepted-types hint on the live page before assuming a bug |
| Form field expects a value `form_metadata.md` marked `<FILL IN>` | stop and ask the human — do not fabricate |
