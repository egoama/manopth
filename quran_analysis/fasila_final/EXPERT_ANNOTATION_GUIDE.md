# Expert annotation guide (blinded review of alternative verse endings)

**Purpose (not to be shown to annotators during Task 1):** to test whether, among grammatically and semantically acceptable alternatives, the Quranic ending fits the local verse-ending pattern better. Annotators perform Task 1 (grammar/meaning) **before** seeing Task 2 (rhyme) and without being told the hypothesis.

## Materials
- `EXPERT_REVIEW_PACKAGE.csv` — 30 items (15 verse pairs × 2 verses). Each item shows the preceding and following verse of the surah for context, and two versions of the target verse (version_1, version_2) that differ only in the final word. The order of original vs alternative is randomized per item; the key is held separately (`expert_review/ANSWER_KEY_DO_NOT_SHARE_WITH_ANNOTATORS.csv`) and must not be given to annotators.
- `expert_review/RHYME_TASK.csv` — Task 2 sheet, to be completed only after Task 1 is submitted.

## Annotators
Two independent specialists in classical Arabic grammar/rhetoric or Quranic linguistics. Each works alone; no discussion before both have submitted. A third adjudicator resolves disagreements afterwards; individual judgments are archived before adjudication.

## Task 1 — Grammar and meaning (hypothesis not disclosed)
For each version answer on a 3-point scale (2 = yes / fully; 1 = partly / doubtful; 0 = no), with a short justification:
- **Q1** Is the version syntactically correct in the complete sentence (case, agreement, government)?
- **Q2** Is it morphologically correct (pattern, derivation, inflection)?
- **Q5** Is it natural classical Arabic in this context?

Then comparing version_2 with version_1:
- **Q3** Does version_2 preserve the essential proposition of version_1? (2/1/0)
- **Q4** Describe any semantic change in one line (free text; "none" allowed).

Do not consult the Quranic text or tafsir while answering; judge the Arabic as written.

## Task 2 — Rhyme fit (after Task 1 is locked)
`RHYME_TASK.csv` lists, for each item, the final words of the neighbouring verses (k = 3) and the two candidate final words.
- **Q6** Which version fits the local verse-ending pattern better? (1 / 2 / equal)

## Agreement statistics
- Q1, Q2, Q3, Q5 (ordinal 0–2): quadratic-weighted Cohen's κ per question.
- Q6 (nominal 1/2/equal): Cohen's κ.
- Report κ with 95% CI (bootstrap over items); pre-set adequacy threshold κ ≥ 0.6.

## Analysis plan after return (registered)
- An alternative counts as **viable** only if both annotators (after adjudication) give Q1 = 2, Q2 = 2 and Q3 ≥ 1 for the non-original version.
- Primary outcome for Claim B: among items with a viable alternative, the share where the original (per key) is judged to fit the local pattern better (Q6). Test: exact binomial vs 0.5, pair as unit (collapse the two items of a pair by majority; ties excluded); acceptance p < 0.05 and ≥ 10 viable items.
- If fewer than 10 viable alternatives exist, Claim B is reported as **not evaluable** with the current package.

## Status
**Pending.** No independent expert review has been performed. Exploratory model-generated grammatical notes (by the analyst's language model) are provided in `expert_review/EXPLORATORY_MODEL_NOTES.csv` and are **not** validation.
