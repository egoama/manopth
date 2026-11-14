# Experiment protocols (revision)

## General rules
- **Seeds:** every random seed is fixed in each script.
- **When acceptance rules were set:** each rule is written in the script's docstring before the experiment was run.
- **Exploratory status:** all experiments are **exploratory**. The same corpus was used to generate the hypotheses in the original study, so no independent confirmation sample exists. Holm and BH corrections are applied within each family and globally.

## Status labels
| Label | Meaning |
|---|---|
| Robust | Passed the pre-set acceptance rule under the stronger controls |
| Corrected | The number changed after a bug or leakage was fixed |
| Preliminary | The direction holds, but the pre-set rule was not fully met |
| Reproduced | The original number was obtained again with no material change in controls |
| Not supported | Failed the decisive baseline |
| Not evaluated | Too little data, or not re-run |

## Per-experiment protocols

| Exp | Question / H0 | Unit (dependence) | n | Exclusions | Test | Effect size | Baseline | Acceptance rule |
|---|---|---|---|---|---|---|---|---|
| **E1** | H0: the rate of runs of ≥5 harakat per 1000 prosodic symbols is equal in Quran quotes and Hadith prose | Deduplicated quote / hadith record; the cluster is the unit | 1,267 quotes, 41,071 matns | <12 symbols; duplicates | Cluster bootstrap (2,000) of the rate ratio; 40-symbol length-matched windows | Rate ratio; Cohen's d for secondary features | Same-orthography comparison | 95% CI of the ratio > 1 in both the dedup and the length-matched analyses |
| **E1b** | Does the engine encoding preserve metrical information? | Poetry line | 5 meters × 300 test lines | coverage <0.85 | Nearest-centroid on 4-gram profiles; noise ablation | Balanced accuracy | Chance 0.2 | Balanced accuracy ≥ 0.5 |
| **E2** | H0: adjacent-verse cohesion is unrelated to a change of verse ending | Verse transition; clustered by surah | 6,122 | Cross-surah transitions | Within-surah label permutation (5,000); OLS with surah fixed effects and cluster-robust SE | Standardized difference; OLS coefficient | Structure-only model for incremental AUC | Permutation p < 0.01 on ≥2 of 3 measures, and the OLS CI excludes 0 |
| **E3** | H0: the actual ending of a near-duplicate verse is no more compatible with the local rhyme than its partner's ending | **Pair** | 30 pairs (of 193 candidates) | Same-surah pairs; pairs with identical endings | Exact sign test at pair level; k = 1, 2, 3, 5, 7 | Own-match rate (Wilson CI) | Frequency-aware random-ending null | p < 0.01 for k ∈ {2, 3, 5}, and own-match > null |
| **E4** | H0: the context does not distinguish the two near-synonyms | Verse; GroupKFold by surah | Per pair | Verse in both sets; exact duplicate texts | Balanced accuracy; permutation stratified by Meccan/Medinan (300); Holm | Balanced accuracy | Majority; Meccan/Medinan-only; length-only | Holm p < 0.05 and above the Meccan/Medinan-only baseline |
| **E5** | Separation of Quran quotes from Hadith; position of Qudsi | Unit; leave-one-collection-out | 432 Q, 38,660 H, 472 Qudsi | Duplicates; honorific and transmission formulas | LR, NB, Delta; cluster bootstrap (500) | Balanced accuracy, macro-F1, confusion matrix | Majority | Separation: CI lower bound > 0.70. Qudsi: share < 0.25 in every collection with n ≥ 20 |
| **E6a** | H0: no long-range correlation beyond within-surah composition | Verse sequence; within-surah permutation | 6,236 | — | DFA, 5,000 permutations × 4 settings | α vs null p99 | Within-surah shuffle | α > p99 in all settings |
| **E6b** | Is mushaf / chronological order smoother than random? | Surah | 114 | — | Type-preserving permutation (5,000); leave-one-feature-out | Roughness | Type-preserving shuffle | Exploratory |
| **E6c** | Are equal counts among opposite pairs more frequent than chance? | Pair | 63 testable | Zero counts | Frequency-window null (20,000) | Count of equal pairs | Same | Exploratory |
| **E6d** | Does the content predict the divine-name ending pair? | Verse; GroupKFold by surah | 177 | Duplicates; same-root lemmas | Balanced accuracy; stratified permutation | Balanced accuracy | Chance 0.2 | p < 0.01 and balanced accuracy > 0.25 |

## Stopping rule
If a hypothesis fails its decisive baseline, the negative result is recorded and no further variants are tried. This applied to:
- E6d (divine names)
- E4 (فؤاد/قلب contextual; حلف/أقسم; عام/سنة)
- E6c (numerical pairs)
- Sound–meaning (original)
