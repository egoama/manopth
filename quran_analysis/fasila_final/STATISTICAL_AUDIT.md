# Statistical audit

All numbers: `results/fasila_results.json`; command `python3 -I scripts/fasila_pipeline.py` (seed 2028). Protocol: `PROTOCOL_ADDENDUM.md` (frozen before running).

## 1. Reproduction of reported values
| Reported | Reproduced | Note |
|---|---|---|
| 193 candidates / 159 cross-surah / 30 eligible | 193 / 159 / 30 | exact |
| 22 own : 3 swapped : 5 ties | 22 : 3 : 5 | exact |
| sign p ≈ 1.6×10⁻⁴; Holm ≈ 4.3×10⁻⁴ | 1.57×10⁻⁴; 4.35×10⁻⁴ | exact |
| own compatibility ≈ 0.80 | 48/60 = 0.800 (verse occurrences) ; 39/49 = 0.796 (distinct verses) | the two denominators were conflated in earlier text; both now reported |
| N0 null ≈ 0.36 | 0.356 [0.245, 0.469] | exact |
| 8 verses in multiple pairs | 8 (2:39 ×3, 52:17 ×3, 54:54 ×3, 20:116, 48:28, 10:1, 15:45, 51:15 ×2) | exact |

## 2. Dependence structure
Verse–pair graph: 49 nodes, 30 edges, **21 connected components** (sizes in pairs: one component of 5 pairs [the جنات وعيون/ونعيم/ونهر cluster: 15:45, 51:15, 52:17, 54:54], one of 3 [2:39 with 5:10, 5:86, 64:10], three of 2, sixteen singletons). The two verses of a pair are never treated as independent: the pair outcome D is the unit.

## 3. Primary metrics (D1, k = 3), unit = pair, dependence handled at component level
| Quantity | Numerator/denominator | Point | 95% CI | Test | p raw | Holm (5-definition family) | Unit | Assumptions |
|---|---|---|---|---|---|---|---|---|
| Share of decisive pairs favouring original ending | 22/25 | 0.880 | [0.727, 1.000] component bootstrap (10,000) | exact sign test | 1.57×10⁻⁴ | 4.35×10⁻⁴ | pair; CI resamples components | pairs within a component may be dependent; components independent |
| same, component-level sign-flip randomization | 22 − 3 = 19 | — | — | flip all pairs of a component together (MC 100,000) | 8.4×10⁻⁴ | — | component | exchangeability of sign within component under H0 |
| Verse-occurrence own-match rate | 48/60 | 0.800 | [0.660, 0.912] component bootstrap | — | — | — | verse occurrence | — |
| Verse-occurrence swapped-match rate | 15/60 | 0.250 | — | — | — | — | — | — |
| Own − swapped match difference | (48−15)/60 | 0.550 | [0.333, 0.750] component bootstrap | — | — | — | — | — |
| Own-match vs N0 frequency null | 39/49 vs 0.356 | Δ = 0.44 | null 95% [0.245, 0.469] | Monte Carlo 10,000 | < 10⁻⁴ | — | distinct verse | endings exchangeable across the Quran |
| Own-match vs N2 phonologically matched null | 39/49 vs 0.530 | Δ = 0.27 | null 95% [0.408, 0.633] | MC 10,000 | < 10⁻⁴ | — | distinct verse | matched on sonorant class, vowel length, word length |
| Own-match vs N3 morphology-matched null | 36/46 vs 0.501 | Δ = 0.28 | null 95% [0.391, 0.609] | MC 10,000 | < 10⁻⁴ | — | distinct verse (46 with pool ≥ 5) | matched on POS/number/gender/case/definiteness |

**One pair per verse (22 pairs):** 17 : 1 : 4; share 0.944 [0.824, 1.000]; sign p = 1.4×10⁻⁴; component sign-flip p = 1.5×10⁻⁴.

## 4. Does dependence-aware correction change the conclusion?
No. The iid sign test (p = 1.6×10⁻⁴) and the component-level randomization (p = 8.4×10⁻⁴) differ by a factor of ~5, as expected when the largest component (5 pairs, all favouring the original) is treated as one exchangeable block; the component-bootstrap CI for the share of pairs favouring the original has lower bound 0.727 (> 0.5, and above the 0.70 minimum interpretable effect set in the addendum). The pre-set acceptance (CI lower bound > 0.5 and randomization p < 0.01) is met.

## 5. Distinct quantities (not interchangeable)
- 0.80 (or 0.796): proportion of verse endings matching their own neighbourhood.
- 0.88: proportion of decisive pairs favouring the original ending over the partner's.
- 0.55: difference between own-match and swapped-match rates.
- 0.44 / 0.27 / 0.28: excess of own-match over the N0 / N2 / N3 null expectations.

## 6. Multiplicity
Primary family = 5 rhyme definitions at k = 3 (Holm). All other k, the robustness grid (45 extraction settings) and the null models are exploratory/sensitivity analyses and are not corrected; all are reported in full.

## 7. Known limitations of the inference
- n = 30 pairs (25 decisive) is small; CIs are wide (lower bound 0.73).
- Surah-initial verses (3 cases) have only following neighbours.
- The N1b diagnostic shows the effect is surah-level, not local beyond the surah pattern.
- The swapped ending is a phonological counterfactual; in 5 of the 15 top pairs it is not a grammatical substitute (see `expert_review/EXPLORATORY_MODEL_NOTES.csv`), so Claim B requires N4.
