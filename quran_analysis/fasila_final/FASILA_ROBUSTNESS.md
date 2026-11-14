# Robustness analyses

Source: `results/fasila_results.json` → `definitions_grid`, `robustness_grid`, `null_models`. Command: `python3 -I scripts/fasila_pipeline.py`.

## 1. Rhyme definitions × neighbourhood size (own : swapped : ties; exact sign p)
| Definition | k=1 | k=2 | **k=3 (primary)** | k=5 | k=7 | Holm at k=3 |
|---|---|---|---|---|---|---|
| D1 final phoneme | 23:0:7, 2.4e-7 | 22:2:6, 3.6e-5 | **22:3:5, 1.6e-4** | 20:3:7, 4.9e-4 | 20:3:7, 4.9e-4 | 4.3e-4 |
| D2 final two phonemes | 18:1:11, 7.6e-5 | 17:1:12, 1.4e-4 | 17:1:12, 1.4e-4 | 18:3:9, 1.5e-3 | 18:2:10, 4.0e-4 | 4.3e-4 |
| D3 final syllable | 23:1:6, 3.0e-6 | 21:1:8, 1.1e-5 | 21:1:8, 1.1e-5 | 23:3:4, 8.8e-5 | 22:2:6, 3.6e-5 | 4.4e-5 |
| D4 ending pattern (19 testable) | 19:0:0, 3.8e-6 | 18:0:1, 7.6e-6 | 18:0:1, 7.6e-6 | 17:0:2, 1.5e-5 | 17:0:2, 1.5e-5 | 3.8e-5 |
| D5 prosodic tail | 22:0:8, 4.8e-7 | 21:2:7, 6.6e-5 | 21:3:6, 2.8e-4 | 20:4:6, 1.5e-3 | 20:4:6, 1.5e-3 | 4.3e-4 |

Direction (own > swapped) holds in all 25 cells. D4 is coarse (only 19 pairs differ on it) and its 100% own-match partly reflects that coarseness; it is reported but not emphasised.

## 2. Extraction robustness (D1, k = 3; eligible = cross-surah, distinguishable final phoneme)
| Shared seq | Threshold | Min words | Candidates | Eligible | Own : Swapped : Ties | p |
|---|---|---|---|---|---|---|
| 3 | 0.60 | 4 | 431 | 100 | 82 : 4 : 14 | 5.8e-20 |
| 3 | 0.60 | 5 | 373 | 80 | 63 : 4 : 13 | 1.1e-14 |
| 3 | 0.60 | 6 | 301 | 57 | 46 : 2 : 9 | 8.4e-12 |
| 3 | 0.65 | 4 | 292 | 58 | 45 : 3 : 10 | 1.3e-10 |
| 3 | 0.65 | 5 | 250 | 44 | 31 : 3 : 10 | 7.7e-7 |
| 3 | 0.65 | 6 | 205 | 32 | 23 : 1 : 8 | 3.0e-6 |
| 3 | 0.70 | 4 | 228 | 41 | 33 : 3 : 5 | 2.3e-7 |
| **3** | **0.70** | **5** | **193** | **30** | **22 : 3 : 5** | **1.6e-4** |
| 3 | 0.70 | 6 | 154 | 19 | 15 : 1 : 3 | 5.2e-4 |
| 3 | 0.75 | 4 | 174 | 29 | 26 : 0 : 3 | 3.0e-8 |
| 3 | 0.75 | 5 | 144 | 19 | 16 : 0 : 3 | 3.1e-5 |
| 3 | 0.75 | 6 | 116 | 11 | 10 : 0 : 1 | 2.0e-3 |
| 3 | 0.80 | 4 | 120 | 13 | 10 : 0 : 3 | 2.0e-3 |
| 3 | 0.80 | 5 | 112 | 13 | 10 : 0 : 3 | 2.0e-3 |
| 3 | 0.80 | 6 | 84 | 5 | 4 : 0 : 1 | 0.125 (n too small) |
| 2 | 0.70 | 5 | 192 | 30 | 22 : 3 : 5 | 1.6e-4 |
| 4 | 0.70 | 5 | 173 | 31 | 23 : 3 : 5 | 8.8e-5 |

(The full 45-setting grid is in the JSON.) The direction is uniform across all 45 settings; no setting with ≥ 11 eligible pairs has p > 0.002. The frozen setting (0.70 / 3 / 5) is **among the more conservative** choices: looser thresholds yield 44–100 eligible pairs with the same proportion (0.84–0.95 of decisive pairs favouring the original). Thresholds were fixed before outcomes were seen and were not tuned.

**Scope.** At threshold 0.60 the "near-duplicate" notion becomes loose (many pairs share only a formula); the frozen 0.70 setting restricts generalization to verses that are near-verbatim repetitions. The result generalizes to this class, not to all repeated formulae.

## 3. Null models (D1, k = 3; verse-level own-match 0.796)
| Null | Null mean [95%] | p | Interpretation |
|---|---|---|---|
| N0 global frequency | 0.356 [0.245, 0.469] | < 1e-4 | passes |
| N1a partner-surah frequency | 0.245 [0.184, 0.327] | < 1e-4 | passes; the partner's surah repertoire fits even worse |
| N1b same surah outside window (diagnostic) | 0.774 [0.673, 0.878] | 0.42 | the effect is surah-level, not local beyond the surah pattern |
| N2 phonologically matched | 0.530 [0.408, 0.633] | < 1e-4 | passes |
| N3 morphology matched (46 verses) | 0.501 [0.391, 0.609] | < 1e-4 | passes (lexical/syntactic control, not valid substitution) |
| Pair-level vs N0 random alternative | 0.937 of decisive pairs favour original [0.842, 1.0] | — | — |
| Pair-level vs N3 alternative | 0.866 [0.765, 1.0] | — | — |

## 4. Dependence
Component bootstrap CI for share of decisive pairs favouring the original: [0.727, 1.000]; component sign-flip p = 8.4×10⁻⁴; one-pair-per-verse 17 : 1 : 4 (p = 1.4×10⁻⁴). Conclusion unchanged.

## 5. Negative/limiting findings
- N1b: no evidence of local accommodation beyond the surah's rhyme repertoire.
- Three of the eight non-supporting pairs involve surah-initial verses.
- D2 (two final phonemes) has 12 ties because many endings differ only in the last phoneme; its own-match rate (0.52) is far lower than D1's.
- In 5 of the 15 highest-similarity pairs the partner's ending is **not** a grammatical substitute (`expert_review/EXPLORATORY_MODEL_NOTES.csv`), so the swapped-ending test is a phonological, not lexical, counterfactual.
