# Final validation report

**Scope:** final validation of the three priority findings, the mandatory stylometry correction, and a novelty review. Protocols were frozen in `PROTOCOL_FROZEN.md` before any new result was examined. Every number below is in `FINAL_VALIDATED_RESULTS.csv`, each with its result file and the command that produced it.

**Status labels:** Robust = passed the frozen acceptance rule. Preliminary = direction holds but a pre-set threshold or an independence requirement is unmet. Corrected = the number changed after a fix. Not evaluated = not possible with available resources.

---

## Priority A — Verse endings in near-duplicate (mutashābih) verses

| Item | Detail |
|---|---|
| Original result (revision) | 22 pairs favour own ending vs 3 (k=3, last letter), p = 1.6×10⁻⁴ |
| Remaining weaknesses | single rhyme definition; verses appearing in several pairs; no frequency-matched alternative; selection rule not audited; no multiplicity adjustment across definitions |
| Corrected method | 5 rhyme definitions (last phoneme, last two phonemes, final syllable = vowel+final consonant, ending pattern = vowel length × sonorant class, prosodic tail = last 3 prosodic symbols + consonant); k ∈ {1,2,3,5,7}; exact pair-level sign test; Holm over 5 definitions at k=3; one-pair-per-verse sensitivity; 10,000-draw frequency-matched random-ending null |

**Selection audit.** 193 candidate pairs reproduced exactly (3-word seed, 0.70 ≤ similarity < 1.0, ≥ 5 words). Exclusions: 34 same-surah pairs (uninformative: both verses share the surah rhyme), 108 cross-surah pairs with identical last word, 21 with different last word but identical last phoneme (cannot discriminate own vs swapped). None of these exclusions depends on the outcome. 30 pairs analysed; 8 verses appear in more than one pair (e.g. 2:39 ×3, 52:17 ×3, 54:54 ×3).

**Results at k = 3**

| Definition | Pairs own : swapped (ties) | Exact p | Holm p | Own-match rate [95% CI] | Frequency null |
|---|---|---|---|---|---|
| last phoneme (primary) | 22 : 3 (5) | 1.6×10⁻⁴ | 4.3×10⁻⁴ | 0.80 [0.68, 0.88] | 0.36 |
| last two phonemes | 17 : 1 (12) | 1.4×10⁻⁴ | 4.3×10⁻⁴ | 0.52 [0.39, 0.64] | 0.29 |
| final syllable | 21 : 1 (8) | 1.1×10⁻⁵ | 4.0×10⁻⁵ | 0.72 [0.59, 0.82] | 0.30 |
| ending pattern (coarse; 19 testable) | 18 : 0 (1) | 7.6×10⁻⁶ | 4.0×10⁻⁵ | 1.00 [0.91, 1.00] | 0.65 |
| prosodic tail | 21 : 3 (6) | 2.8×10⁻⁴ | 4.3×10⁻⁴ | 0.78 [0.66, 0.87] | 0.37 |

- **Direction:** own > swapped for every definition at every k (25 of 25 cells).
- **Dependence:** one pair per verse (22 pairs): 17 : 1, p = 1.4×10⁻⁴ (last phoneme).
- **Frequency-matched alternative endings:** observed own-match 0.796 vs Monte-Carlo null mean 0.357 (95% interval 0.245–0.469); p < 10⁻⁴.
- **Counterexamples (k=3, last phoneme):** 2:39/5:86 (خالدون vs الجحيم), 7:59/23:23, 10:1/12:1, 10:1/15:1, 38:2/85:19.

**Remaining limitations.** n = 30 pairs; "rhyme compatibility" is phonological, not grammatical; no expert-built, syntactically valid alternative endings were constructed (flagged for expert review); the classical hypothesis (Ibn al-Ṣāʾigh via al-Suyūṭī, al-Kirmānī, al-Samarrāʾī) is the source of the hypothesis and must be credited.

**Verdict: Robust** (frozen rule met: Holm p < 0.01 for ≥ 3 of 5 definitions; direction consistent). Effect size is large (own-match 0.80 vs 0.36 expected).

---

## Priority B — Independent validation of the phonological engine

| Item | Detail |
|---|---|
| Original result | runs ≥ 5 harakat: Hadith/Quran-quotes ratio 2.89 [2.53, 3.29]; 2.44 length-matched; engine validated only on 6 analyst items + indirect meter test (0.467) |
| Remaining weaknesses | no explicit transcription spec; no benchmark; no independent implementation; unknown sensitivity to pause/junction conventions; incremental AUC without CI |
| Corrected method | `PHONOLOGICAL_SPEC.md` (19 rules) written first; an independent spec-based transcriber (`ind_transcriber.py`, syllable-structured, separate code path); 320-item benchmark (100 Quran Uthmani, 80 Quran quotes in hadith orthography, 80 hadith segments, 60 poetry lines); 30-item analyst-adjudicated set; sensitivity of the main ratio to transcriber × pause convention; paired ΔAUC with bootstrap CI on identical GroupKFold splits |

**Benchmark (engine vs independent transcriber; reference UNVERIFIED by external annotators)**

| Subset | n | Exact match | Symbol error rate | runs5 totals (engine / ref) |
|---|---|---|---|---|
| All | 320 | 0.772 | 0.0072 | 138 / 146 (item agreement 0.975) |
| Quran, Uthmani | 100 | 0.65 | 0.0115 | 17 / 18 |
| Quran quotes, hadith orthography | 80 | 0.75 | 0.0086 | 23 / 29 |
| Hadith matn | 80 | 0.85 | 0.0045 | 93 / 94 |
| Poetry | 60 | 0.90 | 0.0026 | 5 / 5 |

By phenomenon (exact / SER): madda 0.37 / 0.0185 (worst); wasl 0.61 / 0.0112; dagger alif 0.64 / 0.0113; lafẕ al-jalāla 0.69 / 0.0098; shadda 0.78 / 0.0069; tanwin 0.85 / 0.0044; final tanwin-fatḩ 1.00 / 0.

**Adjudicated 30-item set** (analyst-derived from the spec; over-represents junction phenomena): engine 20/30 exact, SER 2.7%; independent transcriber 29/30, SER 0.27%. Three **systematic engine errors** were identified: (1) utterance-initial hamzat al-waṣl dropped (ٱلْحَمْدُ → starts with 0); (2) at a sakin + waṣl junction the engine deletes a symbol instead of vowelling the consonant (مِنْ ٱلْكِتَابِ → 101100 instead of 1101100); (3) the madda sign on أ (ٱلۡأٓخِرِينَ) not treated as a long vowel. Twelve gold strings in the first draft of the set were themselves wrong and were corrected after re-derivation — direct evidence that analyst-derived gold is not a substitute for independent annotation.

**Sensitivity of the main effect**

| Transcriber | Pause convention | Quotes rate | Hadith rate | Ratio |
|---|---|---|---|---|
| engine | sakin / ishbāʿ / none | 9.09 / 9.05 / 9.15 | 26.27 / 26.28 / 26.37 | 2.89 / 2.90 / 2.88 |
| independent | sakin / ishbāʿ / none | 9.70 / 9.65 / 9.75 | 26.69 / 26.70 / 26.78 | **2.75** / 2.77 / 2.75 |

The engine's junction errors remove voweled positions in both corpora; correcting them raises both rates and lowers the ratio from 2.89 to 2.75 (−5%). Direction and magnitude are preserved. The earlier cluster-bootstrap CI [2.53, 3.29] was computed with the engine; the independent-transcriber point estimate lies inside it.

**Incremental predictive value (identical GroupKFold splits, paired)**

| Transcriber | AUC lexical | AUC prosodic | AUC combined | ΔAUC (combined − lexical) 95% CI | Folds positive |
|---|---|---|---|---|---|
| engine | 0.811 | 0.821 | 0.880 | [0.051, 0.086] | 4/4 |
| independent | 0.811 | 0.812 | 0.876 | [0.047, 0.082] | 4/4 |

**Remaining limitations.** No independent human annotation (the reference and the gold set are by the same analyst); no diacritized early-oratory/sajʿ corpus (Tashkeela contains vowelled Shamela hadith books and could extend the comparison; not downloaded here); hadith vowelling is editorial; the comparison is relative to hadith prose, not to "all Arabic".

**Verdict:** the *difference* is **Robust** to transcriber and convention; the *engine validation* is **Preliminary** (two implementations agree at 99.3% symbol level, but neither is externally verified). Incremental AUC: **Robust**.

---

## Priority C — Lexical selection (near-synonyms)

| Item | Detail |
|---|---|
| Original result (revision) | 5 pairs "Robust" with Holm within family |
| Remaining weaknesses | global multiplicity unresolved; no interpretable baselines; no ablation; no cluster CI; counterexamples not shown |
| Corrected method | grouped CV reproduced; baselines (Naive Bayes, bag-of-roots, function-words-only, Meccan/Medinan-only, all-features); surah-cluster bootstrap CI (1,000); stratified permutation p (300); Holm within family and Holm over the union with the 44 revision tests |

| Pair | n (A/B) | Balanced acc. [95% CI] | Meccan-only | Roots | Function words | NB | p raw | Holm family | Holm global | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| إنسان/بشر | 69/37 | 0.815 [0.745, 0.898] | 0.518 | 0.801 | 0.729 | 0.660 | 0.0033 | 0.017 | 0.066 | Robust (family) / borderline (global) |
| نظر/رأى | 72/243 | 0.766 [0.709, 0.820] | 0.513 | 0.740 | 0.599 | 0.615 | 0.0033 | 0.017 | 0.066 | same |
| جاء/أتى | 238/215 | 0.706 [0.658, 0.756] | 0.489 | 0.666 | 0.649 | 0.686 | 0.0033 | 0.017 | 0.066 | same |
| خوف/خشية | 97/37 | 0.661 [0.583, 0.750] | 0.520 | 0.613 | 0.517 | 0.557 | 0.0033 | 0.017 | 0.066 | same |
| ريح/رياح | 18/10 | 0.700 [0.555, 0.875] | 0.561 | 0.800 | 0.733 | 0.672 | 0.023 | 0.023 | 0.209 | Borderline |

- **Reconciliation of "five survive":** within the 7-pair family all five remain significant after Holm. Under a global Holm correction over all 49 tests run across the revision the four priority pairs reach p ≈ 0.066 (borderline) and ريح/رياح p = 0.21. The permutation test used 300 draws, so the minimum attainable raw p is 0.0033; a finer permutation (≥ 3,000 draws) would be needed to resolve the global correction and is listed as a required action.
- **Confounding check:** Meccan/Medinan-only baselines are at chance (0.49–0.56) for the four priority pairs, so chronology does not trivially determine the target. For خوف/خشية, adding type and function words *lowers* accuracy (0.631), indicating a small, content-driven signal.
- **Drivers (top coefficients):** إنسان ← خلق، كفور، شيطان، إذا ; بشر ← مثل، قال، ملك، نذير. نظر ← كيف، عاقبة، قبل، ماذا ; رأى ← لمّا، إذا، كفر، الله. جاء ← لمّا، إذا، حق، كذب، بيّنة ; أتى ← يوم، سماء، صادق، سلطان. خوف ← يحزن، أمن، عذاب ; خشية ← مشفق، ناس، غيب.
- **Counterexamples:** misclassified verses 16/106, 47/315, 133/453, 35/134 respectively (examples in `results/expC_lexical.json`).

**Verdict: Robust within family, borderline under global correction.** Effects are moderate (balanced accuracy 0.66–0.82 vs 0.5). The substantive claim (non-interchangeability) is the consensus of the classical furūq tradition; the contribution is the leakage-controlled quantification.

---

## Mandatory correction — Stylometry reporting

Full metric set for the leave-one-collection-out logistic model (432 Quran-quote units, 38,660 hadith units; natural prevalence of Quran = 1.1%).

| Metric | Natural prevalence | Balanced subsamples (mean [95% CI], 1,000 draws) |
|---|---|---|
| Accuracy | 0.821 | 0.767 [0.748, 0.785] |
| Balanced accuracy | 0.767 | 0.767 [0.748, 0.785] |
| Precision (Quran) | **0.043** | 0.801 [0.767, 0.834] |
| Recall (Quran) | 0.711 | 0.711 |
| Precision (Hadith) | 0.996 | — |
| Specificity (recall Hadith) | 0.822 | — |
| Macro-F1 | 0.491 | 0.766 [0.747, 0.784] |
| Weighted-F1 | 0.892 | — |
| MCC | **0.144** | 0.537 [0.497, 0.576] |
| ROC-AUC | 0.836 | 0.836 [0.821, 0.850] |
| PR-AUC (Quran) | **0.241** | 0.859 [0.842, 0.875] |

Macro-F1 audit: F1(Quran) = 0.081, F1(Hadith) = 0.901, mean 0.491 — consistent with the confusion matrix [[307, 125], [6864, 31796]]. **Interpretation:** discrimination is moderate (ROC-AUC 0.84), but at natural prevalence the Quran class is practically unusable (precision 4%, MCC 0.14). Earlier texts that cited only "balanced accuracy 0.767" overstated practical performance. No further optimization was attempted (not a primary claim).

---

## Novelty review (summary; full matrix in `PUBLICATION_READINESS.md`)

Full-text access to several hosts was blocked; findings rest on indexed records and snippets. Classical precedent for A (riʿāyat al-fāṣila in mutashābih tawjīh: Ibn al-Ṣāʾigh via al-Suyūṭī; al-Kirmānī; al-Iskāfī; al-Samarrāʾī; Stewart 1990/2013; Klar 2021; Tajabadi & Kord Zafaranloo 2018 on verse-final phonology) is firmly established; no counterfactual, pair-level quantification was found. For B, adjacent work exists (Brierley/Sawalha/Atwell Boundary-Annotated Qur'an with IPA tiers; Sindran et al. 2016 syllable statistics; Sayoud stylometry) but no Quran–Hadith run-length comparison. For C, qualitative furūq literature and collocational studies exist (Almakrob & Al-Ahdal 2020); no surah-grouped classifier evaluation was found. Novelty of the quantitative designs: high; novelty of the phenomena: low (A, C) to medium (B).
