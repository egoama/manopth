# Rhyme Accommodation in Near-Duplicate Quranic Verses and the Rhythmic Profile of the Quranic Text: Controlled Computational Tests with a Hadith Prose Comparison

## Abstract
**Background.** Classical Arabic rhetoric holds that differing endings of near-duplicate (*mutashābih*) Quranic verses accommodate the rhyme (*fāṣila*) of their surah, and that Quranic composition has a distinctive rhythm. Neither claim has been tested quantitatively with null models and matched controls. **Methods.** We extracted all near-duplicate verse pairs in the Quran (193 candidates) and, for the 30 eligible cross-surah pairs with different endings, tested whether each verse's actual ending matches the rhyme of its neighbouring verses more often than its partner's ending would, under five rhyme definitions and five neighbourhood sizes, using exact pair-level sign tests and a frequency-matched random-ending null. We compared Khalilian prosodic encodings (voweled/unvoweled positions) of 1,267 deduplicated Quranic quotations in the nine canonical hadith collections with 41,071 Prophetic hadith texts in identical orthography, using two independently implemented transcribers and three pausal conventions. **Results.** In 22 of 25 decisive pairs the original ending matched the local rhyme (exact p = 1.6×10⁻⁴; Holm 4.3×10⁻⁴); own-ending match rate 0.80 [0.68, 0.88] vs 0.36 expected; direction consistent across all 25 definition × neighbourhood cells and under one-pair-per-verse analysis. Runs of ≥ 5 consecutive voweled positions were 2.75–2.90 times more frequent in hadith prose depending on transcriber; prosodic features added ΔAUC = 0.065 [0.047, 0.086] over function-word features in discriminating the two corpora. **Conclusions.** Rhyme acts as an operative constraint in the wording of near-duplicate verses, and the Quranic text shows a rhythmic profile distinct from hadith prose. The second finding is conditional on independent validation of the prosodic transcription, which was not available.

**Keywords:** Quranic rhyme (fāṣila); mutashābih verses; computational prosody; stylometry; Hadith; permutation tests; reproducibility.

## 1. Introduction
Two observations about Quranic composition have long been described qualitatively. First, *riʿāyat al-fāṣila*: Ibn al-Ṣāʾigh (d. 776/1375), as transmitted by al-Suyūṭī in *al-Itqān*, and before him al-Kirmānī (*al-Burhān*) and al-Iskāfī (*Durrat al-tanzīl*), explain many ending differences between near-duplicate verses by the rhyme of the surrounding verses. Second, rhythm: al-Rummānī and al-Bāqillānī described the Quran as "neither poetry nor prose"; Neuwirth (1981/2007) argued that rhyme delimits semantic verse groups; Stewart (1990, 2013) described *sajʿ* and clausular rhythm. Both claims lack (i) a null model and (ii) a matched control corpus from the same language and period. We ask: **RQ1** Do the endings of near-duplicate verses accommodate the local rhyme beyond chance and ending-frequency effects? **RQ2** Does the prosodic rhythm of Quranic quotations differ from hadith prose in identical orthography, robustly to transcription choices? **RQ3** Do rhythmic features carry discriminative information beyond function words? We distinguish statistical significance, relative distinctiveness, rhetorical value, and claims about origin (the last being outside measurement).

## 2. Related work
Classical and modern *mutashābih* literature (al-Kirmānī; al-Iskāfī; Ibn al-Zubayr; al-Samarrāʾī; a 2022 Zarqa University study on phonetic factors) explains ending differences case by case; computational *mutashābihāt* finders exist (Tarteel QUL). We found no counterfactual test of the rhyme explanation. Tajabadi & Kord Zafaranloo (2018) mined phonetic patterns of verse-final pauses; Brierley, Sawalha & Atwell (LREC 2012, 2014) built the Boundary-Annotated Qur'an with IPA tiers and studied *qalqalah* prosody. Sayoud (2012, *LLC*; 2015; 2018; 2025) separated Quran from Hadith with lexical and bigram features (up to 100% under some protocols); Sadeghi (2011, *Arabica*) proposed a stylometric chronology. Sindran et al. (2016) reported syllable statistics for >5M words of vowelled Arabic. No Quran–Hadith comparison of voweled-run distributions was found. Resources: Quranic Arabic Corpus (Dukes & Habash 2010; Dukes, Atwell & Habash 2013); APCD (Abandah et al. 2022); Open-Hadith-Data.

## 3. Materials and methods
### 3.1 Data
Quran: quran-json 3.1.2 (Tanzil Uthmani), 6,236 verses, 77,429 words; morphology: Quranic Arabic Corpus v0.4 (130,030 segments, 4,776 lemmas). Hadith: nine vowelled canonical collections (Open-Hadith-Data), 41,071 deduplicated *matn* texts (text after the first *wa-sallam*, quotations removed). Quranic quotations: braces in the same files, 1,267 distinct (of 2,716 occurrences). Poetry (APCD, ≥ 85% diacritized) for engine validation only. SHA-256 hashes and counts are in `revision/results/dataset_manifest.json`.

### 3.2 Experiment 1: endings of near-duplicate verses
Candidate pair = two verses sharing a 3-word sequence, SequenceMatcher ratio in [0.70, 1.0), each ≥ 5 words. Exclusions, fixed a priori: same-surah pairs (share the surah rhyme) and pairs with identical final phoneme (cannot discriminate own vs swapped). Five rhyme definitions: (1) final phoneme at pause (tanwīn-fatḩ → ā); (2) final two phonemes; (3) final syllable = (pre-final vowel/length, final consonant); (4) ending pattern = (vowel length × sonorant class); (5) prosodic tail = last three prosodic symbols + final consonant. For a pair (i, j), with neighbourhood $N_k$ of k verses on each side within the surah,
$$D=[\mathbb 1(e_i\in N_k(i))+\mathbb 1(e_j\in N_k(j))]-[\mathbb 1(e_j\in N_k(i))+\mathbb 1(e_i\in N_k(j))],$$
pairs with D > 0 favour the original ending, D < 0 the swapped one; exact sign test at pair level; Holm over the five definitions at k = 3 (definition 1 pre-registered as primary). Frequency-matched null: each verse's ending replaced by a draw from the global ending distribution (10,000 draws). Dependence sensitivity: one pair per verse.

### 3.3 Experiment 2: prosodic rhythm
Each vowelled unit is encoded as 1 (voweled consonant) / 0 (unvoweled consonant or vowel lengthening) following a 19-rule specification (`PHONOLOGICAL_SPEC.md`) covering shadda, tanwīn, hamzat al-waṣl, long vowels, assimilation, consonant-cluster resolution at junctures, pausal forms and Uthmani conventions; implemented in two independent code paths (original `prosody.encode`; syllable-structured `ind_transcriber`). Primary statistic: rate ratio of runs of ≥ 5 consecutive voweled positions per 1,000 symbols (hadith/quotations), with cluster bootstrap CI (2,000); secondary: voweled share, mean run length, adjacent-sakin rate (Cohen's d). Engine validation: 320-item cross-source benchmark (exact match, symbol error rate, run-count agreement, by phenomenon) and a 30-item analyst-derived set (not independent). Sensitivity: transcriber × pausal convention. Incremental value: GroupKFold logistic regression (hadith grouped by collection; quotations by deduplicated buckets) with lexical (46 function words/pronouns), prosodic (5), and combined features; paired ΔAUC on identical folds with within-fold unit bootstrap (1,000).

### 3.4 Experiment 3 (exploratory): near-synonym selection
For إنسان/بشر, نظر/رأى, جاء/أتى, خوف/خشية: surah-grouped CV classification of verses from content lemmas (all lemmas sharing a root with either target removed), balanced accuracy, baselines (Meccan/Medinan-only, roots, function words, Naive Bayes), surah-cluster bootstrap CI, type-stratified permutation p (300), Holm within family and over all 49 revision tests.

### 3.5 Pre-specification
Hypotheses, primary outcomes, tests, acceptance thresholds and independence units were frozen before running (`PROTOCOL_FROZEN.md`). All results are exploratory in the strict sense (hypotheses arose from the same corpus).

## 4. Results
### 4.1 Near-duplicate endings
193 candidates; 159 cross-surah; 108 with identical last word; 21 with identical final phoneme; **30 analysed**; 8 verses occur in >1 pair.

| Definition | Pairs own : swapped (ties) | Exact p | Holm p | Own match [95% CI] | Chance |
|---|---|---|---|---|---|
| final phoneme (primary) | 22 : 3 (5) | 1.6×10⁻⁴ | 4.3×10⁻⁴ | 0.80 [0.68, 0.88] | 0.36 |
| final two phonemes | 17 : 1 (12) | 1.4×10⁻⁴ | 4.3×10⁻⁴ | 0.52 [0.39, 0.64] | 0.29 |
| final syllable | 21 : 1 (8) | 1.1×10⁻⁵ | 4.0×10⁻⁵ | 0.72 [0.59, 0.82] | 0.30 |
| ending pattern (19 testable) | 18 : 0 (1) | 7.6×10⁻⁶ | 4.0×10⁻⁵ | 1.00 [0.91, 1.00] | 0.65 |
| prosodic tail | 21 : 3 (6) | 2.8×10⁻⁴ | 4.3×10⁻⁴ | 0.78 [0.66, 0.87] | 0.37 |

Direction consistent in all 25 definition × k cells. One pair per verse (22 pairs): 17 : 1, p = 1.4×10⁻⁴. Frequency-matched null: observed 0.796 vs null mean 0.357 (95% interval 0.245–0.469), p < 10⁻⁴. Examples: 15:45 *wa-ʿuyūn* (-ūn surah) vs 52:17 *wa-naʿīm* (-īm surah); 2:39 *khālidūn* vs 5:86 *al-jaḩīm*. Counterexamples: 7:59/23:23; 10:1/12:1; 10:1/15:1; 38:2/85:19 (several involve surah-initial verses lacking a preceding neighbourhood).

### 4.2 Prosodic rhythm
**Engine validation.** 320 items: exact match 0.772, symbol error rate 0.0072, run-count item agreement 0.975 (138 vs 146 runs). Uthmani Quran is hardest (0.65 / 0.0115) vs hadith (0.85 / 0.0045) and poetry (0.90 / 0.0026); worst phenomena: madda (0.37 / 0.0185), hamzat al-waṣl (0.61), dagger alif (0.64). The 30-item adjudicated set exposed three systematic original-engine errors (utterance-initial waṣl dropped; a symbol deleted instead of a consonant being vowelled at sakin+waṣl junctures; madda sign on *alif* ignored); the independent transcriber scored 29/30 (SER 0.27%). Twelve analyst-derived gold strings in the first draft were themselves wrong and were corrected—evidence that same-analyst gold is not independent validation.

**Primary statistic.**

| Transcriber | Pause | Quotations | Hadith | Ratio |
|---|---|---|---|---|
| original | sakin / ishbāʿ / none | 9.09 / 9.05 / 9.15 | 26.27 / 26.28 / 26.37 | 2.89 / 2.90 / 2.88 |
| independent | sakin / ishbāʿ / none | 9.70 / 9.65 / 9.75 | 26.69 / 26.70 / 26.78 | **2.75** / 2.77 / 2.75 |

Cluster-bootstrap CI (original engine) [2.53, 3.29]; length-matched 40-symbol windows 2.44 [2.05, 2.94]. Correcting the juncture errors raises both rates and lowers the ratio by 5%. Secondary: voweled share 0.621 vs 0.660 (d = 1.22); mean voweled-run length 1.75 vs 2.01 (d = 0.99); adjacent-sakin 1.87 vs 0.81 per 100 (d = −0.69).

**Incremental value.** Original engine: AUC lexical 0.811, prosodic 0.821, combined 0.880; ΔAUC 0.069 [0.051, 0.086], 4/4 folds positive. Independent transcriber: 0.811 → 0.876; Δ [0.047, 0.082].

### 4.3 Near-synonyms (exploratory)
| Pair | n | Balanced acc. [95% CI] | Meccan-only | Holm (family) | Holm (global) |
|---|---|---|---|---|---|
| إنسان/بشر | 69/37 | 0.815 [0.745, 0.898] | 0.518 | 0.017 | 0.066 |
| نظر/رأى | 72/243 | 0.766 [0.709, 0.820] | 0.513 | 0.017 | 0.066 |
| جاء/أتى | 238/215 | 0.706 [0.658, 0.756] | 0.489 | 0.017 | 0.066 |
| خوف/خشية | 97/37 | 0.661 [0.583, 0.750] | 0.520 | 0.017 | 0.066 |

Chronology baselines are at chance. Top drivers: إنسان ← *khuliqa, kafūr, shayṭān*; بشر ← *mithl, qāla, malak*; نظر ← *kayfa, ʿāqiba, qabl*; رأى ← *lammā, idhā, kafara*; جاء ← *lammā, ḩaqq, kadhdhaba, bayyina*; أتى ← *yawm, samāʾ, ṣādiq*. Misclassification 15–29%.

### 4.4 Stylometry reporting correction
Leave-one-collection-out logistic regression (432 quotation units vs 38,660 hadith units; Quran prevalence 1.1%): ROC-AUC 0.836; balanced accuracy 0.767; but Quran-class precision **0.043**, MCC **0.144**, PR-AUC **0.241** at natural prevalence; in balanced subsamples precision 0.801 [0.767, 0.834], MCC 0.537, PR-AUC 0.859. Macro-F1 (0.491) is consistent with the confusion matrix. Discrimination is moderate; practical classification of Quranic text at its natural prevalence is poor.

## 5. Discussion
**Rhyme accommodation.** The result quantifies, with a realistic counterfactual, what Ibn al-Ṣāʾigh and al-Kirmānī stated: where near-duplicate verses end differently, each ending follows its local rhyme. The effect is large (0.80 vs 0.36) and invariant across phonological granularities. The design is counterfactual, not causal: it shows the ending is compatible with the local rhyme, not that rhyme overrode meaning; the alternative tested is the partner's actual ending, not an expert-constructed grammatical variant.

**Rhythm.** Quranic quotations show fewer long voweled runs and more lengthening/sakin positions than hadith prose in identical orthography, with a large effect (d ≈ 1.2 for voweled share) that survives transcriber and pausal-convention changes, consistent with a recited text and with the Khalilian constraint of at most four consecutive voweled positions. Prosodic features add information beyond function words (ΔAUC ≈ 0.065, CI excluding 0). The comparison is relative to hadith—editorially vowelled and partly transmitted by meaning—not to Arabic prose in general.

**Near-synonyms.** Within-family results support the classical *furūq* view, but global correction places the four pairs at the significance boundary (0.066) with a 300-draw permutation floor; we classify them as exploratory.

**Relation to prior work.** Experiment 1 differs from the classical and modern *mutashābih* literature by exhaustive pair extraction, a realistic counterfactual and exact testing, and from Tajabadi & Kord Zafaranloo by testing a specific hypothesis rather than describing patterns. Experiment 2 adds an external hadith comparison at the prosodic level to the Leeds Boundary-Annotated Qur'an work and a prosodic dimension plus quote-leakage controls to Sayoud's stylometry.

## 6. Limitations
(1) No independent human annotation of the prosodic encoding; both references are by the same analyst. (2) No expert-constructed grammatical alternative endings. (3) The only prose control is hadith; no vowelled early oratory/sajʿ corpus was available. (4) Multiple testing: hypotheses arose from the same data; raw and adjusted p-values are reported. (5) Several publisher/DOI hosts were unreachable; references were verified through indexes and must be resolved before submission.

## 7. Conclusion
We provide, to our knowledge, the first corpus-wide counterfactual test of the classical claim that near-duplicate verse endings accommodate the local rhyme, and find it strongly supported and robust across rhyme definitions. We document a rhythmic profile of the Quranic text relative to hadith prose, conditional on independent transcription validation, and correct earlier stylometric reporting with prevalence-aware metrics. The study is **not yet submission-ready**: independent prosodic annotation and expert review of alternative endings are required.

## 8. References
Abandah, G. A., et al. (2022). *J. King Saud Univ. – CIS* 34(6B):3775–3788. doi:10.1016/j.jksuci.2020.12.002 · Brierley, C., Sawalha, M., Atwell, E. (2012). LREC, L12-1092 · Brierley, C., Sawalha, M., Atwell, E. (2014). LREC, L14-1131 · Burrows, J. (2002). *LLC* 17(3):267–287. doi:10.1093/llc/17.3.267 · Dukes, K., Habash, N. (2010). LREC, L10-1190 · Dukes, K., Atwell, E., Habash, N. (2013). *LRE* 47(1):33–62. doi:10.1007/s10579-011-9167-7 · Klar, M. (2021). In *Structural Dividers in the Qur'an*, Routledge · Massey, K. (1996). *Arabica* 43(3):497–501. doi:10.1163/1570058962582804 · Neuwirth, A. (2007). *Studien zur Komposition der mekkanischen Suren*, de Gruyter. doi:10.1515/9783110920383 · Sadeghi, B. (2011). *Arabica* 58(3–4):210–299. doi:10.1163/157005810X529692 · Sayoud, H. (2012). *LLC* 27(4):427–444. doi:10.1093/llc/fqs014 · Sindran, F., et al. (2016). *Int. J. Computational Linguistics* 7(2) · Stewart, D. J. (1990). *JAL* 21(2):101–139 · Stewart, D. J. (2013). *JQS* 15(2):22–64 · Tajabadi, F., Kord Zafaranloo Kambuziya, A. (2018). *Quranic Studies* (IHCS), article 3215 · al-Suyūṭī, *al-Itqān* (on fawāṣil, citing Ibn al-Ṣāʾigh); al-Kirmānī, *al-Burhān*; al-Iskāfī, *Durrat al-tanzīl*; al-Samarrāʾī, *al-Taʿbīr al-Qurʾānī*. Verification status of each item: `revision/COMPARISON_SOTA.md`.

## 9. Reproducibility statement
Code, frozen protocol, phonological specification, independent transcriber, 320-item benchmark, JSON results, `FINAL_VALIDATED_RESULTS.csv` (each number with its command) and regression tests are in `quran_analysis/revision_final/`. Seeds fixed; full run < 10 CPU-minutes. Data download and hash verification: `revision/REPRODUCIBILITY.md`.
