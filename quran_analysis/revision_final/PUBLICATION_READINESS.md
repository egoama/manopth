# Publication readiness

## Contribution classification

| Contribution | Classification | Reason |
|---|---|---|
| **A. Mutashābih endings conform to local rhyme** (22:3 pairs; own-match 0.80 vs 0.36 null; robust across 5 definitions and k=1–7; dependence-robust) | **Requires minor validation** | Statistically robust on existing data; needs (i) expert-constructed grammatical alternative endings for a handful of pairs, (ii) replication on a second riwāya text (Warsh), (iii) a check of Arabic theses (Dār al-Manẕūma) for prior tallies. Publishable as the core result of a methods paper once (i)–(iii) are done or explicitly scoped out. |
| **B1. Quran quotes vs Hadith prose: rhythmic difference** (ratio 2.75–2.90; share mutaḩarrik d = 1.2) | **Requires independent validation** | Effect is robust to transcriber and conventions, but both transcribers are by the same analyst; the 30-item gold set had 12 analyst errors in its first draft. Needs ≥ 2 independent annotators on ≥ 300 segments with inter-annotator agreement, and a diacritized non-hadith prose corpus (e.g. Tashkeela Shamela books) to test specificity. |
| **B2. Prosodic features add predictive information beyond function words** (ΔAUC 0.065, CI [0.047, 0.086]) | **Ready for submission as a supporting result** (conditional on B1's engine caveat being stated) | Paired, identical splits, CI excludes 0 under both transcribers. |
| **C. Near-synonym contextual differentiation** (4 pairs, balanced acc. 0.66–0.82) | **Exploratory only (borderline)** | Robust within family; global Holm p ≈ 0.066 with a 300-draw permutation floor. Needs ≥ 3,000 permutations and a pre-registered confirmatory pair list. Substantive claim is classical consensus; contribution is methodological. |
| ريح/رياح | **Exploratory only** | n = 28; global Holm 0.21. |
| Stylometry Quran/Hadith separation | **Ready as a corrected replication note**, not as a contribution | Reproduces Sayoud et al.; adds leakage control and honest prevalence-aware metrics (precision 4% at natural prevalence). |
| Hadith Qudsī position | **Exploratory only** | Threshold-sensitive (9–53%); per-collection range 8–33% vs 18% for ordinary hadith. |
| Fāṣila change vs lexical cohesion | **Exploratory only** | Real but very small effect (standardized −0.04 to −0.07); no predictive gain. |
| Divine-name endings; equal-count pairs; pangram/Fātiḩa letter counts; sound–meaning | **Not supported** | Failed decisive baselines after leakage removal / null models. |
| Zamakhsharī half-classes; Massey order; initials→Book | **Exploratory only** | Post hoc patterns; Massey replicated; classical attributions not re-verified from editions. |

## Highest-priority remaining actions (ordered)
1. **Independent prosodic annotation** (two annotators, ≥ 300 segments stratified by phenomenon; report κ). Blocks B1 from "independent validation" to "ready". Est. 2–3 annotator-days.
2. **Expert review of alternative endings for Priority A** (10–15 pairs; a grammarian supplies syntactically valid variants preserving meaning). Converts A's "phonological compatibility" to a stronger lexical-choice claim.
3. **Second riwāya / second text edition** for A and B1 (Warsh text from Tanzil; standard-orthography Quran) to test orthographic robustness.
4. **Non-hadith prose control** for B1 (Tashkeela: vowelled Shamela non-hadith books; early khuṭab if a vowelled edition exists). Tests whether the effect is "Quran vs prose" or "Quran vs hadith".
5. **Finer permutation test for C** (≥ 3,000 draws) and a pre-registered confirmatory list of 4 pairs; report global Holm.
6. **Resolve all DOIs at doi.org** (blocked in this session) and read the Zarqa (2022) and al-Dosari studies in full before asserting novelty for A.

## Answers to the final scientific questions
1. **Robust:** A (all definitions, all k, dependence-robust); B1 as a *relative* difference (robust to transcriber/convention); B2 (ΔAUC CI excludes 0); C within-family.
2. **Potentially novel:** the quantitative, counterfactual design of A (hypothesis itself is classical); the Quran–Hadith run-length comparison of B1; the leakage-controlled lexical-choice evaluation of C. None is "first to notice the phenomenon".
3. **Scientifically meaningful effect size:** A (0.80 vs 0.36); B1 (ratio ≈ 2.8; d ≈ 1.2 for voweled share); B2 (ΔAUC 0.065). C moderate (0.66–0.82). Fāṣila–cohesion: negligible.
4. **Publishable without additional data:** A as a methods-and-results core (with the grammatical-alternatives limitation stated), B2, the stylometry correction note.
5. **Requires independent expert annotation:** B1 (prosodic gold), A's grammatical alternatives, any topic-boundary claim.
6. **Abandon:** divine-name endings, equal-count coincidences, pangram/Fātiḩa counts, sound–meaning, lexical ring composition, speculative scientific readings.
7. **One article or several:** the evidence supports **one methods-focused article** centred on A, with B1/B2 as a second results section explicitly labelled "pending independent annotation", and C as an exploratory section. A separate short note is appropriate for the stylometry correction (replication of Sayoud with leakage control). The study is **not submission-ready today**: B1 lacks independent validation and A lacks the expert alternative-endings control; both are feasible within weeks.

## Novelty matrix (condensed from the literature agent; full-text access partly blocked)

| Claim | Prior work | Dataset | Method | Controls | Existing finding | Our addition | Novelty confidence | Uncertainty |
|---|---|---|---|---|---|---|---|---|
| A | al-Kirmānī *al-Burhān*; al-Iskāfī *Durrat al-tanzīl*; Ibn al-Ṣāʾigh via al-Suyūṭī *Itqān* (al-Dosari collection, Mandumah 1507750); Zarqa J. Hum. c.2022 "المتشابهات القرآنية في ضوء العناصر الصوتية"; Stewart JAL 1990, JQS 2013; Klar 2021; Tajabadi & Kord Zafaranloo 2018 | hand-selected pairs; all verse endings (Tajabadi) | qualitative tawjīh; descriptive prosody; clustering of final phonemes | none | fāṣila is one accepted explanation of ending differences | corpus-wide pair extraction, swap counterfactual, exact tests, 5 definitions | High (design) / low (phenomenon) | Zarqa & al-Dosari full texts unread; Persian papers unread |
| B1 | Brierley/Sawalha/Atwell LREC 2012 (L12-1092), 2014 IPA tiers; Sindran et al. 2016 IJCL 7(2); Sayoud 2012–2025; Sinai 2017 verse-length | Quran only / mixed Arabic / Bukhari | phonemic tiers; frequency laws; stylometry | CV (Sayoud) | Quran prosody is characteristic; Quran–Hadith lexically separable | first Quran–Hadith comparison on a mutaḩarrik/sakin code; run-length statistic; identical vowelling | High (metric) / medium (general claim) | no Arabic thesis on hadith syllable runs found; engine not externally validated |
| C | Bint al-Shāṭiʾ; al-Samarrāʾī; Almakrob & Al-Ahdal 2020; Alshammeri 2022 (Leeds); QurSim | hand-chosen sets; whole Quran | exegesis; collocation; embeddings | none with held-out grouped CV | non-interchangeability (consensus) | surah-grouped CV, Meccan/Medinan baselines, ablation | High (design) / low (phenomenon) | Almakrob full text unread |

Independent resources identified for future B1 validation: Boundary-Annotated Qur'an v2.0 IPA tiers (Leeds), Quranic Phonemizer (PyPI), Nawar Halabi Arabic-Phonetiser, IqraEval, Tashkeela (vowelled Shamela prose), CAMeL Tools.
