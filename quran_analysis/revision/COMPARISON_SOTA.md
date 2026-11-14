# Comparison with the state of the art

## How the references were checked
- **Method:** searches of authoritative indexes: ACL Anthology, PubMed/PMC, NASA ADS, arXiv, dblp, and publisher and library pages.
- **Limitation:** doi.org and crossref were blocked in this session, so no DOI was resolved directly. Each DOI should be resolved once before submission.
- **Verification column:**
  - *index* = the metadata agree across one or more authoritative indexed records.
  - *partial* = some fields were not confirmed.

## Literature table

| Research | Dataset | Method | Evaluation | Main finding | Difference from our study | Verification |
|---|---|---|---|---|---|---|
| Dukes & Habash 2010, LREC, aclanthology L10-1190 | Quranic Arabic Corpus | Automatic tagging + manual correction | Annotation quality | Gold morphology of the Quran | We **use** this resource (v0.4 mirror); we do not evaluate it | index |
| Dukes, Atwell & Habash 2013, *LRE* 47(1):33–62, doi:10.1007/s10579-011-9167-7 | QAC treebank | Supervised collaborative annotation | — | Dependency treebank | Resource only | index |
| Sayoud 2012, *LLC* 27(4):427–444, doi:10.1093/llc/fqs014 | Quran vs Bukhari | Stylometric features + classifiers | Classification | Quran and Hadith separate | We replicate separation with function words under leave-one-collection-out, quote dedup and balanced metrics (0.767 balanced accuracy). We add Hadith Qudsi (preliminary). Accuracies are **not comparable** across protocols. | index |
| Sayoud 2015, IVAPP, doi:10.5220/0005355601770181; Sayoud & Hadjadj 2018, Springer CCIS, doi:10.1007/978-3-319-73500-9_14; Sayoud MyJICT 10(2), doi:10.53840/myjict10-2-222 | Quran, Hadith | PCA/clustering; n-gram fusion; word-bigram transitions | Clustering / accuracy (up to 100%) | Two distinct "authors" | Same as above. We do **not** infer authorship. | index / partial |
| Sadeghi 2011, *Arabica* 58(3–4):210–299, doi:10.1163/157005810X529692 | Quran | Morpheme-frequency stylometry | Correlation of rankings | Stylistic continuity supports a relative chronology | We do not estimate a chronology. We find that the **Egyptian standard order** is not smoother than random within the Meccan/Medinan groups on our 8 features, whereas the **mushaf order** is. Different feature sets mean the results do not contradict Sadeghi. | index |
| Farrin 2014, *Structure and Qur'anic Interpretation*, White Cloud Press, ISBN 978-1-935952-98-5 | — | Qualitative ring analysis | — | Symmetry and ring composition | Our lexical mirror test found no general ring structure. Farrin's structure is thematic, so this is not a refutation. | index |
| Cuypers 2009 *The Banquet* (Convivium); 2015 *The Composition of the Qur'an* (Bloomsbury) | — | Semitic rhetorical analysis | — | Concentric composition | Same as above | index |
| Burrows 2002, *LLC* 17(3):267–287, doi:10.1093/llc/17.3.267 | English poetry | Delta | Attribution | Delta works for texts over ~1,500 words | Delta fails on our short, imbalanced units (balanced accuracy 0.51); LR and NB do better | index |
| Shokrollahi-Far 2026, *Quran Mining: Computational Stylometry of Quranic Texts*, PhD thesis, Tilburg University | Quran + imitations | Morpho-syntactic tag sequences ("stylome"), ML | Classification | Meccan/Medinan verification; authorship evidence | Not compared numerically. The thesis was **not read**. | portal exists; DOI not verified |
| Massey 1996, *Arabica* 43(3):497–501, doi:10.1163/1570058962582804 | Muqattaʿat | Ordering analysis | Qualitative / improbability | A fixed ranking of the letters; Q42 a problem | We **replicate** a single consistent order (p = 0.0012 under within-combination shuffles), treating «حم» and «عسق» as two units | index |
| Abandah et al. 2022, *JKSU-CIS* 34(6B):3775–3788, doi:10.1016/j.jksuci.2020.12.002 | APCD2 | BiLSTM meter classification and diacritization | Accuracy | High meter accuracy | We use the APCD data. Our engine with a simple classifier reaches 0.467 balanced accuracy over 5 meters, so it is far from the state of the art. Used only as validation. | index |
| Dror et al. 2004, *LLC* 19(4):431–452 | Quran | Finite-state morphology | — | Morphological database | Background | partial |
| Brierley, Sawalha & Atwell 2014, LREC, aclanthology L14-1131 | Boundary-annotated Quran | Regex detection of qalqalah | Keyness | Qalqalah as a salience marker | Closest prosodic precedent. We study runs of harakat and sakin, not qalqalah. | index |
| Sharaf & Atwell 2012, LREC, aclanthology L12-1051 (QurSim) | 7,600 related verse pairs | Tafsir-based relatedness | — | Benchmark | Could validate our cohesion measure in future work | index |
| Khadangi et al. 2022, *JIQS* 1(2):29–56; Khadangi & Shabani 2023, *JIQS* 2(2), doi:10.37264/jiqs.v2i2.December2023.4 | Quran | TF-IDF/word2vec/BERT similarity | Descriptive | Adjacent surahs are similar | We add type-preserving permutation tests and a comparison with the chronological order | partial / index |
| Neuwirth 1981/2007, de Gruyter, doi:10.1515/9783110920383 | Meccan surahs | Qualitative | — | Rhyme unites semantic verse groups | E2 gives the first **quantitative** test we found. The effect is real but very small. | index |
| Almakrob & Al-Ahdal 2020, *Asian ESP J.* 16(5.1) | Quran | Collocation analysis | — | Near-synonyms are not interchangeable | E4 adds grouped classifiers and chronology baselines | partial |

## How our claims relate to prior work

| Our claim | Prior work | Classification |
|---|---|---|
| Hadith prose has more long harakat runs than Quran quotes (same orthography) | Not found | **Potentially novel observation.** Conditional on independent engine validation. |
| Rhyme change coincides with lower lexical cohesion | Neuwirth (qualitative) | Methodological extension: first quantitative test found |
| Near-duplicate verses carry endings matching the local rhyme | Mutashābih literature (qualitative) | Methodological extension. Quantification appears novel; novelty not fully established (Arabic theses not searched exhaustively). |
| Qudsi closer to Hadith | Not found | Hypothesis requiring further evidence (preliminary) |
| فؤاد only in Meccan surahs | Avcı 2020 (descriptive; citation incomplete) | **Previously reported.** The probability estimate is new, but it is chronological, not contextual. |
| Zamakhsharī half-classes | Zamakhsharī (classical); Ibn al-Munayyir (critique) | Previously proposed theory; probability estimate appears new. Preliminary (post hoc). |
| Massey single order | Massey 1996 | Independent replication |
| Mushaf order smoother than chronological | Khadangi (adjacency similarity only) | Methodological extension; novelty not yet established |
