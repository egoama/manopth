# Audit findings

Scope: every analysis module in `quran_analysis/` (the original code is left unchanged). Revised code is in `quran_analysis/revision/`.

**The audit guide is missing.** The audit-guide file named in the task, `Quran_Report_Scientific_Rewrite_AR.md`, is **not present in the workspace**. A full-filesystem search found nothing. Each concern listed in the task brief was therefore checked directly against the code.

**Severity levels**
- **Critical:** changes a headline conclusion.
- **Major:** changes a number or claim materially.
- **Minor:** reporting or consistency issue.

| ID | Severity | Location | Problem | Evidence | Fix | Status |
|---|---|---|---|---|---|---|
| AUDIT-01 (Issue A) | Minor | `deep/deep.py` L27-45 | The table rates (7.29 vs 22.77) and the ratio 3.18 come from **two different estimators on different samples**. The rates use concatenated strings truncated to 200k symbols (`prose[:40000]`). The ratio uses a per-unit estimate on `prose[:20000]`. It is not an arithmetic error. No run crosses a unit boundary (verified as 0). Rerunning the original pipeline gives 23.47 for prose, so the number is **sample-sensitive**. | `results/audit_issues_ABC.json` | Single per-unit estimator on deduplicated full data, with a cluster bootstrap CI | Corrected: 2.89 [2.53, 3.29] |
| AUDIT-02 (Issue B) | Minor | `semantic/semantic.py` L51-56 | 192 verses carry one of the five endings. 26:220 and 36:5 were dropped because they had **no content lemma left** after removing the last 3 words and the divine names. They are not duplicates. | `audit_issues_ABC.json` | Documented; revised test in `exp6d` | Resolved |
| AUDIT-03 (Issue C) | **Critical** | `extra/extra_fix.py` L46-50 | **(a)** Accuracy of 0.815 is pooled accuracy on an imbalanced test set (360 Quran vs 750 hadith, so the majority baseline is 0.676), but it was compared with 50%. **(b)** Quran chunks were built by **concatenating quotes and splitting randomly**. 1,370 of the 2,716 quote instances are repeats, so identical texts **leaked** between train and test. **(c)** Hadith repeated across collections also leaked. | `audit_issues_ABC.json` | Deduplication; leave-one-collection-out; group keys for quotes; balanced accuracy, macro-F1, confusion matrix, cluster CI | Corrected: balanced acc. 0.767 [0.734, 0.796] |
| AUDIT-04 (Issue D) | Major | `extra/extra_fix.py` DFA | Only **20 permutations** were used, at one scale set and one detrending order | — | 5,000 within-surah permutations × 4 settings (scales, detrending order 1/2) | Rhyme: robust. Verse length: preliminary (one setting below p99) |
| AUDIT-05 (Issue E) | Major | all reports | **No multiplicity correction.** Exploratory and confirmatory results were not separated. | — | Holm and BH applied within each family and globally, in `RESULTS_REVISED.csv`. Every result is labelled exploratory unless a pre-set rule is stated. | Done |
| AUDIT-06 | Major | `scripts/pairs.py` | `norm()` merges إِنس and آنَسَ, giving a **spurious جن/انس equality**. The list has **70 pairs, not 74**; the report's "74" added 4 pairs computed separately. | `tests/test_regressions.py` | Documented. The null model is re-run under one consistent rule (`exp6c`). | Corrected: 2/63 equal vs 4.15 expected |
| AUDIT-07 | Major | `scidiscourse/discourse.py` (draft) | `norm('كَأَنَّ') == norm('كانَ')`, so the verb «كان» was counted as a simile marker | test | Already replaced in the original study by a manual sample; flagged here | Documented |
| AUDIT-08 | **Critical** | `semantic/semantic.py`, `robust.py` | The divine-name ending test used pooled leave-one-out: **verbatim refrains** (e.g. «وإن ربك لهو العزيز الرحيم» ×8) sat in train and test, and the evaluation was **not grouped by surah** | `exp6d` | Exact duplicates collapsed; GroupKFold by surah; permutation stratified by surah type | The finding **does not survive** (p = 0.063) |
| AUDIT-09 | **Critical** | `deep/synonyms.py` | Synonym classifiers used leave-one-out **without grouping by surah**, and had no chronology baseline | `exp4` | GroupKFold by surah; Meccan/Medinan-only baseline; type-stratified permutation; Holm | فؤاد/قلب and حلف/أقسم **lose their contextual effect** (it is chronological). Five pairs survive. |
| AUDIT-10 | Major | `extra/extra_fix.py` (Qudsi) | The 6.5% Qudsi→Quran figure rests on one threshold and one leaky model | `exp5` | Per-collection estimates; threshold sensitivity; comparison with the ordinary-hadith false-positive rate | Preliminary: 24% pooled, 8–33% across collections, vs 18% for hadith |
| AUDIT-11 | Minor | `FINAL_REPORT_AR.md` | "4,728 lemmas" excludes particles. The full count is 4,776 (`dataset_manifest.json`). | manifest | Definition stated | Documented |
| AUDIT-12 | Minor | `linguistics/prosody.py` | No independent gold standard exists. Validation used 6 analyst-annotated items (exact match) plus an external proxy: meter identification on APCD. Balanced accuracy was 0.467 vs chance 0.2, **below the pre-set 0.5 threshold**. | `exp1b` | Reported as partial validation | Open |
| AUDIT-13 | Major | `deep/deep.py` B (sound–meaning) | Already found non-robust in the original study after stratification | — | Kept as a negative result | Not supported |
| AUDIT-14 | Minor | `FINAL_REPORT_AR.md` §8.4 (earlier versions) | The claim "no verse contradicts science" was based on one-sided sampling. It was already corrected in the original report (chapter 8 مكرر 4). | — | Kept as corrected | Corrected |

**Resources that were unavailable** (documented, not worked around):
- HuggingFace (Arabic sentence-embedding models), so **no neural semantic measure** was used.
- Independent topic-boundary annotations.
- Corpora of early Arabic oratory and sajʿ with diacritics.
- Several hosts were blocked during the literature checks: doi.org, crossref, quran.ksu.edu.sa, eajaz.org.

**Not re-evaluated** (kept with their original labels):
- Moses narrative tailoring
- Emotional arc
- Character voices
- The 46 scientific hypotheses and the 20-verse audit (these are qualitative literature assessments)
