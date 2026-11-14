# Frozen protocols (written before running any revision_final experiment)

## A. Verse endings in near-duplicate verses
- Hypothesis H1: in cross-surah near-duplicate pairs with different endings, each verse's actual ending is compatible with the local rhyme of its own surah more often than the partner's ending would be.
- Primary outcome: pair-level sign test (pair favours own vs favours swapped), k=3, definition = last phoneme (pre-registered primary); all other definitions/k are sensitivity analyses.
- Independence unit: pair. Verses appearing in several pairs: report, and run a sensitivity analysis keeping one pair per verse (greedy, by highest similarity).
- Multiplicity: Holm over the 5 definitions at k=3.
- Frequency-matched alternative endings: for each analysed verse, draw an alternative ending from the global ending distribution of the *same rhyme-key family* (frequency-matched), 10,000 draws -> null own-match rate.
- Grammatical alternatives: NOT constructed automatically. Flagged for expert review.
- Acceptance: pair sign test p<0.01 under ≥3 of 5 definitions at k=3, Holm-adjusted; direction consistent for all k.
- Runtime: <1 min CPU. Stop after this run.

## B. Phonological engine validation
- Spec: see revision_final/PHONOLOGICAL_SPEC.md (written before benchmark).
- Benchmark: ≥300 items drawn from Quran (Uthmani), Quran quotes in hadith orthography, hadith matn, APCD poetry. Reference transcriptions produced by an INDEPENDENT rule set written from the spec (not by prosody.encode), then every item hand-checked by the analyst against the spec. Labelled UNVERIFIED (no external annotators).
- Metrics: symbol error rate (Levenshtein/len), exact-match rate, runs5 detection agreement, by phenomenon.
- Sensitivity: recompute the main ratio under alternative conventions (pausal ishba vs sakin; tanwin-fath pause; wasl at every unit boundary).
- Incremental AUC: paired per-fold differences, identical GroupKFold splits, 1000 bootstrap of folds+units -> CI for ΔAUC. Acceptance: CI excludes 0.
- Runtime: <10 min CPU.

## C. Lexical selection
- Reproduce exp4 grouped CV; add interpretable baselines (bag-of-roots logistic, Naive Bayes, single-feature rules); ablation by feature families (function words, content lemmas, verse length, type); surah-cluster bootstrap CI for balanced accuracy; Holm over 7 pairs (family) and report global adjusted values.
- Acceptance unchanged from revision: Holm p<0.05 and > Meccan/Medinan-only baseline.
- Runtime: <15 min CPU.

## Stylometry reporting (mandatory correction)
- Recompute full metric set from stored predictions under natural prevalence and under balanced subsamples (1000 draws); verify macro-F1 vs confusion matrix.
