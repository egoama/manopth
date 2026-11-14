# Publication decision — rhyme accommodation in near-duplicate Quranic verses

## Classification: **Ready with explicitly disclosed limitations** (for Claim A) / **Requires external validation** (for Claim B)

### What is established (Claim A — phonological association)
- Reproduced exactly: 193 → 159 → 30 pairs; 22 : 3 : 5; p = 1.6×10⁻⁴; Holm 4.3×10⁻⁴.
- Dependence-aware: component bootstrap CI [0.727, 1.000] for the share of decisive pairs favouring the original; component sign-flip p = 8.4×10⁻⁴; one-pair-per-verse 17 : 1.
- Null models: own-match 0.80 exceeds frequency (0.36), partner-surah (0.25), phonology-matched (0.53) and morphology-matched (0.50) nulls, all p < 10⁻⁴.
- Robust to 5 rhyme definitions × 5 neighbourhood sizes (25/25 cells) and to 45 extraction settings (frozen setting is conservative).
- Replicates identically on the Warsh text (22 : 3 : 5) with verified ayah alignment.
- Scope made precise: the effect is surah-level rhyme accommodation (N1b diagnostic), among near-verbatim cross-surah repetitions.

### What is not established
- **Claim B (lexical accommodation among viable alternatives):** N4 blinded expert review is prepared (30 items, guide, answer key separated) but **not performed**. Exploratory model notes suggest ~9/15 pairs have a grammatically viable swap; this is not validation.
- **Claim C (intentional selection):** interpretive; not testable here and not claimed.
- **Novelty:** "first counterfactual test" is qualified by incomplete access to Arabic theses (Dār al-Manẕūma) and to the full texts of the Zarqa (c. 2022) and al-Dosari studies.

### Remaining obstacles and minimum work to overcome them
| Obstacle | Minimum work | Effort |
|---|---|---|
| No independent expert validation of alternative endings | Two annotators complete `EXPERT_REVIEW_PACKAGE.csv` (Task 1 then Task 2); compute κ; run the registered analysis in `EXPERT_ANNOTATION_GUIDE.md` | ~2 annotator-days |
| Novelty qualification | Read full texts of the Zarqa paper and al-Dosari; search Dār al-Manẕūma for tallies of fāṣila-explained mutashābih cases; cite al-Kirmānī/al-Suyūṭī from specific editions and pages | ~1–2 days |
| Reference verification | Resolve every DOI/ISBN at the publisher (blocked in this environment) | hours |
| Small n | None required; report as a limitation. Optional: pre-register the threshold-0.60 extension (100 pairs) as a confirmatory analysis on a different inclusion rule | — |

### Recommendation
Submit Claim A as the core of a focused article in a Quranic-studies or Arabic computational-linguistics venue **with Claim B explicitly labelled pending**, or hold submission ~2–3 weeks to include the completed expert review, which would upgrade the paper from "association" to "accommodation among viable alternatives". Do not describe the phenomenon as newly discovered; credit Ibn al-Ṣāʾigh, al-Kirmānī, al-Iskāfī and al-Sāmarrāʾī as the hypothesis source.

### Summary table
| Component | Previous result | Final result | Robustness | Publication impact |
|---|---|---|---|---|
| Pair extraction & eligibility | 193/159/30 | identical; all exclusions documented | 45 settings, same direction | strengthens transparency |
| Primary test (D1, k=3) | 22:3:5, p=1.6e-4 | identical | Holm 4.3e-4 | core result |
| Dependence correction | not done | CI [0.73, 1.00]; sign-flip p=8.4e-4; 1-per-verse 17:1 | passes pre-set rule | removes a reviewer objection |
| Null models | N0 only (0.36) | N0 0.36, N1a 0.25, N2 0.53, N3 0.50 all passed; N1b shows surah-level scope | consistent | defines scope precisely |
| Rhyme definitions | 5 defs, k=3 | 5 defs × 5 k, 25/25 consistent | — | supports generality |
| Warsh replication | not done | 22:3:5 identical | — | adds a second reading |
| Grammatical alternatives (N4) | not done | package prepared; pending | — | needed for Claim B |
| Counterexamples | 5 listed | 8 analysed; 3 involve surah-initial verses | — | honest limits |
