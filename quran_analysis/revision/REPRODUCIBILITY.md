# Reproducibility

## Environment
- **Python:** 3.x.
- **Packages:** numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, statsmodels 0.15.0. Install with:

  ```
  pip install numpy scipy scikit-learn statsmodels
  ```
- **Hardware:** CPU only. Total runtime is about 10 minutes.

## Data
1. Download the data:

   ```
   cd <DATA_DIR> && bash quran_analysis/fetch_data.sh
   ```
2. Arrange the downloaded files so that this layout exists under `<DATA_DIR>`:

   | Path under `<DATA_DIR>` | Contents |
   |---|---|
   | `data/package/dist/quran.json` | Quran text |
   | `data/morph.txt` | Morphological analysis |
   | `ext/hadith/<Collection>/*mushakkala*.csv` | Hadith collections |
   | `ext/apcd_x/APCD_plus_porse_all.csv` | Poetry corpus |
   | `ext/qmeta/package/dist/HafsLists-*.js` | Revelation order |

3. **Verify the data:** compare the files' hashes and sizes with `results/dataset_manifest.json` (SHA-256, sizes, counts).

**Preprocessing rules** (code in `audit/common.py`):

| Data | Rule |
|---|---|
| Hadith matn | The text after the first «وَسَلَّمَ» |
| Quotes | The `{…}` spans, split at «إلى قوله» / «الآية» |
| Normalization | The original `norm()` function |
| Prosody | The original `prosody.encode()` function, unchanged |

## Commands
Run from `quran_analysis/revision/`, with `export QDATA=<DATA_DIR>` set first.

```
python3 -I audit/audit_issues.py                 # -> results/audit_issues_ABC.json
python3 -I experiments/exp1_phonology.py         # -> results/exp1_phonology.json        (seed 2026)
python3 -I experiments/exp1b_engine_validation.py# -> results/exp1b_engine_validation.json (seed 7)
python3 -I experiments/exp2_fasila_topic.py      # -> results/exp2_fasila_topic.json     (seed 42, 5000 perms)
python3 -I experiments/exp3_mutashabih.py        # -> results/exp3_mutashabih.json       (seed 3)
python3 -I experiments/exp4_synonyms.py          # -> results/exp4_synonyms.json         (seed 4)
python3 -I experiments/exp5_stylometry.py        # -> results/exp5_stylometry.json       (seed 5)
python3 -I experiments/exp6_structure_numeric.py # -> results/exp6_structure.json        (seed 6, 5000 perms)
python3 -I experiments/exp6c_pairs.py            # -> results/exp6c_pairs.json           (seed 74)
python3 -I experiments/exp6d_divine_names.py     # -> results/exp6d_divine_names.json    (seed 8)
python3 -I experiments/build_results.py          # -> RESULTS_REVISED.csv
python3 -m unittest discover -s tests -v         # 9 regression tests
```

## Expected outputs
- **Key values:** see `RESULTS_REVISED.csv`. Each row names the JSON file it comes from.
- **What else to cite:** the letter analyses come from the original `letters/` scripts (`letters.py`, `nulls.py`, `deeper_letters.py`).

## Caveats
- **Parallel run-to-run variation:** small differences can appear between runs. This happens only in the scikit-learn models that run in parallel across processor threads (n_jobs).
- **Random seeds:** every seed is fixed in its script, so repeated runs give the same results apart from the variation above.
