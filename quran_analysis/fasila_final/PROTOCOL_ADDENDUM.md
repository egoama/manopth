# Protocol addendum (frozen before running any fasila_final analysis)

The historical plan (`revision_final/PROTOCOL_FROZEN.md`, section A) is unchanged: primary outcome = pair-level exact sign test, definition D1 (final phoneme at pause), k = 3.

## New analyses registered here
| ID | Primary outcome | H0 | Independence unit | Method | Multiplicity family | Min. interpretable effect | Acceptance | Stopping rule |
|---|---|---|---|---|---|---|---|---|
| DEP | Share of decisive pairs favouring own ending | 0.5 | connected component of the verse–pair graph | component-level bootstrap (10,000) for CIs; component-level sign-flip randomization (exact enumeration if ≤ 2^20 components, else 100,000 MC); one-pair-per-verse | none (same primary) | share ≥ 0.70 | component-level CI lower bound > 0.5 and randomization p < 0.01 | run once |
| N0 | own-ending match rate vs global-frequency random ending | equal | verse (within pair) → component bootstrap for CI | 10,000 MC | sensitivity | Δ ≥ 0.15 | MC p < 0.01 | run once |
| N1a | own-ending match vs random ending drawn from the *partner's surah* ending distribution | equal | same | 10,000 MC | sensitivity | Δ ≥ 0.15 | MC p < 0.01 | run once |
| N1b | own-ending match vs random ending drawn from the *same surah outside the k-neighbourhood* | equal | same | 10,000 MC | **diagnostic only** (tests local-vs-surah structure, not the main hypothesis) | — | — | run once |
| N2 | own-ending match vs random ending matched on (final-consonant sonorant class, pre-final vowel length, final-word letter length), NOT on final phoneme | equal | same | 10,000 MC | sensitivity | Δ ≥ 0.15 | MC p < 0.01 | run once |
| N3 | own-ending match vs corpus-derived final words matched on morphology (POS, number, gender, definiteness, case where tagged) from verse-final positions elsewhere in the Quran | equal | same | 10,000 MC | sensitivity | Δ ≥ 0.15 | MC p < 0.01 | run once; if < 5 matched alternatives exist for a verse, record 'insufficient' |
| N4 | grammatical/semantic alternatives | — | pair | blinded expert review (pending) | — | — | not evaluable without returned reviews | prepare package only |
| ROB | selection robustness: similarity thresholds {0.60,0.65,0.70,0.75,0.80}, shared-sequence {2,3,4} words, min words {4,5,6} | — | pair | exact sign test per setting | exploratory | direction consistency | report all | run once |
| WARSH | replication on a second reading | — | pair | same as primary | — | — | only if a verified Warsh text is obtainable | otherwise 'Not evaluated' |

Decision rule for scientific claims: Claim A (phonological association) is supported if DEP and N0/N2 pass. Claim B (lexical accommodation) requires N3 to pass AND N4 to be completed by independent experts. Claim C (intentional selection) is not testable here and will be stated as interpretive.
