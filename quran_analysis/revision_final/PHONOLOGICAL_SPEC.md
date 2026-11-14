# Phonological transcription specification (written before the benchmark)

**Output alphabet:** `1` = a consonant carrying a short vowel (mutaharrik); `0` = a consonant with no vowel (sakin) or the lengthening half of a long vowel. This is the Khalilian prosodic code; it is not a full phonemic transcription.

## Orthography → pronunciation rules
| # | Phenomenon | Rule |
|---|---|---|
| S1 | Consonant + fatha/damma/kasra | `1` |
| S2 | Consonant + sukun (U+0652, U+06E1) | `0` |
| S3 | Shadda | geminate: `0` then the vowel/tanwin of the same letter |
| S4 | Tanwin (ـً ـٌ ـٍ, incl. U+08F0-8F2) | `1 0` (vowel + nasal); at pause: fath → `1 0` (long ā), damm/kasr → `0` (vowel and nasal dropped) |
| S5 | Long vowels: ا after fatha, و after damma, ي after kasra, with no own vowel mark | `0` |
| S6 | Dagger alif (U+0670), small waw/ya (U+06E5/6) | `0` (lengthening) unless an explicit long vowel letter already supplies it |
| S7 | Madda آ | `1 0` (hamza+ā); after a fatha-bearing letter the hamza is still a consonant → `1 0` |
| S8 | Hamzat al-wasl ٱ | silent inside connected speech; at utterance start pronounced with its vowel → `1` |
| S9 | Lam of the definite article before a shamsi letter (next letter has shadda) | assimilated: lam contributes nothing (the shadda rule S3 supplies the geminate) |
| S10 | Unmarked consonant inside a word that is not a long vowel (Uthmani ikhfa/idgham conventions) | `0`; if the next letter carries shadda (idgham) it contributes nothing |
| S11 | Silent alif: unmarked alif not preceded by fatha (e.g. after waw al-jama'a, أُولُوا) and alif with sukun | nothing |
| S12 | Word boundary, next word begins with hamzat al-wasl: previous word ends in long vowel | long vowel shortened (drop its `0`) |
| S13 | Word boundary, next word begins with hamzat al-wasl: previous word ends in a sakin consonant | iltiqa' al-sakinayn: the consonant takes a vowel → its `0` becomes `1` |
| S14 | Word boundary, next word begins with hamzat al-wasl: previous word ends in tanwin | nasal takes a vowel (tanwin `1 0` → `1 1`) |
| S15 | Pause (end of unit), convention `sakin` | final short vowel dropped: final `1` → `0`; tanwin rule S4 applies |
| S16 | Pause, convention `ishba` (poetry) | final short vowel lengthened: append `0` |
| S17 | Lafz al-jalala ٱللَّه without written alif of length | `0` lengthening inserted after the geminated lam |
| S18 | Ta marbuta at pause | `0` (as h) |
| S19 | Interior unmarked ي / و not matching preceding vowel | treated as consonant `0` (sakin) |

**Known limitations:** no treatment of qira'at-specific imala, ishmam, or the hadith variant spellings with missing diacritics; items with diacritic coverage < 0.85 are excluded from the benchmark.
