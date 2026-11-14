# Counterexamples and ties (primary analysis D1, k = 3)

Source: `results/counterexamples.json` (command: `python3 -I scripts/fasila_pipeline.py`). "Local" = set of final-phoneme keys among the available neighbours (count n).

## Pairs favouring the swapped ending (3)

| Pair | Verse i ending / local (n) | Verse j ending / local (n) | Why it counts against |
|---|---|---|---|
| 2:39 ~ 5:86 | خالدون: ن — matches local {ن×5, م} (6) | الجحيم: م — does **not** match local {ن×6} (6); خالدون would | 5:86 sits in a run of -ūn/-īn endings in al-Māʾida; the Quranic الجحيم breaks the local pattern. |
| 10:1 ~ 15:1 | الحكيم: م — does not match {ن×3} (3) | مبين: ن — matches {ن×2, م} (3); الحكيم would also match | 10:1 is surah-initial (3 neighbours only, all following); the -īm ending of 10:1 does not match the following -ūn verses. |
| 38:2 ~ 85:19 | وشقاق: ق — does not match {ر, ص, ب×2} (4); تكذيب (ب) would | تكذيب: ب — no match in {د×4, ط, ظ} (6) | Surah Ṣād has a heterogeneous opening rhyme; 85:19 sits among -d/-ṭ/-ẕ endings. |

## Ties (5)

| Pair | Reason for tie |
|---|---|
| 2:39 ~ 5:10 | both endings (ن, م) occur in both neighbourhoods |
| 7:59 ~ 23:23 | mirror case: عظيم (م) fails in 7:59's all-ن neighbourhood while تتقون (ن) would fit; تتقون fits in 23:23 while عظيم would not → D = 0 |
| 10:1 ~ 12:1 | 10:1 الحكيم fails its -ūn neighbourhood; 12:1 المبين fits; swapped would reverse → D = 0 |
| 15:45 ~ 52:17 | both -ن and -م present in both neighbourhoods (ن×5, م×1) |
| 17:83 ~ 41:51 | neither يؤوساً nor عريض matches either neighbourhood (17:83: -rā/-qā/-lā; 41:51: -ṣ/-ṭ/-ẕ/-d) |

## Observations
- Three of the eight non-supporting pairs involve **surah-initial verses** (10:1 twice, 38:2), which have only following neighbours. The analysis records the actual number of available neighbours (3–4) and does not treat missing neighbours as mismatches, but these verses are intrinsically weaker tests.
- The 7:59/23:23 tie is a genuine counter-pattern: 7:59 ends in عظيم inside a -ūn run (A'rāf 55–63 all end -ūn/-īn), i.e. a Quranic verse that **does not** accommodate its local rhyme, while its partner does.
- The 2:39/5:86 case is the clearest counterexample: al-Māʾida 84–88 end in -ūn/-īn, yet 5:86 ends in الجحيم.
- Under the one-pair-per-verse reduction (22 pairs) only one counterexample remains (38:2 ~ 85:19).

These cases show that rhyme accommodation is a strong **tendency**, not a rule, and that the phenomenon operates at the level of the surah's dominant rhyme (see N1b in `NULL_MODEL_PROTOCOL.md`) rather than as a strict local constraint.
