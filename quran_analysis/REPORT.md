# Searching the Quran for unidentified إعجاز: method, results, verdict

## 1. Data
- **Text:** the full Quran, 114 surahs and 6,236 verses (Uthmani script, Tanzil-derived `quran-json` 3.1.2).
- **Morphology:** the Quranic Arabic Corpus v0.4. Every one of the 77,429 words is segmented, with its root, lemma, gender and number.
- `fetch_data.sh` downloads both. The scripts are in `scripts/`. Every number below comes from running them.

## 2. Your example: سورة الطارق (86:1-3)
- The words are والسماء والطارق، وما أدراك ما الطارق، النجم الثاقب.
- **The idea is already in the literature.** Modern i'jaz writers, Zaghloul El-Naggar among them, link النجم الثاقب to **pulsars and neutron stars**. Their reasons:
  - الطارق means "one who knocks", which they read as a pulsar's regular radio pulses.
  - الثاقب means "piercing", which they read as intense radiation.
- Other writers link **black holes** to a different passage, الخنّس الجوار الكنّس (81:15-16).
- **The black-hole reading fits the words poorly.** Classical tafsir (al-Tabari, Ibn Kathir) gives الثاقب as "bright, piercing the darkness". A black hole emits no light, so it is the opposite of ثاقب.
- **Black holes were not wholly unthinkable before modern physics.** Michell (1783) and Laplace (1796) described "dark stars". That doesn't date them to the 7th century, but it weakens the "impossible to conceive" argument.

## 3. Checking the best-known numerical claims
Each claim was counted by strict lemma, using the corpus's own grammatical analysis.

| Claim | Strict count | Verdict |
|---|---|---|
| دنيا = آخرة (115 / 115) | 115 / 115 | **Holds** |
| ملائكة = شياطين (88 / 88) | 88 / 88 | **Holds** |
| رجل = امرأة, singular (24 / 24) | 24 / 24 | **Holds** |
| شهر, singular = 12 | 12 | **Holds** |
| يوم, singular = 365 | 445 singular, 475 total | **Fails** unless you choose which forms to count |
| حياة = موت (145 / 145) | 76 / 50 | **Fails** as lemmas; needs a selective use of derived forms |
| جنة = نار (77 / 77) | 159 / 145 | **Fails** |
| Sea : land = 71 : 29 (32 / 13) | بحر 33 or 41, برّ 15 or 22 | **Depends** on the counting rule |
| 19: letters of the Basmala | 19 | Holds in this spelling |
| 19: words of 96:1-5 | 20 | **Fails** in this spelling |
| Iron: abjad(حديد) = 26, abjad(الحديد) = 57 = surah number | 26 and 57 | Holds, but see §5 |

## 4. Test: do opposite-word pairs come out equal more often than chance?
- I tested 70 pairs of opposites or complements, such as ليل/نهار, شمس/قمر and خير/شر, using strict lemma counts.
- **Exact equalities found:** 6 in total. Two are trivial, صيف/شتاء (1/1) and مسلم/مشرك singular (2/2). The other four are the known pairs confirmed in §3.
- **Baseline:** two random lemmas of similar frequency are equal **6.5%** of the time. That predicts about 4.8 equal pairs out of 74 by pure chance, close to the 6 observed.
- A pair can also look equal when unrelated lemmas are merged by accident. In a first pass, جن = إنس seemed to give 23 / 23. The strict count is 22 / 18.
- **Conclusion:** equal word counts appear at about the rate chance predicts. The published lists collect the hits and leave out the much larger number of misses.

## 5. Test: how often does chance produce a "scientific constant"?
- **Method:** for each surah name, I computed a few cheap numbers:
  - the surah number and its verse count;
  - the abjad value of the name, with and without ال;
  - the verse numbers where the name's word appears, and those numbers plus the surah number.
- I then checked them against about 25 "meaningful" constants of 10 or more: atomic numbers, 12 months, 365 days, 23 chromosome pairs, 206 bones, 19, and so on.
- **Result:** **46 of 114 surahs (40%)** hit at least one constant.
- The famous iron coincidence (الحديد → 26 and 57) is a typical output of this process. It is not an outlier.

## 6. Scientific readings (الإعجاز العلمي)
I reviewed the verses usually cited, including:
- the expanding universe (51:47) and the heavens and earth once joined (21:30);
- the barrier between the seas (55:19-20) and the stages of the embryo (23:12-14);
- mountains as pegs (78:7), fingertips (75:4) and iron "sent down" (57:25);
- الصلب والترائب (86:7) and the darkness of the deep sea (24:40).

What I found:
1. **All of them are already well covered** (Bucaille, Keith Moore, El-Naggar, Harun Yahya, and the journals of the هيئة الإعجاز العلمي). The text kept giving me readings that are already in this body of work. I found none that is both new and strong.
2. **None meets the standard a genuine prediction would need.** That means wording that is unambiguous, specific, non-obvious and testable, read the way classical exegetes read it before the discovery. In each case the "scientific" meaning is a newer secondary sense of a word with broad meaning, and it is applied only after the science is known.
3. **The strongest classical form of إعجاز is not scientific.** It is rhetorical and linguistic: نظم (al-Jurjani's *Dala'il al-I'jaz*), بلاغة, and the unanswered challenge (التحدي, 2:23, 17:88). That claim rests on literary judgment. Counting and science-matching can't confirm it or refute it.

## 7. Verdict
- I did **not** find any إعجاز that is both new to the literature and holds up under testing.
- Counting patterns can always be generated in large numbers. Sections 4 and 5 show that their rate matches chance.
- Presenting such patterns as new miracles would be misleading.
- What this work adds is **a reproducible check** of the claims already published (§3): four hold under strict counting, and the rest depend on how you count.
