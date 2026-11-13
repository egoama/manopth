#!/usr/bin/env bash
# Downloads the Quran text (quran-json, from Tanzil Uthmani text) and the
# Quranic Arabic Corpus morphology (v0.4) into ./data
set -euo pipefail
mkdir -p data && cd data
curl -sSO https://registry.npmjs.org/quran-json/-/quran-json-3.1.2.tgz && tar xzf quran-json-3.1.2.tgz
curl -sS -o morph.txt https://raw.githubusercontent.com/mustafa0x/quran-morphology/master/quran-morphology.txt
# Comparison corpora for linguistics/ (hadith: Bukhari vocalized; poetry: APCD)
git clone -q --depth 1 https://github.com/mhashim6/Open-Hadith-Data hadith
git clone -q --depth 1 https://github.com/Gheith-Abandah/classify-arabic-poetry apcd && mkdir -p apcd_x && unzip -o -q apcd/APCD_plus_porse_all.zip -d apcd_x
# Revelation order (Egyptian standard) for deeper/: quran-meta npm package
curl -sS -o qmeta.tgz https://registry.npmjs.org/quran-meta/-/quran-meta-7.0.0.tgz && mkdir -p qmeta && tar xzf qmeta.tgz -C qmeta
