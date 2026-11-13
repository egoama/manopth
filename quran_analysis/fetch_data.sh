#!/usr/bin/env bash
# Downloads the Quran text (quran-json, from Tanzil Uthmani text) and the
# Quranic Arabic Corpus morphology (v0.4) into ./data
set -euo pipefail
mkdir -p data && cd data
curl -sSO https://registry.npmjs.org/quran-json/-/quran-json-3.1.2.tgz && tar xzf quran-json-3.1.2.tgz
curl -sS -o morph.txt https://raw.githubusercontent.com/mustafa0x/quran-morphology/master/quran-morphology.txt
