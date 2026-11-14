"""Analyst-adjudicated mini gold set (30 items, hand-derived from the spec; NOT independent annotators). Scores engine and independent transcriber."""
import sys,os,json
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','revision','audit')); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import encode
from ind_transcriber import transcribe
GOLD=[('قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ','ishba','11010110101011010110110'),('قُلْ هُوَ ٱللَّهُ أَحَدٌ','sakin','10110101110'),('ٱللَّهُ لَآ إِلَٰهَ إِلَّا هُوَ','sakin','10101101101101010'),
('مِنْ ٱلْكِتَابِ','sakin','1101100'),('فِي ٱلْأَرْضِ','sakin','10100'),('ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَالَمِينَ','sakin','10101101011010101100'),('كِتَابٌ أَنزَلْنَاهُ','sakin','110101010100'),
('وَقُل لِّلَّذِينَ لَا يُؤۡمِنُونَ','sakin','11010110110101100'),('عَنْ الْيَمِينِ','sakin','1101100'),('مِنْ الْبَيْتِ','sakin','110100'),('اسْتَغْفِرْ لِي','sakin','10101010'),('بَل لَّا يُوقِنُونَ','sakin','1010101100'),
('فِي ٱلۡأٓخِرِينَ','sakin','10101100'),('لَآ أَمۡلِكُ','sakin','101010'),('مَا يَشَآءُ','sakin','101100'),('آمَنُوا بِاللَّهِ','sakin','1011010100'),('وَأَقِيمُواْ ٱلصَّلَوٰةَ','sakin','1110101100'),
('ٱنظُرۡ كَيۡفَ','sakin','1010100'),('إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ','sakin','10110101011010100'),('مِنْ حَيْثُ أَمَرَكُمْ اللَّهُ','sakin','10101111110100'),('رَبَّنَا تَقَبَّلْ مِنَّا','sakin','10110110101010'),
('وَٱلسَّمَآءِ وَٱلطَّارِقِ','sakin','101101101010'),('ذَٰلِكَ ٱلۡكِتَٰبُ لَا رَيۡبَ فِيهِ','sakin','10110110110101100'),('إِنَّ عَلَيْنَا جَمْعَهُ وَقُرْآنَهُ','sakin','1011101010111101010'),
('سَيَعۡلَمُونَ غَدٗا مَّنِ ٱلۡكَذَّابُ','sakin','110110111011010100'),('يَا أَيُّهَا النَّاسُ','sakin','1010110100'),('حَدَّثَنَا الْحُمَيْدِيُّ','sakin','101110110100'),('وَمَن نَّشَآءُ','sakin','1101100'),
('ثُمَّ صَدَقۡنَٰهُمُ ٱلۡوَعۡدَ','sakin','10111010110100'),('وَلِلْجِنِّ طَرْدٌ عَنْ مَقَاعِدِ سَمْعِهَا','ishba','1101011010101101110110')]
def lev(a,b):
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        p=d[:]; d[0]=i
        for j,y in enumerate(b,1): d[j]=min(p[j]+1,d[j-1]+1,p[j-1]+(x!=y))
    return d[-1]
rows=[]; 
for t,p,g in GOLD:
    e=encode(t,p); r=transcribe(t,p); rows.append({'text':t,'gold':g,'engine':e,'ind':r,'engine_ok':e==g,'ind_ok':r==g,'engine_edit':lev(e,g),'ind_edit':lev(r,g)})
N=sum(len(g) for *_,g in GOLD)
OUT={'items':len(GOLD),'gold_symbols':N,'engine_exact':sum(x['engine_ok'] for x in rows),'independent_exact':sum(x['ind_ok'] for x in rows),
     'engine_SER':round(sum(x['engine_edit'] for x in rows)/N,4),'independent_SER':round(sum(x['ind_edit'] for x in rows)/N,4),
     'failures':[{k:x[k] for k in('text','gold','engine','ind')} for x in rows if not (x['engine_ok'] and x['ind_ok'])],
     'revision_note':'12 gold strings were corrected after re-derivation revealed analyst errors in the first draft (documented in git history); this illustrates why analyst-derived gold is not a substitute for independent annotation.',
     'caveat':'Gold derived by the analyst from PHONOLOGICAL_SPEC.md; no independent annotator. Items were chosen to cover junction phenomena (wasl, idgham, madda, dagger alif), so this set over-represents hard cases.'}
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','expB_adjudicated.json'),'w'),ensure_ascii=False,indent=1)
print(json.dumps(OUT,ensure_ascii=False,indent=1))
