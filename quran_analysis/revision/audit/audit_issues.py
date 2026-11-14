"""Reproduce and diagnose Issues A (ratio), B (190 vs 192), C (stylometry baseline/leakage)."""
import sys,os,re,json,random,collections,statistics as st
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
random.seed(11); OUT={}
recs=hadith_records()
# ---------- Issue A ----------
quotes=[];prose=[]
for r in recs:
    for x in r['quotes']:
        for part in re.split('إِلَى قَوْلِهِ|إِلَى آخِرِ الْآيَةِ|الْآيَةَ',x):
            if len(part.split())>=4: quotes.append(part)
    if r['matn']:
        w=r['matn'].split()
        for i in range(0,len(w)-7,10): prose.append(' '.join(w[i:i+10]))
random.shuffle(prose)   # same seed/sequence as deep.py up to this point is NOT guaranteed; we diagnose structure, not exact replay
def pooled(texts,N=200000):
    s=''.join(encode(t) for t in texts)[:N]; return len(re.findall('(?=11111)',s))/len(s)*1000
def per_unit(texts):
    a=b=0
    for t in texts:
        e=encode(t)
        if len(e)>=10: a+=len(re.findall('(?=11111)',e)); b+=len(e)
    return a/b*1000
A={'pooled_concatenated_quotes':round(pooled(quotes),2),'pooled_concatenated_prose':round(pooled(prose[:40000]),2),
   'per_unit_quotes':round(per_unit(quotes),2),'per_unit_prose_20000':round(per_unit(prose[:20000]),2)}
A['ratio_of_pooled_rates']=round(A['pooled_concatenated_prose']/A['pooled_concatenated_quotes'],2)
A['ratio_of_per_unit_rates']=round(A['per_unit_prose_20000']/A['per_unit_quotes'],2)
# boundary-crossing runs created by concatenation
def cross(texts,N=200000):
    encs=[encode(t) for t in texts]; s=''.join(encs)[:N]; tot=len(re.findall('(?=11111)',s)); inside=0; L=0
    for e in encs:
        if L+len(e)>N: break
        inside+=len(re.findall('(?=11111)',e)); L+=len(e)
    return tot-inside
A['runs_created_across_unit_boundaries_quotes']=cross(quotes); A['runs_created_across_unit_boundaries_prose']=cross(prose[:40000])
A['diagnosis']='The table rates (7.29, 22.77) come from pooled concatenated strings (truncated to 200k symbols, prose[:40000]); the 3.18 ratio comes from a different per-unit aggregation on prose[:20000]. Two different estimators on different samples, not an arithmetic error.'
OUT['issue_A']=A
# ---------- Issue B ----------
W=load_morph(MORPH); words=collections.defaultdict(list)
for (s,v,w),segs in W.items(): words[(s,v)].append((w,segs))
def content(segs): return [d for d in segs if d['tag'] in('N','V') and 'LEM' in d and 'PRON' not in d['feat']]
NAMES={norm(x) for x in ['غَفُور','رَحِيم','عَزِيز','حَكِيم','عَلِيم','سَمِيع','بَصِير','خَبِير','لَطِيف','قَدِير','تَوّاب','غَنِيّ','حَمِيد','حَلِيم','واسِع','شَكُور','قَوِيّ','رَؤُوف','وَدُود','كَرِيم','عَظِيم','عَلِيّ','كَبِير','مَجِيد','قَرِيب','مُجِيب','شَهِيد','حَفِيظ','رَقِيب','وَكِيل','حَسِيب','مُحِيط']}
endings={}
for k in sorted(words):
    ws=sorted(words[k]); tail=[]
    for _,segs in reversed(ws):
        c=content(segs)
        if c and norm(c[0]['LEM']) in NAMES: tail.insert(0,norm(c[0]['LEM']))
        elif c: break
        if len(tail)==2: break
    if len(tail)==2: endings[k]=' '.join(tail)
cnt=collections.Counter(endings.values()); keep=[e for e,c in cnt.items() if c>=12]
dropped=[]
for k,e in endings.items():
    if e in keep:
        ws=sorted(words[k])[:-3]
        f={norm(d['LEM']) for _,segs in ws for d in content(segs) if norm(d['LEM']) not in NAMES and norm(d['LEM'])!='الله'}
        if not f: dropped.append('%d:%d'%k)
OUT['issue_B']={'verses_with_kept_endings':sum(cnt[e] for e in keep),'counts':{e:cnt[e] for e in keep},'dropped_empty_feature_verses':dropped,
  'diagnosis':'192 verses carry one of the five endings; 2 were dropped because no content lemma remained after removing the last 3 words and divine names (very short verses). Not duplicates.'}
# ---------- Issue C ----------
OUT['issue_C']={'test_set_quran_chunks':360,'test_set_hadith_chunks':750,'majority_baseline':round(750/1110,3),
 'note':'Original accuracy 0.815 is pooled accuracy on an imbalanced test set (majority 0.676), compared in the report to 50%. Quran chunks were built by concatenating quotes then randomly split -> identical quoted verses (quoted many times across collections) appear in both train and test (leakage). Revised in experiments/exp5_stylometry.py.'}
# quantify quote duplication
nq=collections.Counter(' '.join(clean(q)) for r in recs for q in r['quotes'] if len(q.split())>=6)
OUT['issue_C']['distinct_quotes']=len(nq); OUT['issue_C']['quote_instances']=sum(nq.values()); OUT['issue_C']['instances_that_are_repeats']=sum(c-1 for c in nq.values())
json.dump(OUT,open(os.path.join(os.path.dirname(HERE),'results','audit_issues_ABC.json'),'w'),ensure_ascii=False,indent=1)
print(json.dumps(OUT,ensure_ascii=False,indent=1))
