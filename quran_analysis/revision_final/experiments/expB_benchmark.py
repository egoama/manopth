"""Priority B: benchmark prosody.encode against the independent spec-based transcriber on >=300 items (UNVERIFIED reference),
error metrics by phenomenon, runs5 agreement; recompute main ratio with the independent transcriber and under pause conventions."""
import sys,os,re,json,random,collections
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','revision','audit')); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from ind_transcriber import transcribe
import numpy as np
random.seed(300); rng=np.random.default_rng(300)
Q=quran(); recs=hadith_records()
def cov(t):
    L=sum('ء'<=c<='ي' for c in t); M=sum('ً'<=c<='ْ' or c=='ٰ' for c in t); return M/L if L else 0
items=[]
qv=[v['text'] for s in Q for v in s['verses'] if 4<=len(v['text'].split())<=14]; random.shuffle(qv)
items+=[('quran_uthmani',t,'sakin') for t in qv[:100]]
qs=[];seen=set()
for r in recs:
    for x in r['quotes']:
        for p in re.split('إِلَى قَوْلِهِ|الْآيَةَ',x):
            k=' '.join(clean(p))
            if 4<=len(p.split())<=14 and k not in seen and cov(p)>=0.85: seen.add(k); qs.append(p)
random.shuffle(qs); items+=[('quran_quote_hadith_orth',t,'sakin') for t in qs[:80]]
hm=[]
for r in recs:
    if r['matn']:
        w=r['matn'].split()
        if len(w)>=8:
            seg=' '.join(w[:10])
            if cov(seg)>=0.85: hm.append(seg)
random.shuffle(hm); items+=[('hadith_matn',t,'sakin') for t in hm[:80]]
po=[]
with open(APCD,encoding='utf8') as f:
    for line in f:
        txt,_,lab=line.rstrip('\n').rpartition(',')
        if lab not in('نثر','') and cov(txt)>=0.9: po.append(txt)
        if len(po)>=5000: break
random.shuffle(po); items+=[('poetry',t,'ishba') for t in po[:60]]
def lev(a,b):
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        p=d[:]; d[0]=i
        for j,y in enumerate(b,1): d[j]=min(p[j]+1,d[j-1]+1,p[j-1]+(x!=y))
    return d[-1]
PHEN={'shadda':'ّ','tanwin':'[ًࣰٌٍ-ࣲ]','wasl':'ٱ','madda':'آ|ٓ','dagger':'[ٰۥۦ]','lafz_jalala':'لل[َّ]*ه','tanwin_fath_final':'[ًࣰ]\S*$'}
rows=[];bench=[]
for src,t,p in items:
    e=encode(t,p); r=transcribe(t,p)
    d=lev(e,r); r5e=len(re.findall('(?=11111)',e)); r5r=len(re.findall('(?=11111)',r))
    ph=[k for k,rx in PHEN.items() if re.search(rx,t)]
    rows.append({'src':src,'exact':e==r,'edit':d,'len':len(r),'r5_engine':r5e,'r5_ref':r5r,'phen':ph,'mut_engine':e.count('1')/max(1,len(e)),'mut_ref':r.count('1')/max(1,len(r))})
    bench.append({'source':src,'text':t,'pause':p,'engine':e,'reference_independent_unverified':r,'agree':e==r})
json.dump(bench,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','benchmark','prosody_benchmark_UNVERIFIED.json'),'w'),ensure_ascii=False,indent=0)
def summ(R):
    n=len(R); 
    return {'items':n,'exact_match':round(sum(x['exact'] for x in R)/n,3),'symbol_error_rate':round(sum(x['edit'] for x in R)/sum(x['len'] for x in R),4),
            'runs5_agreement_items':round(sum(x['r5_engine']==x['r5_ref'] for x in R)/n,3),'runs5_engine_total':sum(x['r5_engine'] for x in R),'runs5_ref_total':sum(x['r5_ref'] for x in R),
            'mutaharrik_share_engine':round(float(np.mean([x['mut_engine'] for x in R])),4),'mutaharrik_share_ref':round(float(np.mean([x['mut_ref'] for x in R])),4)}
OUT={'overall':summ(rows),'by_source':{s:summ([x for x in rows if x['src']==s]) for s in sorted({x['src'] for x in rows})},
     'by_phenomenon':{k:summ([x for x in rows if k in x['phen']]) for k in PHEN if any(k in x['phen'] for x in rows)},
     'without_any_phenomenon':summ([x for x in rows if not x['phen']]) if any(not x['phen'] for x in rows) else None}
# disagreement examples for adjudication
dis=[b for b in bench if not b['agree']]; random.shuffle(dis)
OUT['disagreement_examples']=[{k:b[k] for k in('source','text','engine','reference_independent_unverified')} for b in dis[:20]]
# ---- main ratio with independent transcriber; conventions ----
seen=set(); quotes=[]
for r in recs:
    for x in r['quotes']:
        for part in re.split('إِلَى قَوْلِهِ|إِلَى آخِرِ الْآيَةِ|الْآيَةَ',x):
            k=' '.join(clean(part))
            if len(part.split())>=6 and k not in seen: seen.add(k); quotes.append(part)
seen=set(); matns=[]
for r in recs:
    if r['matn']:
        k=' '.join(clean(r['matn']))[:200]
        if k not in seen and len(r['matn'].split())>=12: seen.add(k); matns.append(r['matn'])
def ratio(fn,pause):
    def agg(T):
        a=b=0
        for t in T:
            e=fn(t,pause)
            if len(e)>=12: a+=len(re.findall('(?=11111)',e)); b+=len(e)
        return a/b*1000
    rq,rh=agg(quotes),agg(matns); return round(rq,2),round(rh,2),round(rh/rq,2)
sens={}
for name,fn in (('engine',encode),('independent',transcribe)):
    for pause in ('sakin','ishba','none'):
        q,h,r=ratio(fn,pause); sens[f'{name}|pause={pause}']={'quotes':q,'hadith':h,'ratio':r}
OUT['main_ratio_sensitivity']=sens
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','expB_benchmark.json'),'w'),ensure_ascii=False,indent=1)
print(json.dumps({k:v for k,v in OUT.items() if k!='disagreement_examples'},ensure_ascii=False,indent=1))
print('\n'.join(f"{d['source']} | {d['text'][:60]} | E={d['engine']} | R={d['reference_independent_unverified']}" for d in OUT['disagreement_examples'][:12]))
