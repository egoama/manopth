"""Experiment 1 (revised): rhythmic profile of Quran vs Hadith prose vs poetry.
Changes vs original (deep.py/analyze.py):
 - per-unit estimation only (no concatenation), dedup of repeated quotes and repeated matns
 - cluster bootstrap (unit = hadith record / distinct quote / surah) for CIs
 - extra rhythmic features; engine validation against APCD meter labels (external labels)
 - incremental value of prosodic features over lexical function-word features (grouped CV)
Acceptance rule (set before running): the Quran-vs-Hadith difference in runs5 is 'Robust' only if the
cluster-bootstrap 95% CI of the ratio excludes 1 for BOTH quote-dedup and length-matched analyses."""
import sys,os,re,json,random,math,collections,statistics as st
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','audit'))
from common import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(2026); random.seed(2026)
OUT={}
recs=hadith_records()
# ---- units ----
seen=set(); quotes=[]
for r in recs:
    for x in r['quotes']:
        for part in re.split('إِلَى قَوْلِهِ|إِلَى آخِرِ الْآيَةِ|الْآيَةَ',x):
            key=' '.join(clean(part))
            if len(part.split())>=6 and key not in seen: seen.add(key); quotes.append(part)
seen=set(); matns=[]
for r in recs:
    if r['matn']:
        key=' '.join(clean(r['matn']))[:200]
        if key not in seen and len(r['matn'].split())>=12: seen.add(key); matns.append((r['col'],r['id'],r['matn']))
Q=quran(); verses=[(s['id'],v['text']) for s in Q for v in s['verses']]
def feats(e):
    n=len(e); runs1=[len(m) for m in re.findall('1+',e)]; runs0=[len(m) for m in re.findall('0+',e)]
    return {'n':n,'runs5':len(re.findall('(?=11111)',e)),'ones':e.count('1'),'max1':max(runs1) if runs1 else 0,
            'mean_run1':st.mean(runs1) if runs1 else 0,'double0':len(re.findall('(?=00)',e))}
def unit_rows(texts,group_fn,minlen=12):
    rows=[]
    for i,t in enumerate(texts):
        e=encode(t if isinstance(t,str) else t[2])
        if len(e)>=minlen: rows.append((group_fn(i,t),e))
    return rows
Uq=unit_rows(quotes,lambda i,t:'q%d'%i)
Uh=unit_rows(matns,lambda i,t:t[0]+':'+t[1])
Uv=unit_rows([t for _,t in verses],lambda i,t:'s%d'%verses[i][0])
def rate(rows,key='runs5'):
    a=sum(feats(e)[key] for _,e in rows); b=sum(len(e) for _,e in rows); return a/b*1000
def cluster_boot(rowsA,rowsB,key='runs5',B=2000):
    ga=collections.defaultdict(list); gb=collections.defaultdict(list)
    for g,e in rowsA: f=feats(e); ga[g].append((f[key],f['n']))
    for g,e in rowsB: f=feats(e); gb[g].append((f[key],f['n']))
    A=np.array([[sum(x for x,_ in v),sum(y for _,y in v)] for v in ga.values()],float)
    Bm=np.array([[sum(x for x,_ in v),sum(y for _,y in v)] for v in gb.values()],float)
    ra=A[:,0].sum()/A[:,1].sum(); rb=Bm[:,0].sum()/Bm[:,1].sum(); out=[]
    for _ in range(B):
        ia=rng.integers(0,len(A),len(A)); ib=rng.integers(0,len(Bm),len(Bm))
        out.append((Bm[ib,0].sum()/Bm[ib,1].sum())/(A[ia,0].sum()/A[ia,1].sum()))
    out=np.sort(out); return rb/ra,(out[int(.025*B)],out[int(.975*B)]),len(A),len(Bm)
r,ci,na,nb=cluster_boot(Uq,Uh)
OUT['runs5']={'quotes_dedup_rate':round(rate(Uq),2),'hadith_dedup_rate':round(rate(Uh),2),'quran_uthmani_rate':round(rate(Uv),2),
  'ratio_hadith_over_quotes':round(r,2),'ci95_cluster':[round(ci[0],2),round(ci[1],2)],'clusters_quotes':na,'clusters_hadith':nb}
# ---- length matched: windows of exactly 40 symbols from units >=40 ----
def windows(rows,W=40):
    out=[]
    for g,e in rows:
        for i in range(0,len(e)-W+1,W): out.append((g,e[i:i+W]))
    return out
wq,wh=windows(Uq),windows(Uh)
r2,ci2,_,_=cluster_boot(wq,wh)
OUT['runs5_length_matched_40']={'ratio':round(r2,2),'ci95_cluster':[round(ci2[0],2),round(ci2[1],2)],'windows_quotes':len(wq),'windows_hadith':len(wh)}
# ---- additional rhythmic features (per-unit means with cluster CI on difference) ----
def featmean(rows,fn):
    g=collections.defaultdict(list)
    for k,e in rows: g[k].append(fn(e))
    return np.array([np.mean(v) for v in g.values()])
FN={'share_mutaharrik':lambda e:e.count('1')/len(e),'mean_run_of_harakat':lambda e:feats(e)['mean_run1'],'adjacent_sakin_per100':lambda e:feats(e)['double0']/len(e)*100}
extra={}
for k,fn in FN.items():
    a=featmean(Uq,fn); b=featmean(Uh,fn); v=featmean(Uv,fn)
    diffs=[rng.choice(b,len(b)).mean()-rng.choice(a,len(a)).mean() for _ in range(2000)]
    d=b.mean()-a.mean(); sd=math.sqrt((a.var()+b.var())/2)
    extra[k]={'quotes':round(a.mean(),4),'hadith':round(b.mean(),4),'quran_uthmani':round(v.mean(),4),'diff_hadith_minus_quotes':round(d,4),
              'ci95':[round(np.percentile(diffs,2.5),4),round(np.percentile(diffs,97.5),4)],'cohen_d':round(d/sd,3)}
OUT['other_rhythm_features']=extra
# ---- engine validation against APCD meter labels (external annotation) ----
lines=[]
with open(APCD,encoding='utf8') as f:
    for line in f:
        txt,_,lab=line.rstrip('\n').rpartition(',')
        if lab not in('نثر','') :
            L=sum('ء'<=c<='ي' for c in txt); M=sum('ً'<=c<='ْ' for c in txt)
            if L and M/L>=0.85: lines.append((lab,encode(txt,'ishba')))
        if len(lines)>=40000: break
random.shuffle(lines); cnt=collections.Counter(l for l,_ in lines); keepm=[m for m,c in cnt.items() if c>=300]
lines=[x for x in lines if x[0] in keepm]
tr,te=lines[:len(lines)//2],lines[len(lines)//2:]
K=5
def prof(seqs):
    c=collections.Counter(); 
    for s in seqs:
        for i in range(len(s)-K+1): c[s[i:i+K]]+=1
    tot=sum(c.values()); return c,tot
P={m:prof([s for l,s in tr if l==m]) for m in keepm}
def score(s,m):
    c,tot=P[m]; return sum(math.log((c[s[i:i+K]]+1)/(tot+32)) for i in range(len(s)-K+1))
corr=sum(max(keepm,key=lambda m:score(s,m))==l for l,s in te[:4000])
maj=max(collections.Counter(l for l,_ in te[:4000]).values())/4000
# corrupted engine control: flip 10% of symbols at random
def corrupt(s): return ''.join(('1' if c=='0' else '0') if random.random()<0.10 else c for c in s)
corr2=sum(max(keepm,key=lambda m:score(corrupt(s),m))==l for l,s in te[:4000])
OUT['engine_validation_meter_id']={'meters':len(keepm),'test_lines':4000,'accuracy':round(corr/4000,3),'majority_baseline':round(maj,3),
  'accuracy_with_10pct_symbol_noise':round(corr2/4000,3),
  'interpretation':'High accuracy of meter identification from engine output (an external label) indicates the 1/0 encoding preserves metrical structure; it is indirect validation, not a gold-standard symbol error rate.'}
# ---- small author-annotated gold set (NOT independent) ----
GOLD=[('قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ','ishba','11010110101011010110110'),
      ('قُلْ هُوَ ٱللَّهُ أَحَدٌ','sakin','10110101110'),('وَٱلسَّمَآءِ وَٱلطَّارِقِ','sakin','101101101010'),
      ('لَن تَنَالُواْ ٱلۡبِرَّ حَتَّىٰ تُنفِقُواْ مِمَّا تُحِبُّونَ','sakin','10110101011010101101010110100'),
      ('بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ','sakin','101010101010101100'),
      ('إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ','sakin','10110101011010100')]
def lev(a,b):
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        p=d[:]; d[0]=i
        for j,y in enumerate(b,1): d[j]=min(p[j]+1,d[j-1]+1,p[j-1]+(x!=y))
    return d[-1]
g=[(t,encode(t,p),gold) for t,p,gold in GOLD]
OUT['gold_author_annotated']={'items':len(g),'exact_match':sum(o==gd for _,o,gd in g),'symbol_edit_distance_total':sum(lev(o,gd) for _,o,gd in g),
  'symbols_total':sum(len(gd) for *_,gd in g),'details':[(t[:30],o,gd) for t,o,gd in g if o!=gd],'caveat':'annotated by the same analyst; not an independent gold standard'}
# ---- incremental value: lexical function words vs + prosodic features (grouped CV) ----
FW="و ف ثم او ام بل لا ما لم لن ان انما الا الذي التي الذين من في علي الي عن ب ل ك قد لقد هل يا اذا اذ لو لولا كان كل بعض هو هي هم انت انتم نحن انا ذلك هذا تلك عند حتي كما يوم الله رب".split()
FWi={w:i for i,w in enumerate(FW)}
def lexf(t):
    v=np.zeros(len(FW)); ws=clean(t)
    for w in ws:
        if w in FWi: v[FWi[w]]+=1
        elif len(w)>3 and w[0] in 'وف': v[FWi[w[0]]]+=1
    return v/max(1,len(ws))
def prosf(e):
    f=feats(e); return np.array([f['runs5']/f['n'],f['ones']/f['n'],f['mean_run1'],f['double0']/f['n'],f['max1']])
HON=re.compile(r'صلي الله عليه وسلم|رضي الله عنهما|رضي الله عنها|رضي الله عنه|رسول الله|\bقال\b|\bفقال\b|\bقالت\b')
X1=[];X2=[];y=[];G=[]
for i,t in enumerate(quotes):
    if len(clean(t))<12: continue
    e=encode(t); X1.append(lexf(t)); X2.append(prosf(e)); y.append(1); G.append('q%d'%(i%50))
hs=random.sample(matns,min(len(matns),3000))
for col,i_,t in hs:
    w=t.split()[:40]; tt=' '.join(w); tn=HON.sub(' ',' '.join(clean(tt)))
    e=encode(tt); 
    if len(e)<20: continue
    X1.append(lexf(tn)); X2.append(prosf(e)); y.append(0); G.append(col)
X1=np.array(X1); X2=np.array(X2); y=np.array(y); G=np.array(G)
def cv_auc(X):
    aucs=[]
    for tr,te in GroupKFold(5).split(X,y,G):
        if len(set(y[te]))<2: continue
        m=LogisticRegression(max_iter=2000,class_weight='balanced').fit((X[tr]-X[tr].mean(0))/(X[tr].std(0)+1e-9),y[tr])
        p=m.predict_proba((X[te]-X[tr].mean(0))/(X[tr].std(0)+1e-9))[:,1]; aucs.append(roc_auc_score(y[te],p))
    return float(np.mean(aucs)),aucs
a_lex,_=cv_auc(X1); a_pro,_=cv_auc(X2); a_both,_=cv_auc(np.hstack([X1,X2]))
OUT['incremental_value']={'n_quotes':int(y.sum()),'n_hadith':int((1-y).sum()),'auc_lexical':round(a_lex,3),'auc_prosodic_only':round(a_pro,3),'auc_lexical_plus_prosodic':round(a_both,3),
  'note':'GroupKFold: hadith grouped by collection; quotes grouped into 50 buckets of distinct deduplicated quotes. Prosodic features add information if lexical+prosodic AUC exceeds lexical alone.'}
OUT['decision']='Robust' if (OUT['runs5']['ci95_cluster'][0]>1 and OUT['runs5_length_matched_40']['ci95_cluster'][0]>1) else 'Not supported'
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','exp1_phonology.json'),'w'),ensure_ascii=False,indent=1,default=float)
print(json.dumps(OUT,ensure_ascii=False,indent=1,default=float))
