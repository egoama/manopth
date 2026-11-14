"""Experiment 4 (revised): near-synonym contextual differentiation.
Fixes: exact (diacritized) lemma matching; removal of all same-root lemmas from features; exact-duplicate verse texts collapsed;
GroupKFold by surah (no surah in both train and test); baselines: majority, Meccan/Medinan-only, length-only;
metric: balanced accuracy; permutation p (labels permuted within surah-type strata); Holm across pairs.
Acceptance (pre-set): a pair is 'Robust' if Holm-adjusted p<0.05 AND balanced accuracy exceeds the Meccan/Medinan-only baseline."""
import sys,os,json,collections,math,random
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','audit'))
from common import *
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from statsmodels.stats.multitest import multipletests
rng=np.random.default_rng(4); random.seed(4)
Q=quran(); W=load_morph(MORPH); typ={s['id']:s['type'] for s in Q}
txt={(s['id'],v['id']):' '.join(clean(v['text'])) for s in Q for v in s['verses']}
lem=collections.defaultdict(list); lr={}; forms=collections.defaultdict(list)
for (s,v,w),segs in W.items():
    for d in segs:
        if d['tag'] in('N','V') and 'LEM' in d and 'PRON' not in d['feat']:
            lem[(s,v)].append(d['LEM']); lr[d['LEM']]=d.get('ROOT','')
            forms[d['LEM']].append((d['form'],tuple(x for x in d['feat'] if x in('MS','MP','FS','FP','MD','FD','M','F','P'))))
PAIRS=[('فؤاد/قلب',['فُؤاد'],['قَلْب']),('حلف/أقسم',['حَلَفْ'],['أَقْسَمُ','قَسَم','قَسَمْ']),('إنسان/بشر',['إِنسان'],['بَشَر']),
('مطر/غيث',['مَطَر'],['غَيْث']),('نظر/رأى',['نَظَرَ'],['رَأَى']),('خوف/خشية',['خافَ','خَوْف'],['خَشِيَ','خَشْيَة']),
('جاء/أتى',['جاءَ'],['أَتَى']),('عام/سنة',['عام'],['سَنَة']),('ريح(مفرد)/رياح(جمع)',None,None)]
def verses_with(L): return {k for k,ls in lem.items() if set(ls)&set(L)}
# wind singular/plural via number feature of lemma ريح
wind_s=set();wind_p=set()
for (s,v,w),segs in W.items():
    for d in segs:
        if d.get('LEM')=='رِيح':
            (wind_p if ('FP' in d['feat'] or 'MP' in d['feat']) else wind_s).add((s,v))
res=[]
for name,A,B in PAIRS:
    if A is None: VA,VB=wind_s-wind_p,wind_p-wind_s; roots_ex={'روح'}
    else:
        VA,VB=verses_with(A)-verses_with(B),verses_with(B)-verses_with(A); roots_ex={lr[x] for x in A+B if x in lr}
    # collapse exact-duplicate verse texts (keep first)
    def dedup(S):
        seen=set(); out=[]
        for k in sorted(S):
            if txt[k] not in seen: seen.add(txt[k]); out.append(k)
        return out
    a,b=dedup(VA),dedup(VB)
    data=[(k,0) for k in a]+[(k,1) for k in b]
    row={'pair':name,'n':[len(a),len(b)],'meccan_share':[round(np.mean([typ[k[0]]=='meccan' for k in a]),2) if a else None,round(np.mean([typ[k[0]]=='meccan' for k in b]),2) if b else None]}
    if min(len(a),len(b))<5 or len({k[0] for k,_ in data})<5:
        row.update({'status':'Not evaluated (too few verses/surahs)'}); res.append(row); continue
    X=[{l:1 for l in lem[k] if lr.get(l,'') not in roots_ex} for k,_ in data]
    y=np.array([c for _,c in data]); g=np.array([k[0] for k,_ in data]); mec=np.array([[1.0 if typ[k[0]]=='meccan' else 0.0] for k,_ in data]); ln=np.array([[len(lem[k])] for k,_ in data],float)
    Xv=DictVectorizer().fit_transform(X)
    nf=min(5,len(set(g)))
    def cv_bal(Xm,yy):
        preds=np.zeros(len(yy)); 
        for tr,te in GroupKFold(nf).split(Xm,yy,g):
            if len(set(yy[tr]))<2: preds[te]=yy[tr][0]; continue
            m=LogisticRegression(max_iter=2000,class_weight='balanced',C=1.0).fit(Xm[tr],yy[tr]); preds[te]=m.predict(Xm[te])
        return balanced_accuracy_score(yy,preds)
    acc=cv_bal(Xv,y); acc_mec=cv_bal(mec,y); acc_len=cv_bal(ln,y)
    # permutation: shuffle labels within Meccan/Medinan strata (controls chronology/type composition)
    null=[]
    for _ in range(300):
        yp=y.copy()
        for t in (0.0,1.0):
            ix=np.where(mec[:,0]==t)[0]; yp[ix]=rng.permutation(y[ix])
        null.append(cv_bal(Xv,yp))
    p=(sum(x>=acc for x in null)+1)/(len(null)+1)
    # bootstrap CI by surah clusters for balanced accuracy (refit inside? costly) -> report CI of permutation-null and observed only
    row.update({'balanced_acc':round(acc,3),'baseline_meccan_only':round(acc_mec,3),'baseline_length_only':round(acc_len,3),'perm_null_mean':round(float(np.mean(null)),3),'p_raw':p,'surahs':len(set(g))})
    res.append(row)
ev=[r for r in res if 'p_raw' in r]
rej,ph,_,_=multipletests([r['p_raw'] for r in ev],method='holm'); _,pb,_,_=multipletests([r['p_raw'] for r in ev],method='fdr_bh')
for r,a,b_ in zip(ev,ph,pb):
    r['p_holm']=round(float(a),4); r['p_bh']=round(float(b_),4)
    r['status']='Robust' if (a<0.05 and r['balanced_acc']>r['baseline_meccan_only']) else ('Preliminary' if r['p_raw']<0.05 else 'Not supported')
# فؤاد chronology check
fu=sorted(verses_with(['فُؤاد'])); 
mec_words=sum(1 for (s,v,w) in W if typ[s]=='meccan'); allw=len(W); pw=mec_words/allw
FU={'verses':['%d:%d'%k for k in fu],'n_verses':len(fu),'all_in_meccan_surahs':all(typ[k[0]]=='meccan' for k in fu),
    'meccan_word_share':round(pw,3),'P_all_meccan_if_random_by_words':pw**len(fu),
    'with_sam_basar_formula':sum(1 for k in fu if set(lem[k])&{'سَمْع','بَصَر'}),
    'qalb_meccan_share_in_sam_basar_contexts':None,
    'verse_level_exceptions_note':'Verse-level Medinan exceptions listed in the Egyptian mushaf headers (from memory, unverified) do not include any of these verses.',
    'prior_literature':'Avcı 2020 (Dinî Araştırmalar) reports fuʾād/afʾida occurrences in Meccan suras descriptively (citation incomplete, not verified).'}
qs=[k for k in verses_with(['قَلْب']) if set(lem[k])&{'سَمْع','بَصَر'}]
FU['qalb_meccan_share_in_sam_basar_contexts']=round(np.mean([typ[k[0]]=='meccan' for k in qs]),2) if qs else None; FU['qalb_sam_basar_n']=len(qs)
OUT={'pairs':res,'fuad':FU}
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','exp4_synonyms.json'),'w'),ensure_ascii=False,indent=1,default=float)
for r in res: print(r)
print(FU)
