"""Priority B: incremental AUC of prosodic over lexical features with identical GroupKFold splits, paired per-fold differences,
and bootstrap CI (resampling units within folds). Also with the independent transcriber."""
import sys,os,re,json,random,collections
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','revision','audit')); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from ind_transcriber import transcribe
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
random.seed(2026); rng=np.random.default_rng(2026)
recs=hadith_records()
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
        if k not in seen and len(r['matn'].split())>=12: seen.add(k); matns.append((r['col'],r['matn']))
FW="و ف ثم او ام بل لا ما لم لن ان انما الا الذي التي الذين من في علي الي عن ب ل ك قد لقد هل يا اذا اذ لو لولا كان كل بعض هو هي هم انت انتم نحن انا ذلك هذا تلك عند حتي كما يوم الله رب".split(); FWi={w:i for i,w in enumerate(FW)}
HON=re.compile(r'صلي الله عليه وسلم|رضي الله عنهما|رضي الله عنها|رضي الله عنه|رسول الله|\bقال\b|\bفقال\b|\bقالت\b')
def lexf(ws):
    v=np.zeros(len(FW))
    for w in ws:
        if w in FWi: v[FWi[w]]+=1
        elif len(w)>3 and w[0] in 'وف': v[FWi[w[0]]]+=1
    return v/max(1,len(ws))
def prosf(e):
    n=len(e); r1=[len(m) for m in re.findall('1+',e)]
    return np.array([len(re.findall('(?=11111)',e))/n,e.count('1')/n,np.mean(r1) if r1 else 0,len(re.findall('(?=00)',e))/n,max(r1) if r1 else 0])
hs=random.sample(matns,3000)
def build(fn):
    X1=[];X2=[];y=[];G=[]
    for i,t in enumerate(quotes):
        ws=clean(t)
        if len(ws)<12: continue
        e=fn(t,'sakin'); X1.append(lexf(ws)); X2.append(prosf(e)); y.append(1); G.append('q%d'%(i%50))
    for col,t in hs:
        tt=' '.join(t.split()[:40]); ws=HON.sub(' ',' '.join(clean(tt))).split(); e=fn(tt,'sakin')
        if len(e)<20: continue
        X1.append(lexf(ws)); X2.append(prosf(e)); y.append(0); G.append(col)
    return np.array(X1),np.array(X2),np.array(y),np.array(G)
def paired(X1,X2,y,G):
    folds=list(GroupKFold(5).split(X1,y,G)); per=[]; store=[]
    def fit(X,tr,te):
        mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9
        m=LogisticRegression(max_iter=3000,class_weight='balanced').fit((X[tr]-mu)/sd,y[tr]); return m.predict_proba((X[te]-mu)/sd)[:,1]
    for tr,te in folds:
        if len(set(y[te]))<2: continue
        pl=fit(X1,tr,te); pp=fit(X2,tr,te); pb=fit(np.hstack([X1,X2]),tr,te)
        per.append((roc_auc_score(y[te],pl),roc_auc_score(y[te],pp),roc_auc_score(y[te],pb))); store.append((te,pl,pp,pb))
    per=np.array(per)
    # bootstrap: resample units within each fold, recompute AUCs, pool as mean over folds
    boots=[]
    for _ in range(1000):
        acc=[]
        for te,pl,pp,pb in store:
            ix=rng.integers(0,len(te),len(te)); yy=y[te][ix]
            if len(set(yy))<2: continue
            acc.append((roc_auc_score(yy,pl[ix]),roc_auc_score(yy,pp[ix]),roc_auc_score(yy,pb[ix])))
        a=np.mean(acc,0); boots.append(a[2]-a[0])
    boots=np.array(boots)
    return {'auc_lexical':round(float(per[:,0].mean()),3),'auc_prosodic':round(float(per[:,1].mean()),3),'auc_combined':round(float(per[:,2].mean()),3),
            'per_fold_delta_combined_minus_lexical':[round(float(x),3) for x in per[:,2]-per[:,0]],'delta_ci95_bootstrap':[round(float(np.percentile(boots,2.5)),3),round(float(np.percentile(boots,97.5)),3)],
            'folds_with_positive_delta':int((per[:,2]>per[:,0]).sum()),'n_folds':len(per)}
OUT={}
for name,fn in (('engine',encode),('independent_transcriber',transcribe)):
    X1,X2,y,G=build(fn); OUT[name]=paired(X1,X2,y,G); OUT[name]['n']=[int(y.sum()),int((1-y).sum())]
OUT['decision']='Robust' if all(v['delta_ci95_bootstrap'][0]>0 for v in OUT.values() if isinstance(v,dict)) else 'Preliminary'
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','expB_auc.json'),'w'),ensure_ascii=False,indent=1)
print(json.dumps(OUT,ensure_ascii=False,indent=1))
