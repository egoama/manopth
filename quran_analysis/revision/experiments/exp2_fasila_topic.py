"""Experiment 2 (revised): do verse-ending changes coincide with lower adjacent-verse cohesion?
Measures: TF-IDF lemma cosine (original), root Jaccard, LSA (SVD-200) cosine.
Controls: log lengths, relative position, surah fixed effects (OLS); within-surah label permutation (5000).
Incremental value: predicting low-cohesion transitions (bottom quartile) with/without ending-change, grouped CV by surah.
Acceptance (pre-set): 'Robust' if permutation p<0.01 for >=2 of 3 measures AND OLS coefficient CI excludes 0 with surah FE."""
import sys,os,re,json,math,collections
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','audit'))
from common import *
import numpy as np, statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(42)
Q=quran(); W=load_morph(MORPH)
keys=[(s['id'],v['id']) for s in Q for v in s['verses']]; txt={(s['id'],v['id']):v['text'] for s in Q for v in s['verses']}
lem=collections.defaultdict(list); roots=collections.defaultdict(set)
for (s,v,w),segs in W.items():
    for d in segs:
        if d['tag'] in('N','V') and 'LEM' in d and 'PRON' not in d['feat']: lem[(s,v)].append(norm(d['LEM']).replace(' ','_'))
        if 'ROOT' in d: roots[(s,v)].add(d['ROOT'])
def fkey(t):
    w=clean(t)[-1]; return (w[-2]+'ا') if w.endswith('ا') and len(w)>1 else w[-1]
docs=[' '.join(lem[k]) or 'x' for k in keys]
T=TfidfVectorizer(token_pattern=r'\S+',min_df=1).fit_transform(docs)
Tn=T.multiply(1/np.sqrt(T.multiply(T).sum(1)+1e-12)).tocsr()
S=TruncatedSVD(200,random_state=0).fit_transform(T); Sn=S/(np.linalg.norm(S,axis=1,keepdims=True)+1e-12)
rows=[]
for i in range(len(keys)-1):
    a,b=keys[i],keys[i+1]
    if a[0]!=b[0]: continue
    n=Q[a[0]-1]['total_verses']
    rows.append({'s':a[0],'chg':int(fkey(txt[a])!=fkey(txt[b])),'tfidf':float(Tn[i].multiply(Tn[i+1]).sum()),'lsa':float(Sn[i]@Sn[i+1]),
      'root':len(roots[a]&roots[b])/max(1,len(roots[a]|roots[b])),'loglen':math.log(len(lem[a])+1)+math.log(len(lem[b])+1),'pos':a[1]/n})
OUT={'n_transitions':len(rows),'n_changes':sum(r['chg'] for r in rows)}
by=collections.defaultdict(list)
for i,r in enumerate(rows): by[r['s']].append(i)
chg=np.array([r['chg'] for r in rows])
for m in ('tfidf','root','lsa'):
    y=np.array([r[m] for r in rows]); obs=y[chg==1].mean()-y[chg==0].mean(); cnt=0; P=5000
    for _ in range(P):
        perm=chg.copy()
        for s,ix in by.items(): perm[ix]=rng.permutation(chg[ix])
        cnt+= (y[perm==1].mean()-y[perm==0].mean())<=obs
    # OLS with surah FE
    X=np.column_stack([chg,[r['loglen'] for r in rows],[r['pos'] for r in rows]])
    FE=np.zeros((len(rows),len(by))); sid={s:j for j,s in enumerate(sorted(by))}
    for i,r in enumerate(rows): FE[i,sid[r['s']]]=1
    Xf=np.column_stack([X,FE[:,1:]]); Xf=sm.add_constant(Xf)
    fit=sm.OLS(y,Xf).fit(cov_type='cluster',cov_kwds={'groups':np.array([r['s'] for r in rows])})
    b=fit.params[1]; ci=fit.conf_int()[1]
    sd=y.std()
    OUT[m]={'mean_same':round(float(y[chg==0].mean()),4),'mean_changed':round(float(y[chg==1].mean()),4),'diff':round(float(obs),4),
      'standardized_diff':round(float(obs/sd),3),'perm_p_within_surah':(cnt+1)/(P+1),'ols_coef_change_FE_clustered':round(float(b),4),'ols_ci95':[round(float(ci[0]),4),round(float(ci[1]),4)],'ols_p':float(fit.pvalues[1])}
# incremental predictive value: low cohesion (bottom quartile of LSA) predicted from structure vs structure+change
y=np.array([r['lsa'] for r in rows]); low=(y<=np.percentile(y,25)).astype(int); G=np.array([r['s'] for r in rows])
Xs=np.column_stack([[r['loglen'] for r in rows],[r['pos'] for r in rows],[r['tfidf'] for r in rows]])
def cv(X):
    au=[]
    for tr,te in GroupKFold(5).split(X,low,G):
        mu,sdv=X[tr].mean(0),X[tr].std(0)+1e-9
        p=LogisticRegression(max_iter=1000).fit((X[tr]-mu)/sdv,low[tr]).predict_proba((X[te]-mu)/sdv)[:,1]; au.append(roc_auc_score(low[te],p))
    return float(np.mean(au))
a0=cv(Xs[:,:2]); a1=cv(np.column_stack([Xs[:,:2],chg]))
OUT['incremental_auc_low_LSA_cohesion']={'structure_only':round(a0,3),'structure_plus_ending_change':round(a1,3),'gain':round(a1-a0,3)}
ok=sum(OUT[m]['perm_p_within_surah']<0.01 for m in ('tfidf','root','lsa'))>=2 and all(OUT[m]['ols_ci95'][1]<0 for m in ('tfidf','lsa'))
OUT['decision']='Robust' if ok else 'Preliminary'
OUT['not_done']='No validated classical-Arabic sentence-embedding model reachable offline (HuggingFace blocked); no independently annotated topic boundaries available.'
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','exp2_fasila_topic.json'),'w'),ensure_ascii=False,indent=1)
print(json.dumps(OUT,ensure_ascii=False,indent=1))
