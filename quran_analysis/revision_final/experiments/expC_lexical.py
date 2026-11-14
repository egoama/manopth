"""Priority C: near-synonym pairs — grouped CV reproduction, interpretable baselines, ablation by feature family,
surah-cluster bootstrap CI, counterexamples, Holm (family) and global adjustment. Only the 4 priority pairs + ريح/رياح (borderline)."""
import sys,os,json,collections,random
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','revision','audit'))
from common import *
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from statsmodels.stats.multitest import multipletests
rng=np.random.default_rng(44); random.seed(44)
Q=quran(); W=load_morph(MORPH); typ={s['id']:s['type'] for s in Q}
txt={(s['id'],v['id']):' '.join(clean(v['text'])) for s in Q for v in s['verses']}; raw={(s['id'],v['id']):v['text'] for s in Q for v in s['verses']}
lem=collections.defaultdict(list); lr={}; isfw=collections.defaultdict(set); forms=collections.defaultdict(collections.Counter)
for (s,v,w),segs in W.items():
    for d in segs:
        if 'LEM' in d:
            if d['tag']=='P': isfw[(s,v)].add(norm(d['LEM']))
            elif d['tag'] in('N','V') and 'PRON' not in d['feat']: lem[(s,v)].append(d['LEM']); lr[d['LEM']]=d.get('ROOT',''); forms[d['LEM']][d['form']]+=1
PAIRS=[('إنسان/بشر',['إِنسان'],['بَشَر']),('نظر/رأى',['نَظَرَ'],['رَأَى']),('جاء/أتى',['جاءَ'],['أَتَى']),('خوف/خشية',['خافَ','خَوْف'],['خَشِيَ','خَشْيَة']),('ريح/رياح',None,None)]
def verses_with(L): return {k for k,ls in lem.items() if set(ls)&set(L)}
wind_s=set();wind_p=set()
for (s,v,w),segs in W.items():
    for d in segs:
        if d.get('LEM')=='رِيح': (wind_p if ('FP' in d['feat'] or 'MP' in d['feat']) else wind_s).add((s,v))
res=[]
for name,A,B in PAIRS:
    if A is None: VA,VB=wind_s-wind_p,wind_p-wind_s; rex={'روح'}; fcount={'ريح':len(wind_s),'رياح':len(wind_p)}
    else: VA,VB=verses_with(A)-verses_with(B),verses_with(B)-verses_with(A); rex={lr[x] for x in A+B if x in lr}; fcount={x:dict(forms[x]) for x in A+B}
    def dedup(S):
        seen=set(); out=[]
        for k in sorted(S):
            if txt[k] not in seen: seen.add(txt[k]); out.append(k)
        return out
    a,b=dedup(VA),dedup(VB); data=[(k,0) for k in a]+[(k,1) for k in b]
    y=np.array([c for _,c in data]); g=np.array([k[0] for k,_ in data]); mec=np.array([[1.0 if typ[k[0]]=='meccan' else 0.0] for k,_ in data])
    FEAT={'content':[{norm(l):1 for l in lem[k] if lr.get(l,'') not in rex} for k,_ in data],
          'roots':[{lr[l]:1 for l in lem[k] if lr.get(l,'') not in rex and lr.get(l)} for k,_ in data],
          'function_words':[{w:1 for w in isfw[k]} for k,_ in data]}
    Xs={f:DictVectorizer().fit_transform(v) for f,v in FEAT.items()}
    def cvpred(X,model='lr'):
        pr=np.zeros(len(y))
        for tr,te in GroupKFold(5).split(X,y,g):
            if len(set(y[tr]))<2: pr[te]=y[tr][0]; continue
            m=(LogisticRegression(max_iter=3000,class_weight='balanced') if model=='lr' else BernoulliNB()).fit(X[tr],y[tr]); pr[te]=m.predict(X[te])
        return pr
    out={'pair':name,'n':[len(a),len(b)],'surahs':len(set(g)),'forms':fcount}
    P={}
    P['lr_content']=cvpred(Xs['content']); P['nb_content']=cvpred(Xs['content'],'nb'); P['lr_roots']=cvpred(Xs['roots']); P['lr_function_words']=cvpred(Xs['function_words'])
    P['meccan_only']=cvpred(mec); 
    from scipy.sparse import hstack
    P['lr_all']=cvpred(hstack([Xs['content'],Xs['function_words'],mec]).tocsr())
    out['balanced_acc']={k:round(balanced_accuracy_score(y,v),3) for k,v in P.items()}
    # cluster bootstrap CI (resample surahs) for lr_content
    us=np.unique(g); gi={s:np.where(g==s)[0] for s in us}; bs=[]
    for _ in range(1000):
        ix=np.concatenate([gi[s] for s in rng.choice(us,len(us))])
        if len(set(y[ix]))<2: continue
        bs.append(balanced_accuracy_score(y[ix],P['lr_content'][ix]))
    out['lr_content_ci95_surah_bootstrap']=[round(float(np.percentile(bs,2.5)),3),round(float(np.percentile(bs,97.5)),3)]
    # permutation p stratified by type
    null=[]
    for _ in range(300):
        yp=y.copy()
        for t in (0.0,1.0):
            ix=np.where(mec[:,0]==t)[0]; yp[ix]=rng.permutation(y[ix])
        pr=np.zeros(len(yp))
        for tr,te in GroupKFold(5).split(Xs['content'],yp,g):
            if len(set(yp[tr]))<2: pr[te]=yp[tr][0]; continue
            pr[te]=LogisticRegression(max_iter=3000,class_weight='balanced').fit(Xs['content'][tr],yp[tr]).predict(Xs['content'][te])
        null.append(balanced_accuracy_score(yp,pr))
    out['p_raw']=(sum(x>=out['balanced_acc']['lr_content'] for x in null)+1)/301
    # drivers: top coefficients (fit on all data, content)
    m=LogisticRegression(max_iter=3000,class_weight='balanced').fit(Xs['content'],y); dv=DictVectorizer().fit(FEAT['content']); names=np.array(dv.get_feature_names_out())
    co=m.coef_[0]; out['top_features_B']=list(names[np.argsort(-co)[:8]]); out['top_features_A']=list(names[np.argsort(co)[:8]])
    mis=[('%d:%d'%k,'A' if c==0 else 'B',raw[k][:70]) for (k,c),p in zip(data,P['lr_content']) if p!=c]
    out['misclassified_n']=len(mis); out['misclassified_examples']=mis[:6]
    res.append(out)
ps=[r['p_raw'] for r in res]; h=multipletests(ps,method='holm')[1]
# global: union with RESULTS_REVISED raw p-values
import csv
allp=[]
for row in csv.DictReader(open(os.path.join(ROOT,'revision','RESULTS_REVISED.csv'),encoding='utf8')):
    try: allp.append(float(row['p_raw']))
    except: pass
gl=multipletests(allp+ps,method='holm')[1][len(allp):]
for r,a,gg in zip(res,h,gl):
    r['p_holm_family']=round(float(a),4); r['p_holm_global_with_revision_tests']=round(float(gg),4)
    r['status']='Robust' if a<0.05 and r['balanced_acc']['lr_content']>r['balanced_acc']['meccan_only'] else 'Preliminary'
    r['status_global']='Robust' if gg<0.05 and r['balanced_acc']['lr_content']>r['balanced_acc']['meccan_only'] else 'Borderline'
json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','expC_lexical.json'),'w'),ensure_ascii=False,indent=1,default=str)
for r in res: print(json.dumps({k:r[k] for k in r if k not in('forms','misclassified_examples')},ensure_ascii=False,default=str))
