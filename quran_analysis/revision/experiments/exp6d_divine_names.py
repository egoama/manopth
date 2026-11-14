"""6d: divine-name verse endings predicted from content, grouped by surah (GroupKFold), same-root lemmas removed,
duplicate verse texts collapsed; balanced accuracy vs majority & permutation (stratified by surah type).
Acceptance (pre-set): 'Robust' if permutation p<0.01 and balanced accuracy > 1/5 + 0.05."""
import sys,os,json,collections
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','audit'))
from common import *
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
rng=np.random.default_rng(8)
Q=quran(); W=load_morph(MORPH); typ={s['id']:s['type'] for s in Q}
txt={(s['id'],v['id']):' '.join(clean(v['text'])) for s in Q for v in s['verses']}
words=collections.defaultdict(list); root_of={}
for (s,v,w),segs in W.items():
    words[(s,v)].append((w,segs))
    for d in segs:
        if 'LEM' in d: root_of[norm(d['LEM'])]=d.get('ROOT','')
content=lambda segs:[d for d in segs if d['tag'] in('N','V') and 'LEM' in d and 'PRON' not in d['feat']]
NAMES=[norm(x) for x in ['غَفُور','رَحِيم','عَزِيز','حَكِيم','عَلِيم','سَمِيع','بَصِير','خَبِير','لَطِيف','قَدِير','تَوّاب','غَنِيّ','حَمِيد','حَلِيم','واسِع','شَكُور','قَوِيّ','رَؤُوف','وَدُود','كَرِيم','عَظِيم','عَلِيّ','كَبِير','مَجِيد','قَرِيب','مُجِيب','شَهِيد','حَفِيظ','رَقِيب','وَكِيل','حَسِيب','مُحِيط']]
NS=set(NAMES); NR={root_of.get(n,'') for n in NAMES}-{''}
end={}
for k in sorted(words):
    ws=sorted(words[k]); tail=[]
    for _,segs in reversed(ws):
        c=content(segs)
        if c and norm(c[0]['LEM']) in NS: tail.insert(0,norm(c[0]['LEM']))
        elif c: break
        if len(tail)==2: break
    if len(tail)==2: end[k]=' '.join(tail)
cnt=collections.Counter(end.values()); keep=[e for e,c in cnt.items() if c>=12]
data=[];seen=set()
for k,e in sorted(end.items()):
    if e not in keep or txt[k] in seen: continue
    seen.add(txt[k])
    f={norm(d['LEM']):1 for _,segs in sorted(words[k])[:-3] for d in content(segs) if d.get('ROOT','') not in NR and norm(d['LEM'])!='الله'}
    if f: data.append((k,e,f))
X=DictVectorizer().fit_transform([f for *_,f in data]); y=np.array([e for _,e,_ in data]); g=np.array([k[0] for k,_,_ in data]); mec=np.array([typ[k[0]] for k,_,_ in data])
def cv(yy):
    pr=np.empty(len(yy),dtype=object)
    for tr,te in GroupKFold(5).split(X,yy,g):
        pr[te]=LogisticRegression(max_iter=3000,class_weight='balanced').fit(X[tr],yy[tr]).predict(X[te])
    return balanced_accuracy_score(yy,pr)
obs=cv(y); null=[]
for _ in range(300):
    yp=y.copy()
    for t in ('meccan','medinan'):
        ix=np.where(mec==t)[0]; yp[ix]=rng.permutation(y[ix])
    null.append(cv(yp))
p=(sum(x>=obs for x in null)+1)/301
OUT={'n_verses_dedup':len(data),'classes':{e:int((y==e).sum()) for e in keep},'balanced_acc_grouped':round(obs,3),'chance_balanced':0.2,'perm_null_mean':round(float(np.mean(null)),3),'p':p,
     'original_reported':'acc 0.436 (pooled, leave-one-out, not grouped)','decision':'Robust' if (p<0.01 and obs>0.25) else ('Preliminary' if p<0.05 else 'Not supported')}
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','exp6d_divine_names.json'),'w'),ensure_ascii=False,indent=1)
print(OUT)
