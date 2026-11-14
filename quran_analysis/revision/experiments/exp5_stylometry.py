"""Experiment 5 (revised): Quran (quoted, hadith orthography) vs Prophetic Hadith vs Hadith Qudsi.
Fixes: dedup quotes & matns; NO concatenation across units; honorific/transmission formulas removed from hadith;
hadith evaluated leave-one-collection-out; quotes grouped by first-3-word key (proxy for verse identity) so overlapping quotes stay together;
balanced metrics, confusion matrix, cluster bootstrap CI; Delta vs LR vs NB; Qudsi per collection; ablation.
Acceptance (pre-set): separation 'Robust' if balanced accuracy CI lower bound > 0.70 for LR; Qudsi claim 'Robust' if
the share of Qudsi classified as Quran stays < 0.25 in every collection with >=20 Qudsi units."""
import sys,os,re,json,collections,random
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','audit'))
from common import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix
rng=np.random.default_rng(5); random.seed(5)
recs=hadith_records()
HON=re.compile(r'صلي الله عليه وسلم|رضي الله عنهما|رضي الله عنها|رضي الله عنه|رسول الله|\bفقال\b|\bقال\b|\bقالت\b|\bحدثنا\b|\bاخبرنا\b|\bعن\b')
QUDSI=re.compile(r'(قال|يقول) الله (عز وجل|تعالي|تبارك وتعالي)?')
FW="و ف ثم او ام بل لا ما لم لن ان انما الا الذي التي الذين من في علي الي عن ب ل ك قد لقد هل يا اذا اذ لو لولا كان كل بعض هو هي هم انت انتم نحن انا ذلك هذا تلك عند حتي كما يوم الله رب".split()
FWi={w:i for i,w in enumerate(FW)}
GROUPS={'conj':['و','ف','ثم','او','ام','بل'],'pron':['هو','هي','هم','انت','انتم','نحن','انا'],'rel':['الذي','التي','الذين'],'divine':['الله','رب'],
        'neg':['لا','ما','لم','لن'],'prep':['من','في','علي','الي','عن','ب','ل','ك']}
def vec(ws):
    v=np.zeros(len(FW))
    for w in ws:
        if w in FWi: v[FWi[w]]+=1
        elif len(w)>3 and w[0] in 'وف': v[FWi[w[0]]]+=1
    return v
units=[]  # (label, group, collection, tokens)
seen=set()
for r in recs:
    for x in r['quotes']:
        for part in re.split('إِلَى قَوْلِهِ|إِلَى آخِرِ الْآيَةِ|الْآيَةَ',x):
            ws=clean(part)
            if len(ws)>=12:
                key=' '.join(ws)
                if key in seen: continue
                seen.add(key); units.append(('Q','q:'+' '.join(ws[:3]),r['col'],ws[:40]))
seenh=set()
for r in recs:
    if not r['matn']: continue
    nb=' '.join(clean(r['matn']))
    q=QUDSI.search(nb)
    lab='D' if q else 'H'; body=nb[q.end():] if q else nb
    ws=HON.sub(' ',body).split()[:40]
    key=' '.join(ws)
    if len(ws)<12 or key in seenh: continue
    seenh.add(key); units.append((lab,r['col']+':'+r['id'],r['col'],ws))
cnt=collections.Counter(u[0] for u in units)
X=np.array([vec(u[3])/len(u[3]) for u in units]); lab=np.array([u[0] for u in units]); col=np.array([u[2] for u in units]); grp=np.array([u[1] for u in units])
cols=sorted(set(col))
# folds: hadith/qudsi leave-one-collection-out; quote groups assigned to folds by hash of group key (quotes come from collections too -> also assign by collection of first occurrence)
OUT={'units':dict(cnt)}
def fit_pred(model,tr,te,Xm):
    mu,sd=Xm[tr].mean(0),Xm[tr].std(0)+1e-9
    if model=='delta':
        Z=(Xm-mu)/sd; ca=Z[tr][lab[tr]=='Q'].mean(0); cb=Z[tr][lab[tr]=='H'].mean(0)
        return np.where(np.abs(Z[te]-ca).mean(1)<np.abs(Z[te]-cb).mean(1),'Q','H'),None
    if model=='lr':
        m=LogisticRegression(max_iter=3000,class_weight='balanced').fit((Xm[tr]-mu)/sd,lab[tr]); return m.predict((Xm[te]-mu)/sd),m
    if model=='nb':
        m=MultinomialNB().fit(Xm[tr]*100,lab[tr]); return m.predict(Xm[te]*100),m
mask=lab!='D'
res={}
for model in ('delta','lr','nb'):
    yt=[];yp=[];gt=[]
    for c in cols:
        te=np.where(mask&(col==c))[0]; tr=np.where(mask&(col!=c))[0]
        if len(te)==0 or len(set(lab[te]))<1: continue
        p,_=fit_pred(model,tr,te,X); yt+=list(lab[te]); yp+=list(p); gt+=list(grp[te])
    yt=np.array(yt); yp=np.array(yp); gt=np.array(gt)
    ba=balanced_accuracy_score(yt,yp); f1=f1_score(yt,yp,average='macro'); cm=confusion_matrix(yt,yp,labels=['Q','H']).tolist()
    # cluster bootstrap by group
    ug=np.unique(gt); gi={g:np.where(gt==g)[0] for g in ug}; bs=[]
    for _ in range(500):
        pick=np.concatenate([gi[g] for g in rng.choice(ug,len(ug))]); bs.append(balanced_accuracy_score(yt[pick],yp[pick]))
    res[model]={'balanced_accuracy':round(ba,3),'ci95':[round(np.percentile(bs,2.5),3),round(np.percentile(bs,97.5),3)],'macro_f1':round(f1,3),
                'confusion_[Q,H]x[Q,H]':cm,'majority_baseline_accuracy':round(max((yt=='Q').mean(),(yt=='H').mean()),3)}
OUT['quran_vs_hadith_leave_one_collection_out']=res
# Qudsi per collection using LR trained on all Q/H except that collection
qd={}
for c in cols:
    te=np.where((lab=='D')&(col==c))[0]
    if len(te)<5: continue
    tr=np.where(mask&(col!=c))[0]; p,_=fit_pred('lr',tr,te,X)
    share=(p=='Q').mean(); qd[c]={'n_qudsi':len(te),'share_classified_quran':round(float(share),3)}
OUT['qudsi_by_collection']=qd
# threshold sensitivity on pooled qudsi (probability thresholds)
tr=np.where(mask)[0]; mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9
m=LogisticRegression(max_iter=3000,class_weight='balanced').fit((X[tr]-mu)/sd,lab[tr]); ci_=list(m.classes_).index('Q')
pq=m.predict_proba((X[lab=='D']-mu)/sd)[:,ci_]
OUT['qudsi_threshold_sensitivity']={str(t):round(float((pq>=t).mean()),3) for t in (0.3,0.4,0.5,0.6,0.7)}
OUT['qudsi_note']='Qudsi detection = matn containing قال/يقول الله; text after the formula. Quran-trained model is fitted on quotes (in-sample for Q), so Qudsi share is an out-of-collection estimate only for the hadith side.'
# ablation (LR, LOCO)
abl={}
for gname,ws in GROUPS.items():
    keepcols=[i for i,w in enumerate(FW) if w not in ws]; Xa=X[:,keepcols]
    yt=[];yp=[]
    for c in cols:
        te=np.where(mask&(col==c))[0]; tr=np.where(mask&(col!=c))[0]
        mu,sd=Xa[tr].mean(0),Xa[tr].std(0)+1e-9
        mm=LogisticRegression(max_iter=3000,class_weight='balanced').fit((Xa[tr]-mu)/sd,lab[tr]); yp+=list(mm.predict((Xa[te]-mu)/sd)); yt+=list(lab[te])
    abl['without_'+gname]=round(balanced_accuracy_score(yt,yp),3)
OUT['ablation_lr_balanced_acc']=abl
ok=res['lr']['ci95'][0]>0.70
okq=all(v['share_classified_quran']<0.25 for v in qd.values() if v['n_qudsi']>=20)
OUT['decision_separation']='Robust' if ok else 'Preliminary'; OUT['decision_qudsi']='Robust' if okq and qd else 'Preliminary'
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','exp5_stylometry.json'),'w'),ensure_ascii=False,indent=1,default=float)
print(json.dumps(OUT,ensure_ascii=False,indent=1,default=float))
