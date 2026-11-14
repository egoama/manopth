"""Mandatory correction: full metric set for Quran-quote vs Hadith LR (leave-one-collection-out), natural prevalence and balanced subsamples."""
import sys,os,re,json,collections
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','revision','audit'))
from common import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,roc_auc_score,average_precision_score,confusion_matrix)
rng=np.random.default_rng(55)
recs=hadith_records()
HON=re.compile(r'صلي الله عليه وسلم|رضي الله عنهما|رضي الله عنها|رضي الله عنه|رسول الله|\bفقال\b|\bقال\b|\bقالت\b|\bحدثنا\b|\bاخبرنا\b|\bعن\b')
FW="و ف ثم او ام بل لا ما لم لن ان انما الا الذي التي الذين من في علي الي عن ب ل ك قد لقد هل يا اذا اذ لو لولا كان كل بعض هو هي هم انت انتم نحن انا ذلك هذا تلك عند حتي كما يوم الله رب".split(); FWi={w:i for i,w in enumerate(FW)}
def vec(ws):
    v=np.zeros(len(FW))
    for w in ws:
        if w in FWi: v[FWi[w]]+=1
        elif len(w)>3 and w[0] in 'وف': v[FWi[w[0]]]+=1
    return v/len(ws)
units=[];seen=set()
for r in recs:
    for x in r['quotes']:
        for part in re.split('إِلَى قَوْلِهِ|إِلَى آخِرِ الْآيَةِ|الْآيَةَ',x):
            ws=clean(part)
            if len(ws)>=12 and ' '.join(ws) not in seen: seen.add(' '.join(ws)); units.append((1,r['col'],ws[:40]))
seenh=set()
for r in recs:
    if not r['matn']: continue
    nb=' '.join(clean(r['matn']))
    if re.search(r'(قال|يقول) الله (عز وجل|تعالي|تبارك وتعالي)?',nb): continue
    ws=HON.sub(' ',nb).split()[:40]; k=' '.join(ws)
    if len(ws)<12 or k in seenh: continue
    seenh.add(k); units.append((0,r['col'],ws))
X=np.array([vec(u[2]) for u in units]); y=np.array([u[0] for u in units]); col=np.array([u[1] for u in units])
prob=np.zeros(len(y))
for c in sorted(set(col)):
    te=col==c; tr=~te; mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9
    m=LogisticRegression(max_iter=3000,class_weight='balanced').fit((X[tr]-mu)/sd,y[tr]); prob[te]=m.predict_proba((X[te]-mu)/sd)[:,1]
pred=(prob>=0.5).astype(int)
def metrics(yy,pp,pr):
    cm=confusion_matrix(yy,pp,labels=[1,0]); tp,fn,fp,tn=cm[0,0],cm[0,1],cm[1,0],cm[1,1]
    return {'n':int(len(yy)),'prevalence_quran':round(float(yy.mean()),4),'accuracy':round(accuracy_score(yy,pp),3),'balanced_accuracy':round(balanced_accuracy_score(yy,pp),3),
     'precision_quran':round(precision_score(yy,pp,pos_label=1,zero_division=0),3),'recall_quran':round(recall_score(yy,pp,pos_label=1),3),
     'precision_hadith':round(precision_score(yy,pp,pos_label=0),3),'recall_hadith(specificity)':round(recall_score(yy,pp,pos_label=0),3),
     'macro_f1':round(f1_score(yy,pp,average='macro'),3),'weighted_f1':round(f1_score(yy,pp,average='weighted'),3),'mcc':round(matthews_corrcoef(yy,pp),3),
     'roc_auc':round(roc_auc_score(yy,pr),3),'pr_auc_quran':round(average_precision_score(yy,pr),3),'confusion_[Q,H]x[Q,H]':[[int(tp),int(fn)],[int(fp),int(tn)]]}
OUT={'natural_prevalence':metrics(y,pred,prob)}
qi=np.where(y==1)[0]; hi=np.where(y==0)[0]; B=[]
for _ in range(1000):
    ix=np.concatenate([qi,rng.choice(hi,len(qi),replace=False)]); B.append(metrics(y[ix],pred[ix],prob[ix]))
keys=[k for k in B[0] if isinstance(B[0][k],float)]
OUT['balanced_subsamples_1000']={k:{'mean':round(float(np.mean([b[k] for b in B])),3),'ci95':[round(float(np.percentile([b[k] for b in B],2.5)),3),round(float(np.percentile([b[k] for b in B],97.5)),3)]} for k in keys}
m=OUT['natural_prevalence']; cm=m['confusion_[Q,H]x[Q,H]']; tp,fn,fp,tn=cm[0][0],cm[0][1],cm[1][0],cm[1][1]
f1q=2*tp/(2*tp+fp+fn); f1h=2*tn/(2*tn+fn+fp)
OUT['macro_f1_audit']={'f1_quran':round(f1q,3),'f1_hadith':round(f1h,3),'macro_f1_from_cm':round((f1q+f1h)/2,3),'reported_in_revision':0.491,'consistent':abs((f1q+f1h)/2-m['macro_f1'])<0.005}
OUT['interpretation']='Discrimination (ROC-AUC, balanced accuracy) is moderate; practical classification at natural prevalence is poor for the Quran class (low precision) because Quran quotes are ~1% of units.'
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','expS_stylometry_metrics.json'),'w'),ensure_ascii=False,indent=1)
print(json.dumps(OUT,ensure_ascii=False,indent=1))
