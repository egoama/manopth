"""Engine validation via meter identification on APCD (external meter labels).
Balanced classes, normalized n-gram profiles, nearest-centroid cosine; compares engine output with noise-corrupted output.
Acceptance (pre-set): validation counts as supportive if balanced accuracy >= 0.5 with 8 meters (chance 0.125)."""
import sys,os,re,json,random,math,collections
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','audit'))
from common import *
import numpy as np
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
random.seed(7)
by=collections.defaultdict(list)
with open(APCD,encoding='utf8') as f:
    for line in f:
        txt,_,lab=line.rstrip('\n').rpartition(',')
        if lab in('نثر',''): continue
        L=sum('ء'<=c<='ي' for c in txt); M=sum('ً'<=c<='ْ' for c in txt)
        if L and M/L>=0.85 and len(by[lab])<1200: by[lab].append(encode(txt,'ishba'))
meters=[m for m,v in by.items() if len(v)>=600]
K=4; grams=[''.join(x) for x in __import__('itertools').product('01',repeat=K)]
gi={g:i for i,g in enumerate(grams)}
def vec(s):
    v=np.zeros(len(grams)+1)
    for i in range(len(s)-K+1): v[gi[s[i:i+K]]]+=1
    v=v/max(1,v.sum()); v[-1]=len(s)/60; return v
def run(noise=0.0):
    tr={};te=[]
    for m in meters:
        seqs=by[m][:600]; random.shuffle(seqs)
        tr[m]=np.mean([vec(s) for s in seqs[:300]],0)
        for s in seqs[300:600]:
            if noise: s=''.join(('1' if c=='0' else '0') if random.random()<noise else c for c in s)
            te.append((m,vec(s)))
    C=np.array([tr[m] for m in meters])
    Cn=C/np.linalg.norm(C,axis=1,keepdims=True)
    y=[m for m,_ in te]; p=[meters[int(np.argmax(Cn@(v/np.linalg.norm(v))))] for _,v in te]
    return balanced_accuracy_score(y,p),confusion_matrix(y,p,labels=meters).tolist()
acc,cm=run(); accn,_=run(0.10); accs,_=run(0.30)
OUT={'meters':meters,'per_class_test':300,'balanced_accuracy':round(acc,3),'chance':round(1/len(meters),3),
     'balanced_accuracy_10pct_noise':round(accn,3),'balanced_accuracy_30pct_noise':round(accs,3),'confusion':cm,
     'decision':'supportive' if acc>=0.5 else 'not supportive'}
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','exp1b_engine_validation.json'),'w'),ensure_ascii=False,indent=1)
print({k:v for k,v in OUT.items() if k!='confusion'})
