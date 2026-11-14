"""6c: frequency-preserving null for the 74 'opposite pair' equal-count test (original scripts/pairs.py).
For each original pair, draw a random lemma pair from the frequency window [0.5*min, 2*max] (as originally) -> count equal pairs in a 74-pair set;
repeat 20000x to get the null distribution of the NUMBER of equal pairs. Sensitivity: ALL-forms vs singular counting."""
import sys,os,re,json,collections,random
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','audit'))
from common import *
import numpy as np
rng=np.random.default_rng(74)
src=open(os.path.join(ROOT,'scripts','pairs.py'),encoding='utf8').read()
plist=re.findall(r"\('([^']+)','([^']+)'\)",src[src.find('pairs=['):src.find(']\nrows')])
W=load_morph(MORPH); T=collections.defaultdict(collections.Counter)
for segs in W.values():
    for d in segs:
        if 'LEM' in d and d['tag']!='P':
            n='P' if ('MP' in d['feat'] or 'FP' in d['feat']) else ('D' if ('MD' in d['feat'] or 'FD' in d['feat']) else 'S')
            T[norm(d['LEM'])][n]+=1; T[norm(d['LEM'])]['ALL']+=1
idx=collections.defaultdict(list)
for L in T: idx[norm(L)].append(L)
def c(q,mode): return sum(T[L][mode] for L in idx.get(norm(q),[]))
OUT={}
for mode in ('ALL','S'):
    obs=[(a,b,c(a,mode),c(b,mode)) for a,b in plist]
    eq=sum(1 for _,_,x,y in obs if x==y and x>0)
    allc=np.array([T[L][mode] for L in T if T[L][mode]>0])
    tested=[(x,y) for _,_,x,y in obs if x>0 and y>0]
    pools=[allc[(allc>=0.5*min(x,y))&(allc<=2*max(x,y))] for x,y in tested]
    sims=np.zeros(20000,int)
    for j,pool in enumerate(pools):
        a=rng.choice(pool,20000); b=rng.choice(pool,20000); sims+=(a==b)
    OUT[mode]={'pairs_listed':len(plist),'pairs_with_both_nonzero':len(tested),'observed_equal':eq,'equal_pairs':[(a,b,x) for a,b,x,y in obs if x==y and x>0],
      'null_mean':round(float(sims.mean()),2),'null_95_interval':[int(np.percentile(sims,2.5)),int(np.percentile(sims,97.5))],'p_ge_observed':round(float((sims>=eq).mean()),3)}
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','exp6c_pairs.json'),'w'),ensure_ascii=False,indent=1)
print(json.dumps(OUT,ensure_ascii=False,indent=1))
