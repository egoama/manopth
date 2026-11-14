"""Experiment 3 (revised): near-duplicate verses and ending adaptation to local rhyme.
Transparent rule: all candidate pairs (3-gram seed, 0.70<=ratio<1.0, both >=5 words); analysed = cross-surah pairs with different final keys.
Unit of analysis = pair (dependence within pair handled by pair-level outcome).
Frequency-aware null: probability that a random ending drawn from the global ending distribution matches the local rhyme.
Neighbourhood sensitivity k=1,2,3,5,7.
Acceptance (pre-set): 'Robust' if pair-level exact sign test p<0.01 for k in {2,3,5} and own-match exceeds frequency null."""
import sys,os,json,collections,difflib,math
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','audit'))
from common import *
from math import comb
import numpy as np
rng=np.random.default_rng(3)
Q=quran(); V=[(s['id'],v['id'],clean(v['text'])) for s in Q for v in s['verses']]
def fkey(ws):
    w=ws[-1]; return (w[-2]+'ا') if w.endswith('ا') and len(w)>1 else w[-1]
idx=collections.defaultdict(set)
for i,(s,v,ws) in enumerate(V):
    for j in range(len(ws)-2): idx[tuple(ws[j:j+3])].add(i)
cand=set()
for ids in idx.values():
    if 2<=len(ids)<=6:
        ids=sorted(ids); cand|={(a,b) for x,a in enumerate(ids) for b in ids[x+1:]}
pairs=[]
for i,j in sorted(cand):
    wi,wj=V[i][2],V[j][2]
    if min(len(wi),len(wj))<5: continue
    r=difflib.SequenceMatcher(None,wi,wj).ratio()
    if 0.7<=r<1: pairs.append((i,j,r))
cross=[(i,j,r) for i,j,r in pairs if V[i][0]!=V[j][0]]
analysed=[(i,j,r) for i,j,r in cross if fkey(V[i][2])!=fkey(V[j][2])]
glob=collections.Counter(fkey(ws) for *_,ws in V); G=sum(glob.values())
def local(i,k):
    s=V[i][0]; return collections.Counter(fkey(V[x][2]) for x in range(max(0,i-k),min(len(V),i+k+1)) if x!=i and V[x][0]==s)
def sign_p(a,b):
    n=a+b; return min(1,2*sum(comb(n,x) for x in range(max(a,b),n+1))/2**n) if n else 1.0
def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-h)/d,(c+h)/d)
OUT={'candidate_pairs':len(pairs),'cross_surah':len(cross),'analysed_pairs_diff_endings':len(analysed),'same_surah_pairs':len(pairs)-len(cross)}
sens={}
for k in (1,2,3,5,7):
    own_only=sw_only=both=neither=0; pf=pa=0; own_hits=0; verses=0; null_exp=0.0
    for i,j,r in analysed:
        ei,ej=fkey(V[i][2]),fkey(V[j][2]); li,lj=local(i,k),local(j,k)
        if not li or not lj: continue
        s_own=(li[ei]>0)+(lj[ej]>0); s_sw=(li[ej]>0)+(lj[ei]>0)
        if s_own>s_sw: pf+=1
        elif s_sw>s_own: pa+=1
        for L,o,a_ in ((li,ei,ej),(lj,ej,ei)):
            verses+=1; own_hits+=L[o]>0
            a=L[o]>0; b=L[a_]>0
            own_only+=a and not b; sw_only+=b and not a; both+=a and b; neither+=(not a) and (not b)
            null_exp+=sum(glob[e]/G for e in L)  # P(random ending in local set)
    lo,hi=wilson(own_hits,verses)
    sens[k]={'pairs_favour_own':pf,'pairs_favour_swapped':pa,'pair_sign_p':sign_p(pf,pa),'verse_own_only':own_only,'verse_swapped_only':sw_only,'both':both,'neither':neither,
             'own_match_rate':round(own_hits/verses,3),'own_match_ci95':[round(lo,3),round(hi,3)],'frequency_null_match_rate':round(null_exp/verses,3)}
OUT['by_neighbourhood']=sens
# counterexamples at k=3
ce=[]
for i,j,r in analysed:
    ei,ej=fkey(V[i][2]),fkey(V[j][2]); li,lj=local(i,3),local(j,3)
    if (li[ej]>0 and not li[ei]>0) or (lj[ei]>0 and not lj[ej]>0): ce.append(('%d:%d'%V[i][:2],V[i][2][-1],'%d:%d'%V[j][:2],V[j][2][-1]))
OUT['counterexamples_k3']=ce
ok=all(sens[k]['pair_sign_p']<0.01 for k in (2,3,5)) and all(sens[k]['own_match_rate']>sens[k]['frequency_null_match_rate'] for k in (2,3,5))
OUT['decision']='Robust' if ok else 'Preliminary'
OUT['not_done']='Controls with alternative grammatical endings require expert-constructed variants; not performed.'
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','exp3_mutashabih.json'),'w'),ensure_ascii=False,indent=1)
print(json.dumps(OUT,ensure_ascii=False,indent=1))
