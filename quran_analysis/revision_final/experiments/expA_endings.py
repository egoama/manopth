"""Priority A: verse endings in near-duplicate verses under 5 rhyme definitions, k in {1,2,3,5,7}, pair-level exact tests,
Holm over definitions at k=3, dependence sensitivity (one pair per verse), frequency-matched alternative-ending null."""
import sys,os,json,collections,difflib,math,re
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','revision','audit'))
from common import *
from math import comb
import numpy as np
from statsmodels.stats.multitest import multipletests
rng=np.random.default_rng(2027)
Q=quran()
RAW=[(s['id'],v['id'],v['text']) for s in Q for v in s['verses']]
V=[(s,v,clean(t)) for s,v,t in RAW]
MARKS=set('ًٌٍَُِّْٰۥۦ')
LET=lambda c:'ء'<=c<='ي' or c=='ٱ'
def last_word_struct(t):
    w=[x for x in t.split() if any(LET(c) for c in x)][-1]
    base=[]  # list of [letter, marks]
    for c in w:
        if LET(c): base.append([c,set()])
        elif base and (c in MARKS or 'ۖ'<=c<='ۭ' or 'ࣰ'<=c<='ࣲ'): base[-1][1].add(c)
    return base
def nrm(c): return {'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ء':'ا','ؤ':'و','ئ':'ي','ى':'ي','ة':'ه'}.get(c,c)
def defs(t):
    b=last_word_struct(t); L=[nrm(c) for c,_ in b]
    last=L[-1]; m_last=b[-1][1]
    tanwin_fath=bool(m_last&{'ً','ࣰ'})
    # D1 last phoneme (pausal): tanwin-fath -> long a ; alif after fath -> 'ا'
    d1=(L[-2]+'ا') if (last=='ا' and len(L)>1) else ('@'+last if tanwin_fath else last)
    # D2 last two letters (pausal)
    d2=(''.join(L[-3:-1])+'ا') if (last=='ا' and len(L)>2) else (''.join(L[-2:])+'@' if tanwin_fath else ''.join(L[-2:]))
    # D3 final syllable: (vowel before final consonant: long a/u/i or short a/u/i or none, final consonant)
    if last=='ا' and len(L)>1: d3=('longa',L[-2])
    elif tanwin_fath: d3=('longa',last)
    else:
        if len(b)>1:
            p,pm=b[-2]
            if nrm(p) in 'اوي' and not (pm&{'َ','ُ','ِ'}) : vow={'ا':'longa','و':'longu','ي':'longi'}[nrm(p)]
            elif 'ٰ' in pm: vow='longa'
            else: vow={'َ':'a','ُ':'u','ِ':'i'}.get(next(iter(pm&{'َ','ُ','ِ'}),'x'),'none')
        else: vow='none'
        d3=(vow,last)
    # D4 phonological pattern: (long/short, sonorant class of final consonant)
    son= 'son' if d3[1] in 'منلرويا' else 'obs'
    d4=('long' if d3[0].startswith('long') else 'short',son)
    # D5 prosodic tail: last 3 symbols of engine output for the last word (pausal sakin) + final consonant
    e=encode(' '.join(t.split()[-1:]),'sakin'); d5=(e[-3:],last)
    return {'last_phoneme':d1,'last_two':d2,'final_syllable':d3,'ending_pattern':d4,'prosodic_tail':d5}
KEYS=[defs(t) for *_,t in RAW]
# candidates (same rule as revision)
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
same_end=[(i,j,r) for i,j,r in cross if V[i][2][-1]==V[j][2][-1]]
analysed=[(i,j,r) for i,j,r in cross if KEYS[i]['last_phoneme']!=KEYS[j]['last_phoneme']]
OUT={'candidates':len(pairs),'cross_surah':len(cross),'excluded_same_surah':len(pairs)-len(cross),'excluded_identical_last_word':len(same_end),
     'excluded_same_last_phoneme_but_different_word':len(cross)-len(same_end)-len(analysed),'analysed':len(analysed),
     'exclusion_rule':'Pairs within one surah share the surah rhyme (uninformative); pairs with identical last phoneme cannot discriminate own vs swapped. Rule fixed before outcome inspection.'}
# dependence
deg=collections.Counter(x for i,j,_ in analysed for x in (i,j))
OUT['verses_in_multiple_pairs']={'%d:%d'%V[x][:2]:c for x,c in deg.items() if c>1}
def one_per_verse(P):
    used=set(); out=[]
    for i,j,r in sorted(P,key=lambda t:-t[2]):
        if i in used or j in used: continue
        used|={i,j}; out.append((i,j,r))
    return out
def sign_p(a,b):
    n=a+b; return min(1,2*sum(comb(n,x) for x in range(max(a,b),n+1))/2**n) if n else 1.0
def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-h)/d,(c+h)/d)
def run(P,dname,k):
    glob=collections.Counter(KEYS[x][dname] for x in range(len(V))); G=sum(glob.values())
    pf=pa=0; own=0; n=0; nullp=0.0; tested=0
    for i,j,r in P:
        ei,ej=KEYS[i][dname],KEYS[j][dname]
        if ei==ej: continue
        tested+=1
        def loc(x): 
            s=V[x][0]; return collections.Counter(KEYS[y][dname] for y in range(max(0,x-k),min(len(V),x+k+1)) if y!=x and V[y][0]==s)
        li,lj=loc(i),loc(j)
        so=(li[ei]>0)+(lj[ej]>0); ss=(li[ej]>0)+(lj[ei]>0)
        pf+=so>ss; pa+=ss>so
        for L,o in ((li,ei),(lj,ej)):
            n+=1; own+=L[o]>0; nullp+=sum(glob[e]/G for e in L)
    lo,hi=wilson(own,n)
    return {'pairs_tested':tested,'favour_own':pf,'favour_swapped':pa,'ties':tested-pf-pa,'p_sign':sign_p(pf,pa),'own_match':round(own/n,3) if n else None,'own_ci':[round(lo,3),round(hi,3)],'freq_null_match':round(nullp/n,3) if n else None}
DEFS=['last_phoneme','last_two','final_syllable','ending_pattern','prosodic_tail']
res={}
for d in DEFS:
    res[d]={k:run(analysed,d,k) for k in (1,2,3,5,7)}
OUT['by_definition']=res
ps=[res[d][3]['p_sign'] for d in DEFS]; holm=multipletests(ps,method='holm')[1]
OUT['holm_k3']={d:round(float(h),5) for d,h in zip(DEFS,holm)}
OUT['one_pair_per_verse']={d:run(one_per_verse(analysed),d,3) for d in DEFS}
# frequency-matched alternative endings (Monte Carlo): replace each verse's ending with a random ending drawn from global distribution, 10000x, k=3, last_phoneme
glob=collections.Counter(KEYS[x]['last_phoneme'] for x in range(len(V))); keys_=list(glob); w=np.array([glob[e] for e in keys_],float); w/=w.sum()
vs=sorted({x for i,j,_ in analysed for x in (i,j)})
locs=[]
for x in vs:
    s=V[x][0]; locs.append(set(KEYS[y]['last_phoneme'] for y in range(max(0,x-3),min(len(V),x+4)) if y!=x and V[y][0]==s))
own_obs=sum(KEYS[x]['last_phoneme'] in L for x,L in zip(vs,locs))/len(vs)
sims=[]
for _ in range(10000):
    draw=rng.choice(len(keys_),len(vs),p=w); sims.append(sum(keys_[d] in L for d,L in zip(draw,locs))/len(vs))
sims=np.array(sims)
OUT['frequency_matched_random_ending_null']={'observed_own_match':round(own_obs,3),'null_mean':round(float(sims.mean()),3),'null_95':[round(float(np.percentile(sims,2.5)),3),round(float(np.percentile(sims,97.5)),3)],'p_ge_obs':round(float((sims>=own_obs).mean()),4),'n_verses':len(vs)}
OUT['grammatical_alternatives']='NOT constructed. Requires expert-built syntactically valid alternative endings; flagged for expert review.'
OUT['examples']=[('%d:%d'%V[i][:2],V[i][2][-1],'%d:%d'%V[j][:2],V[j][2][-1],round(r,2)) for i,j,r in analysed]
ok=sum(h<0.01 for h in holm)>=3 and all(res[d][k]['favour_own']>res[d][k]['favour_swapped'] for d in DEFS for k in (1,2,3,5,7))
OUT['decision']='Robust' if ok else 'Preliminary'
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','expA_endings.json'),'w'),ensure_ascii=False,indent=1,default=str)
print(json.dumps({k:v for k,v in OUT.items() if k!='examples'},ensure_ascii=False,indent=1,default=str))
