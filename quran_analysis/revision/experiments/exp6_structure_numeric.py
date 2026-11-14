"""Experiment 6 (revised) + numerical-coincidence null.
6a DFA on verse lengths & nun-rhyme: 5000 within-surah permutations, scale sets x detrending order (1,2).
6b Mushaf vs chronological order smoothness: leave-one-feature-out sensitivity (5000 perms each).
6c Equal-frequency opposite pairs: frequency-preserving null distribution for 74 pairs + counting-rule sensitivity.
Acceptance (pre-set): DFA 'Robust' if observed alpha exceeds the 99th percentile of the within-surah null in all settings."""
import sys,os,re,json,math,random,collections
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','audit'))
from common import *
import numpy as np
rng=np.random.default_rng(6); random.seed(6)
Q=quran(); OUT={}
V=[(s['id'],clean(v['text'])) for s in Q for v in s['verses']]
L=np.array([len(ws) for _,ws in V],float); NUN=np.array([1.0 if ws[-1][-1]=='ن' else 0.0 for _,ws in V])
sid=np.array([s for s,_ in V]); blocks=[np.where(sid==s)[0] for s in range(1,115)]
def dfa_alpha(x,scales,order):
    y=np.cumsum(x-x.mean()); F=[]
    for n in scales:
        m=len(y)//n; Y=y[:m*n].reshape(m,n); t=np.arange(n)
        V_=np.vander(t,order+1); coef=np.linalg.lstsq(V_,Y.T,rcond=None)[0]; res=Y.T-V_@coef
        F.append(np.sqrt((res**2).mean()))
    return np.polyfit(np.log(scales),np.log(F),1)[0]
def within_perm(x):
    z=x.copy()
    for b in blocks: z[b]=x[rng.permutation(b)]
    return z
settings={'scales_4-64_order1':((4,8,16,32,64),1),'scales_8-128_order1':((8,16,32,64,128),1),'scales_4-64_order2':((4,8,16,32,64),2),'scales_16-256_order2':((16,32,64,128,256),2)}
P=5000; dfa={}
for name,x in (('verse_length',L),('nun_rhyme',NUN)):
    for sname,(sc,od) in settings.items():
        obs=dfa_alpha(x,sc,od); null=np.array([dfa_alpha(within_perm(x),sc,od) for _ in range(P)])
        dfa[f'{name}|{sname}']={'alpha':round(float(obs),3),'null_mean':round(float(null.mean()),3),'null_p99':round(float(np.percentile(null,99)),3),'p':(int((null>=obs).sum())+1)/(P+1)}
OUT['6a_dfa']=dfa; OUT['6a_decision']='Robust' if all(v['alpha']>v['null_p99'] for v in dfa.values()) else 'Preliminary'
# ---- 6b mushaf vs chronology ----
import glob
js=open(glob.glob(D+'/ext/qmeta/package/dist/HafsLists-*.js')[0],encoding='utf8').read(); i=js.find('SurahList'); seg=js[i:js.find(']\n\t]',i)+4]
rows=re.findall(r'\[\s*(-?\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*"([^"]*)",\s*(true|false)',seg)
rev={k:int(r[2]) for k,r in enumerate(rows) if 1<=k<=114}
W=load_morph(MORPH); typ={s['id']:s['type'] for s in Q}
lem=collections.defaultdict(list); tags=collections.defaultdict(collections.Counter); person=collections.defaultdict(collections.Counter); nw=collections.Counter()
for (s,v,w),segs in W.items():
    nw[s]+=1
    for d in segs:
        if 'LEM' in d: lem[s].append(norm(d['LEM']))
        for x in d['feat']:
            if ':' not in x: tags[s][x]+=1
        if d['tag']=='V':
            for x in d['feat']:
                if re.fullmatch('[123](M|F)?(S|D|P)',x): person[s][x[0]]+=1
F={}
for s in Q:
    k=s['id']; vs=[v['text'] for v in s['verses']]; enc=[encode(t) for t in vs]; ends=[clean(t)[-1][-1] for t in vs]
    c=collections.Counter(ends); H=-sum(n/len(ends)*math.log2(n/len(ends)) for n in c.values()); pc=person[k]; tot=sum(pc.values()) or 1
    F[k]=[sum(e.count('1') for e in enc)/sum(len(e) for e in enc),H,lem[k].count('الذي')/len(lem[k]),pc['2']/tot,ends.count('ن')/len(ends),tags[k]['IMPV']/nw[k],tags[k]['COND']/nw[k],tags[k]['EMPH']/nw[k]]
names=['share_mutaharrik','rhyme_entropy','alladhi_rate','2nd_person_share','nun_rhyme_share','impv_rate','cond_rate','emph_rate']
A=np.array([F[k] for k in range(1,115)]); Z=(A-A.mean(0))/A.std(0)
mec=[k for k in range(1,115) if typ[k]=='meccan']; med=[k for k in range(1,115) if typ[k]=='medinan']
def rough(order,cols): idx=np.array(order)-1; d=np.diff(Z[idx][:,cols],axis=0); return float((d**2).sum())
def ptest(order,cols,P=5000):
    obs=rough(order,cols); m_o=[k for k in order if typ[k]=='meccan']; d_o=[k for k in order if typ[k]=='medinan']; c=0
    for _ in range(P):
        perm=[]; im=iter(rng.permutation(m_o)); idd=iter(rng.permutation(d_o))
        for k in order: perm.append(next(im) if typ[k]=='meccan' else next(idd))
        c+= rough(perm,cols)<=obs
    return obs,(c+1)/(P+1)
trad=sorted(rev,key=lambda s:rev[s]); mush=list(range(1,115))
sens={}
for drop in [None]+list(range(len(names))):
    cols=[i for i in range(len(names)) if i!=drop]
    _,pm=ptest(mush,cols); _,pt=ptest(trad,cols)
    sens['all' if drop is None else 'without_'+names[drop]]={'mushaf_p_type_preserving':pm,'chronological_p_type_preserving':pt}
OUT['6b_order_sensitivity']=sens
OUT['6b_note']='Permutations preserve the Meccan/Medinan label at each position. Chronology = Egyptian standard order (quran-meta 7.0.0); it is itself partly reconstructed from traditional reports and stylistic criteria.'
# ---- 6c equal-frequency pairs null ----
Wm=W; T=collections.defaultdict(collections.Counter)
for segs in Wm.values():
    for d in segs:
        if 'LEM' in d and d['tag']!='P': T[norm(d['LEM'])]['ALL']+=1
counts=np.array([c['ALL'] for c in T.values()])
# reproduce original procedure: for 74 pairs, sample random lemma pairs with counts in [0.5*min,2*max] of the tested pair
pairs_counts=json.load(open(os.path.join(ROOT,'revision','results','orig_pair_counts.json'))) if os.path.exists(os.path.join(ROOT,'revision','results','orig_pair_counts.json')) else None
OUT['6c_note']='See exp6c_pairs.py for the frequency-preserving null of the 74-pair test.'
json.dump(OUT,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','exp6_structure.json'),'w'),ensure_ascii=False,indent=1)
print(json.dumps(OUT,ensure_ascii=False,indent=1))
