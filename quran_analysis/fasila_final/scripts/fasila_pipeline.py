"""Fasila final pipeline: pair audit, dependency graph, dependence-aware inference, null models N0-N3, robustness grid, expert package.
Usage: python3 -I scripts/fasila_pipeline.py   (QDATA env var optional)"""
import sys,os,json,csv,re,math,random,collections,difflib,itertools
H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(H,'..','..','revision','audit'))
from common import *
from math import comb
import numpy as np
rng=np.random.default_rng(2028); random.seed(2028)
OUTD=os.path.join(H,'..'); RES=os.path.join(OUTD,'results'); os.makedirs(RES,exist_ok=True)
Q=quran(); W=load_morph(MORPH)
RAW=[(s['id'],v['id'],v['text']) for s in Q for v in s['verses']]; V=[(s,v,clean(t)) for s,v,t in RAW]; NV=len(V)
NAME={s['id']:s['name'] for s in Q}; NVERSES={s['id']:s['total_verses'] for s in Q}
# ---------------- ending definitions (identical to revision_final/expA_endings.py) ----------------
MARKS=set('ًٌٍَُِّْٰۥۦ'); LET=lambda c:'ء'<=c<='ي' or c=='ٱ'
def last_word_struct(t):
    w=[x for x in t.split() if any(LET(c) for c in x)][-1]; base=[]
    for c in w:
        if LET(c): base.append([c,set()])
        elif base and (c in MARKS or 'ۖ'<=c<='ۭ' or 'ࣰ'<=c<='ࣲ'): base[-1][1].add(c)
    return base
def nrm(c): return {'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ء':'ا','ؤ':'و','ئ':'ي','ى':'ي','ة':'ه'}.get(c,c)
def defs(t):
    b=last_word_struct(t); L=[nrm(c) for c,_ in b]; last=L[-1]; m_last=b[-1][1]; tf=bool(m_last&{'ً','ࣰ'})
    d1=(L[-2]+'ا') if (last=='ا' and len(L)>1) else ('@'+last if tf else last)
    d2=(''.join(L[-3:-1])+'ا') if (last=='ا' and len(L)>2) else (''.join(L[-2:])+'@' if tf else ''.join(L[-2:]))
    if last=='ا' and len(L)>1: d3=('longa',L[-2])
    elif tf: d3=('longa',last)
    else:
        if len(b)>1:
            p,pm=b[-2]
            if nrm(p) in 'اوي' and not (pm&{'َ','ُ','ِ'}): vow={'ا':'longa','و':'longu','ي':'longi'}[nrm(p)]
            elif 'ٰ' in pm: vow='longa'
            else: vow={'َ':'a','ُ':'u','ِ':'i'}.get(next(iter(pm&{'َ','ُ','ِ'}),'x'),'none')
        else: vow='none'
        d3=(vow,last)
    son='son' if d3[1] in 'منلرويا' else 'obs'; d4=('long' if d3[0].startswith('long') else 'short',son)
    e=encode(' '.join(t.split()[-1:]),'sakin'); d5=(e[-3:],last)
    return {'D1_final_phoneme':d1,'D2_final_two':d2,'D3_final_syllable':d3,'D4_ending_pattern':d4,'D5_prosodic_tail':d5,'_len':len(L),'_son':son,'_vlen':d4[0]}
KEYS=[defs(t) for *_,t in RAW]; DEFS=['D1_final_phoneme','D2_final_two','D3_final_syllable','D4_ending_pattern','D5_prosodic_tail']
def kstr(k): return k if isinstance(k,str) else '|'.join(map(str,k))
# ---------------- candidate extraction (parametrised) ----------------
def candidates(seq=3,thr=0.70,minw=5):
    idx=collections.defaultdict(set)
    for i,(s,v,ws) in enumerate(V):
        for j in range(len(ws)-seq+1): idx[tuple(ws[j:j+seq])].add(i)
    cand=set()
    for ids in idx.values():
        if 2<=len(ids)<=6:
            ids=sorted(ids); cand|={(a,b) for x,a in enumerate(ids) for b in ids[x+1:]}
    out=[]
    for i,j in sorted(cand):
        wi,wj=V[i][2],V[j][2]
        if min(len(wi),len(wj))<minw: continue
        r=difflib.SequenceMatcher(None,wi,wj).ratio()
        if thr<=r<1: out.append((i,j,r))
    return out
def shared_seq(i,j):
    a,b=V[i][2],V[j][2]; m=difflib.SequenceMatcher(None,a,b).find_longest_match(0,len(a),0,len(b)); return ' '.join(a[m.a:m.a+m.size])
def neigh(x,k,d):
    s=V[x][0]; ids=[y for y in range(max(0,x-k),min(NV,x+k+1)) if y!=x and V[y][0]==s]
    return collections.Counter(KEYS[y][d] for y in ids),len(ids)
def classify(i,j,d,k):
    ei,ej=KEYS[i][d],KEYS[j][d]
    if ei==ej: return None
    li,ni=neigh(i,k,d); lj,nj=neigh(j,k,d)
    own=(li[ei]>0)+(lj[ej]>0); sw=(li[ej]>0)+(lj[ei]>0)
    return {'own_i':li[ei]>0,'own_j':lj[ej]>0,'sw_i':li[ej]>0,'sw_j':lj[ei]>0,'n_i':ni,'n_j':nj,'D':own-sw}
# ---------------- Phase A: full audit table ----------------
P=candidates(); cross=[(i,j,r) for i,j,r in P if V[i][0]!=V[j][0]]
deg=collections.Counter(x for i,j,_ in cross for x in (i,j))
rows=[]; eligible=[]
for i,j,r in P:
    reason='eligible'
    if V[i][0]==V[j][0]: reason='excluded: same surah (shared surah rhyme)'
    elif V[i][2][-1]==V[j][2][-1]: reason='excluded: identical final word'
    elif KEYS[i]['D1_final_phoneme']==KEYS[j]['D1_final_phoneme']: reason='excluded: identical final phoneme (not discriminable)'
    c=classify(i,j,'D1_final_phoneme',3) if reason=='eligible' else None
    rows.append({'pair_id':f'{V[i][0]}:{V[i][1]}~{V[j][0]}:{V[j][1]}','surah_i':V[i][0],'verse_i':V[i][1],'surah_j':V[j][0],'verse_j':V[j][1],
      'text_i':RAW[i][2],'text_j':RAW[j][2],'shared_sequence':shared_seq(i,j),'similarity':round(r,3),'ending_i':V[i][2][-1],'ending_j':V[j][2][-1],
      'D1_i':KEYS[i]['D1_final_phoneme'],'D1_j':KEYS[j]['D1_final_phoneme'],'D3_i':kstr(KEYS[i]['D3_final_syllable']),'D3_j':kstr(KEYS[j]['D3_final_syllable']),
      'inclusion':reason,'verse_i_in_multiple_pairs':deg[i]>1 if reason=='eligible' else '','verse_j_in_multiple_pairs':deg[j]>1 if reason=='eligible' else '',
      'neighbors_available_i_k3':c['n_i'] if c else '','neighbors_available_j_k3':c['n_j'] if c else '',
      'own_i_matches_local':c['own_i'] if c else '','own_j_matches_local':c['own_j'] if c else '','swapped_i_matches_local':c['sw_i'] if c else '','swapped_j_matches_local':c['sw_j'] if c else '',
      'pair_outcome_D1_k3':('favours_own' if c['D']>0 else 'favours_swapped' if c['D']<0 else 'tie') if c else ''})
    if reason=='eligible': eligible.append((i,j,r))
with open(os.path.join(OUTD,'ALL_VERSE_PAIRS.csv'),'w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(x) for x in rows]
with open(os.path.join(OUTD,'ELIGIBLE_PAIRS.csv'),'w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(x) for x in rows if x['inclusion']=='eligible']
audit={'candidates':len(P),'cross_surah':len(cross),'eligible':len(eligible),'exclusions':dict(collections.Counter(x['inclusion'] for x in rows if x['inclusion']!='eligible'))}
# ---------------- Phase B: dependency graph & dependence-aware inference ----------------
edeg=collections.Counter(x for i,j,_ in eligible for x in (i,j))
adj=collections.defaultdict(set)
for i,j,_ in eligible: adj[i].add(j); adj[j].add(i)
comp={}; cid=0
for n in sorted(adj):
    if n in comp: continue
    st=[n]; comp[n]=cid
    while st:
        u=st.pop()
        for v in adj[u]:
            if v not in comp: comp[v]=cid; st.append(v)
    cid+=1
pair_comp=[comp[i] for i,j,_ in eligible]
graph={'nodes':[{'id':'%d:%d'%V[x][:2],'component':comp[x],'degree':edeg[x]} for x in sorted(adj)],
       'edges':[{'i':'%d:%d'%V[i][:2],'j':'%d:%d'%V[j][:2],'similarity':round(r,3),'component':comp[i]} for i,j,r in eligible],
       'n_components':cid,'component_sizes_pairs':dict(collections.Counter(pair_comp)),'verses_in_multiple_pairs':{'%d:%d'%V[x][:2]:c for x,c in edeg.items() if c>1}}
json.dump(graph,open(os.path.join(OUTD,'DEPENDENCY_GRAPH.json'),'w'),ensure_ascii=False,indent=1)
def sign_p(a,b):
    n=a+b; return min(1,2*sum(comb(n,x) for x in range(max(a,b),n+1))/2**n) if n else 1.0
def wilson(k,n,z=1.96):
    if n==0: return (None,None)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (round((c-h)/d,3),round((c+h)/d,3))
def pair_outcomes(pairs,d,k):
    out=[]
    for i,j,r in pairs:
        c=classify(i,j,d,k)
        if c is None: continue
        out.append((i,j,c))
    return out
def dep_aware(pairs,d,k,label):
    po=pair_outcomes(pairs,d,k); Dv=np.array([c['D'] for _,_,c in po]); cp=np.array([comp[i] for i,_,_ in po])
    own=int((Dv>0).sum()); sw=int((Dv<0).sum()); tie=int((Dv==0).sum()); dec=own+sw
    # verse-level
    vl=[(c['own_i'],c['sw_i'],comp[i]) for i,_,c in po]+[(c['own_j'],c['sw_j'],comp[i]) for i,_,c in po]
    vo=sum(x[0] for x in vl); vs=sum(x[1] for x in vl); nv=len(vl)
    # component bootstrap
    comps=sorted(set(cp)); byc={c:np.where(cp==c)[0] for c in comps}
    B=10000; shares=[]; vshares=[]; diffs=[]
    vl_arr=np.array([(x[0],x[1]) for x in vl],float); vcomp=np.array([x[2] for x in vl])
    vbyc={c:np.where(vcomp==c)[0] for c in comps}
    for _ in range(B):
        pick=rng.choice(comps,len(comps))
        ix=np.concatenate([byc[c] for c in pick]); dv=Dv[ix]; o=(dv>0).sum(); s=(dv<0).sum()
        shares.append(o/(o+s) if o+s else np.nan)
        vx=np.concatenate([vbyc[c] for c in pick]); vshares.append(vl_arr[vx,0].mean()); diffs.append(vl_arr[vx,0].mean()-vl_arr[vx,1].mean())
    shares=np.array(shares); shares=shares[~np.isnan(shares)]
    # component-level sign-flip randomization: flip all pairs in a component together
    stat=own-sw; C=len(comps)
    def flipstat(mask):
        s=0
        for m,c in zip(mask,comps):
            s+= (-1 if m else 1)*int((Dv[byc[c]]>0).sum()-(Dv[byc[c]]<0).sum())
        return s
    if C<=20:
        vals=[flipstat(mask) for mask in itertools.product([0,1],repeat=C)]; pflip=sum(abs(v)>=abs(stat) for v in vals)/len(vals); mode='exact (2^%d)'%C
    else:
        vals=[flipstat(rng.integers(0,2,C)) for _ in range(100000)]; pflip=(sum(abs(v)>=abs(stat) for v in vals)+1)/(len(vals)+1); mode='MC 100000'
    return {'label':label,'pairs_tested':len(po),'favour_own':own,'favour_swapped':sw,'ties':tie,'share_own_of_decisive':round(own/dec,3) if dec else None,
            'share_own_ci95_component_bootstrap':[round(float(np.percentile(shares,2.5)),3),round(float(np.percentile(shares,97.5)),3)],
            'sign_test_p_pair_iid':sign_p(own,sw),'component_signflip_p':pflip,'component_signflip_mode':mode,'n_components':C,
            'verse_own_match':f'{vo}/{nv}','verse_own_rate':round(vo/nv,3),'verse_own_ci95_component_bootstrap':[round(float(np.percentile(vshares,2.5)),3),round(float(np.percentile(vshares,97.5)),3)],
            'verse_swapped_match':f'{vs}/{nv}','verse_swapped_rate':round(vs/nv,3),'own_minus_swapped_ci95':[round(float(np.percentile(diffs,2.5)),3),round(float(np.percentile(diffs,97.5)),3)],
            'verse_own_wilson_iid':wilson(vo,nv)}
def one_per_verse(pairs):
    used=set(); out=[]
    for i,j,r in sorted(pairs,key=lambda t:-t[2]):
        if i in used or j in used: continue
        used|={i,j}; out.append((i,j,r))
    return out
primary=dep_aware(eligible,'D1_final_phoneme',3,'primary D1 k=3')
opv=dep_aware(one_per_verse(eligible),'D1_final_phoneme',3,'one pair per verse D1 k=3')
# ---------------- Phase E: definitions x k, Holm over primary family (5 defs at k=3) ----------------
from statsmodels.stats.multitest import multipletests
grid={}
for d in DEFS:
    grid[d]={}
    for k in (1,2,3,5,7):
        po=pair_outcomes(eligible,d,k); Dv=[c['D'] for _,_,c in po]; o=sum(x>0 for x in Dv); s=sum(x<0 for x in Dv)
        grid[d][k]={'tested':len(po),'own':o,'swapped':s,'ties':len(Dv)-o-s,'p':sign_p(o,s)}
holm=multipletests([grid[d][3]['p'] for d in DEFS],method='holm')[1]
for d,h in zip(DEFS,holm): grid[d][3]['p_holm_primary_family']=round(float(h),6)
# ---------------- Phase C: null models (verse-level own-match vs alternative; D1, k=3) ----------------
verses=sorted({x for i,j,_ in eligible for x in (i,j)})
LOC={x:neigh(x,3,'D1_final_phoneme') for x in verses}
own_rate=np.mean([LOC[x][0][KEYS[x]['D1_final_phoneme']]>0 for x in verses])
glob=collections.Counter(KEYS[y]['D1_final_phoneme'] for y in range(NV))
def mc(pools,B=10000):
    """pools: dict verse-> list of candidate ending keys (with multiplicity). Returns null rates."""
    vs=[x for x in verses if pools.get(x)]; arrs={x:np.array(pools[x]) for x in vs}
    sims=np.array([np.mean([LOC[x][0][arrs[x][rng.integers(len(arrs[x]))]]>0 for x in vs]) for _ in range(B)])
    obs=np.mean([LOC[x][0][KEYS[x]['D1_final_phoneme']]>0 for x in vs])
    return {'verses_used':len(vs),'observed_own_match':round(float(obs),3),'null_mean':round(float(sims.mean()),3),'null_95':[round(float(np.percentile(sims,2.5)),3),round(float(np.percentile(sims,97.5)),3)],'p_ge_obs':round(float((sims>=obs).mean()),4)}
allkeys=[KEYS[y]['D1_final_phoneme'] for y in range(NV)]
N0=mc({x:allkeys for x in verses})
partner={}
for i,j,_ in eligible: partner.setdefault(i,[]).append(j); partner.setdefault(j,[]).append(i)
N1a=mc({x:[KEYS[y]['D1_final_phoneme'] for p in partner[x] for y in range(NV) if V[y][0]==V[p][0]] for x in verses})
N1b=mc({x:[KEYS[y]['D1_final_phoneme'] for y in range(NV) if V[y][0]==V[x][0] and abs(y-x)>3] for x in verses})
def lenb(n): return 'short' if n<=3 else 'mid' if n<=5 else 'long'
strata=collections.defaultdict(list)
for y in range(NV): strata[(KEYS[y]['_son'],KEYS[y]['_vlen'],lenb(KEYS[y]['_len']))].append(KEYS[y]['D1_final_phoneme'])
N2=mc({x:strata[(KEYS[x]['_son'],KEYS[x]['_vlen'],lenb(KEYS[x]['_len']))] for x in verses})
N2['matched_on']='final-consonant sonorant class, pre-final vowel length, final-word length bucket; outcome = exact final phoneme (not matched)'
# N3: morphology-matched verse-final words
def final_sig(s,v):
    segs=[d for (ss,vv,w),sg in W.items() if ss==s and vv==v for d in sg]  # slow path avoided below
    return None
# build final-word signature index efficiently
lastw=collections.defaultdict(int)
for (s,v,w) in W: lastw[(s,v)]=max(lastw[(s,v)],w)
sig={}
for (s,v,w),segs in W.items():
    if w!=lastw[(s,v)]: continue
    cont=[d for d in segs if d['tag'] in('N','V')]
    if not cont: sig[(s,v)]=None; continue
    d=cont[0]; f=set(d['feat'])
    pos=d['tag']+('|'+next((x for x in ['ACT_PCPL','PASS_PCPL','VN','ADJ','PN','PRON','REL','DEM'] if x in f),''))
    num=next((x for x in ['MS','FS','MD','FD','MP','FP','M','F','P'] if x in f),'')
    case=next((x for x in ['NOM','ACC','GEN'] if x in f),''); det='DET' if any('DET' in x['feat'] for x in segs) else ('INDEF' if 'INDEF' in f else '')
    sig[(s,v)]=(pos,num,case,det)
bysig=collections.defaultdict(list)
for (s,v),g in sig.items():
    if g: bysig[g].append((s,v))
VIDX={(V[y][0],V[y][1]):y for y in range(NV)}
pools3={}; poolsize={}
for x in verses:
    g=sig.get((V[x][0],V[x][1]))
    cands=[VIDX[k] for k in bysig.get(g,[]) if k!=(V[x][0],V[x][1]) and VIDX[k] not in verses] if g else []
    poolsize['%d:%d'%V[x][:2]]={'signature':g,'pool':len(cands)}
    if len(cands)>=5: pools3[x]=[KEYS[y]['D1_final_phoneme'] for y in cands]
N3=mc(pools3); N3['matched_on']='verse-final word morphology: POS/subtype, number-gender, case, definiteness (Quranic Arabic Corpus tags); pool = other verse-final words with same signature'; N3['verses_insufficient_pool']=len(verses)-len(pools3)
# pair-level null: own vs random (N0) and vs morphology-matched (N3)
def pair_null(pools,B=5000):
    po=pair_outcomes(eligible,'D1_final_phoneme',3); res=[]
    for _ in range(B):
        o=s=0
        for i,j,c in po:
            if i not in pools or j not in pools: continue
            ai=pools[i][rng.integers(len(pools[i]))]; aj=pools[j][rng.integers(len(pools[j]))]
            own=c['own_i']+c['own_j']; alt=(LOC[i][0][ai]>0)+(LOC[j][0][aj]>0)
            o+=own>alt; s+=alt>own
        res.append((o,s))
    o=np.array([r[0] for r in res]); s=np.array([r[1] for r in res])
    return {'mean_pairs_favour_own':round(float(o.mean()),2),'mean_pairs_favour_alternative':round(float(s.mean()),2),'share_own_mean':round(float((o/(o+s)).mean()),3),'share_own_95':[round(float(np.percentile(o/(o+s),2.5)),3),round(float(np.percentile(o/(o+s),97.5)),3)]}
PN0=pair_null({x:allkeys for x in verses}); PN3=pair_null(pools3)
# ---------------- Phase E2: robustness grid of extraction ----------------
rob=[]
for seq in (2,3,4):
    for thr in (0.60,0.65,0.70,0.75,0.80):
        for minw in (4,5,6):
            P2=candidates(seq,thr,minw); el=[(i,j,r) for i,j,r in P2 if V[i][0]!=V[j][0] and KEYS[i]['D1_final_phoneme']!=KEYS[j]['D1_final_phoneme']]
            po=pair_outcomes(el,'D1_final_phoneme',3); Dv=[c['D'] for _,_,c in po]; o=sum(x>0 for x in Dv); s=sum(x<0 for x in Dv)
            rob.append({'shared_seq':seq,'threshold':thr,'min_words':minw,'candidates':len(P2),'eligible':len(el),'own':o,'swapped':s,'ties':len(Dv)-o-s,'p':sign_p(o,s)})
# ---------------- counterexamples & expert package ----------------
po=pair_outcomes(eligible,'D1_final_phoneme',3)
ce=[{'pair':'%d:%d ~ %d:%d'%(V[i][0],V[i][1],V[j][0],V[j][1]),'ending_i':V[i][2][-1],'ending_j':V[j][2][-1],'own_i':c['own_i'],'own_j':c['own_j'],'sw_i':c['sw_i'],'sw_j':c['sw_j'],'n_i':c['n_i'],'n_j':c['n_j'],'D':c['D'],
     'local_D1_i':dict(neigh(i,3,'D1_final_phoneme')[0]),'local_D1_j':dict(neigh(j,3,'D1_final_phoneme')[0]),'text_i':RAW[i][2],'text_j':RAW[j][2]} for i,j,c in po if c['D']<=0]
json.dump(ce,open(os.path.join(RES,'counterexamples.json'),'w'),ensure_ascii=False,indent=1)
# expert package: 15 highest-similarity eligible pairs; each verse with original vs partner ending substituted; blinded order; key kept separately
top=sorted(eligible,key=lambda t:-t[2])[:15]; items=[]; key=[]
rnd=random.Random(99)
for n,(i,j,r) in enumerate(top,1):
    for x,y in ((i,j),(j,i)):
        words=RAW[x][2].split(); orig=words[-1]; alt=RAW[y][2].split()[-1]
        vA=' '.join(words[:-1]+[orig]); vB=' '.join(words[:-1]+[alt]); first=rnd.random()<0.5
        items.append({'item_id':f'P{n:02d}_{"i" if x==i else "j"}','verse_ref':'%d:%d'%V[x][:2],'surah_name':NAME[V[x][0]],'context_prev':RAW[x-1][2] if x>0 and V[x-1][0]==V[x][0] else '','context_next':RAW[x+1][2] if x+1<NV and V[x+1][0]==V[x][0] else '',
          'version_1':vA if first else vB,'version_2':vB if first else vA,'Q1_syntactically_correct_v1':'','Q1_syntactically_correct_v2':'','Q2_morphologically_correct_v1':'','Q2_morphologically_correct_v2':'',
          'Q3_preserves_proposition_v2_vs_v1':'','Q4_semantic_change':'','Q5_natural_classical_arabic_v1':'','Q5_natural_classical_arabic_v2':'','notes':''})
        key.append({'item_id':items[-1]['item_id'],'original_is_version':1 if first else 2,'original_ending':orig,'alternative_ending':alt,'partner_ref':'%d:%d'%V[y][:2]})
with open(os.path.join(OUTD,'EXPERT_REVIEW_PACKAGE.csv'),'w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(items[0].keys())); w.writeheader(); [w.writerow(x) for x in items]
with open(os.path.join(OUTD,'expert_review','ANSWER_KEY_DO_NOT_SHARE_WITH_ANNOTATORS.csv'),'w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(key[0].keys())); w.writeheader(); [w.writerow(x) for x in key]
# rhyme sheet (separate task, after grammar judgments)
with open(os.path.join(OUTD,'expert_review','RHYME_TASK.csv'),'w',newline='',encoding='utf8') as f:
    w=csv.writer(f); w.writerow(['item_id','verse_ref','neighbor_endings_k3','version_1_ending','version_2_ending','Q6_which_version_fits_local_pattern_better(1/2/equal)','notes'])
    for it,kk in zip(items,key):
        x=VIDX[tuple(map(int,it['verse_ref'].split(':')))]; nb=[V[y][2][-1] for y in range(max(0,x-3),min(NV,x+4)) if y!=x and V[y][0]==V[x][0]]
        w.writerow([it['item_id'],it['verse_ref'],' / '.join(nb),it['version_1'].split()[-1],it['version_2'].split()[-1],'',''])
OUT={'audit':audit,'graph_summary':{k:graph[k] for k in('n_components','component_sizes_pairs','verses_in_multiple_pairs')},'primary':primary,'one_pair_per_verse':opv,'definitions_grid':grid,
     'null_models':{'N0_global':N0,'N1a_partner_surah':N1a,'N1b_same_surah_outside_neighbourhood_DIAGNOSTIC':N1b,'N2_phonologically_matched':N2,'N3_morphology_matched':N3,'N3_pool_sizes':poolsize,'pair_level_N0':PN0,'pair_level_N3':PN3},
     'robustness_grid':rob,'counterexamples_n':len(ce),'expert_package_items':len(items)}
json.dump(OUT,open(os.path.join(RES,'fasila_results.json'),'w'),ensure_ascii=False,indent=1,default=str)
print(json.dumps({k:OUT[k] for k in ('audit','graph_summary','primary','one_pair_per_verse')},ensure_ascii=False,indent=1,default=str))
print(json.dumps({d:{k:(v['own'],v['swapped'],v['ties'],'%.2g'%v['p'],v.get('p_holm_primary_family','')) for k,v in g.items()} for d,g in grid.items()},ensure_ascii=False))
print(json.dumps({k:v for k,v in OUT['null_models'].items() if k!='N3_pool_sizes'},ensure_ascii=False,indent=1))
print('ROBUSTNESS'); [print(r) for r in rob if r['shared_seq']==3 or (r['threshold']==0.7 and r['min_words']==5)]
