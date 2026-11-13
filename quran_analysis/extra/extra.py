"""Four additional analyses:
A stylometry (Quran vs hadith vs hadith qudsi), B mutashabih near-duplicates & fasila adaptation,
C long-range correlations (DFA) in verse lengths / rhyme, D formulaic repetition density.
Usage: python3 -I extra.py <quran.json> <hadith_dir> <apcd.csv> <scripts_dir> <linguistics_dir>"""
import sys,os,re,csv,glob,json,math,random,collections,statistics as st
sys.path.insert(0,sys.argv[4]); sys.path.insert(0,sys.argv[5])
from load import norm
from prosody import encode,diacritic_coverage
random.seed(13)
Q=json.load(open(sys.argv[1])); R={}
def clean(t): return re.sub('[^ء-ي ]',' ',norm(t)).split()
qv=[(s['id'],v['id'],s['type'],v['text']) for s in Q for v in s['verses']]
# ---------- corpora ----------
quotes=[];prose=[];qudsi=[]
QUDSI=re.compile(r'(قال|يقول) الله (عز وجل|تعالي|تبارك وتعالي)?')
for f in glob.glob(os.path.join(sys.argv[2],'*','*mushakkala*.csv')):
    for r in csv.reader(open(f,encoding='utf8')):
        t=r[1].replace('‏',' ')
        for x in re.findall(r'\{([^{}]+)\}',t):
            if len(x.split())>=6: quotes.append(x)
        m=re.search('وَسَل[َّ]{2}مَ',t)
        if not m: continue
        body=re.sub(r'\{[^{}]*\}',' ',t[m.end():])
        nb=' '.join(re.sub('[^\u0621-\u064A ]',' ',norm(body)).split())
        q=QUDSI.search(nb)
        if q: qudsi.append(nb[q.end():])
        else: prose.append(body)
R['sizes']={'quran_quotes':len(quotes),'hadith_prose':len(prose),'hadith_qudsi_candidates':len(qudsi)}
# ---------- A. stylometry on function words, equal-length chunks (40 words) ----------
FW="و ف ثم او ام بل لا ما لم لن ان انما الا الذي التي الذين من في علي الي عن ب ل ك قد لقد هل يا اذا اذ لو لولا كان كل بعض هو هي هم انت انتم نحن انا ذلك هذا تلك هؤلاء عند حتي كما قال يوم الله رب".split()
FWi={w:i for i,w in enumerate(FW)}
def chunks(texts,n=40,limit=1500):
    toks=[w for t in texts for w in clean(t)]
    out=[toks[i:i+n] for i in range(0,len(toks)-n,n)]
    random.shuffle(out); return out[:limit]
def feat(ch):
    v=[0]*len(FW)
    for w in ch:
        # strip attached conj/prep prefixes into separate counts
        if w in FWi: v[FWi[w]]+=1
        elif len(w)>3 and w[0] in 'وف' : v[FWi[w[0]]]+=1
    return [x/len(ch) for x in v]
qc=chunks([t for *_,t in qv]); hc=chunks(prose); dc=chunks(qudsi,limit=800); qq=chunks(quotes,limit=800)
def zscore_fit(rows):
    cols=list(zip(*rows)); mu=[st.mean(c) for c in cols]; sd=[st.pstdev(c) or 1e-9 for c in cols]; return mu,sd
def nearest_centroid(train_a,train_b,test):
    mu,sd=zscore_fit(train_a+train_b)
    z=lambda v:[(x-m)/s for x,m,s in zip(v,mu,sd)]
    ca=[st.mean(c) for c in zip(*[z(v) for v in train_a])]; cb=[st.mean(c) for c in zip(*[z(v) for v in train_b])]
    d=lambda v,c:sum(abs(x-y) for x,y in zip(z(v),c))/len(c)   # Burrows-style Delta
    return [(d(v,ca),d(v,cb)) for v in test]
FA=[feat(c) for c in qc]; FB=[feat(c) for c in hc]
half=lambda L:(L[:len(L)//2],L[len(L)//2:])
a1,a2=half(FA); b1,b2=half(FB)
acc_q=sum(x<y for x,y in nearest_centroid(a1,b1,a2))/len(a2); acc_h=sum(y<x for x,y in nearest_centroid(a1,b1,b2))/len(b2)
qud=nearest_centroid(a1,b1,[feat(c) for c in dc]); quo=nearest_centroid(a1,b1,[feat(c) for c in qq])
R['A_stylometry']={'chunk_words':40,'accuracy_quran_vs_hadith':round((acc_q+acc_h)/2,3),
  'hadith_qudsi_classified_as_quran':round(sum(x<y for x,y in qud)/len(qud),3),'n_qudsi_chunks':len(qud),
  'quran_quotes_in_hadith_books_classified_as_quran':round(sum(x<y for x,y in quo)/len(quo),3)}
# internal homogeneity: Meccan vs Medinan distance vs Quran-hadith distance (Delta between centroids)
mec=[feat(c) for c in chunks([t for s,v,ty,t in qv if ty=='meccan'])]; med=[feat(c) for c in chunks([t for s,v,ty,t in qv if ty=='medinan'])]
def cdist(A,B):
    mu,sd=zscore_fit(A+B); z=lambda v:[(x-m)/s for x,m,s in zip(v,mu,sd)]
    ca=[st.mean(c) for c in zip(*[z(v) for v in A])]; cb=[st.mean(c) for c in zip(*[z(v) for v in B])]
    return sum(abs(x-y) for x,y in zip(ca,cb))/len(ca)
R['A_stylometry']['delta_meccan_vs_medinan']=round(cdist(mec,med),3)
R['A_stylometry']['delta_quran_vs_hadith']=round(cdist(FA,FB),3)
R['A_stylometry']['delta_qudsi_vs_hadith']=round(cdist([feat(c) for c in dc],FB),3)
R['A_stylometry']['delta_qudsi_vs_quran']=round(cdist([feat(c) for c in dc],FA),3)
# ---------- B. mutashabih near-duplicates ----------
def words(t): return clean(t)
V=[(s,v,words(t)) for s,v,_,t in qv]
def fkey(ws):
    if not ws: return ''
    w=ws[-1]; return (w[-2]+'ا') if w.endswith('ا') and len(w)>1 else w[-1]
idx=collections.defaultdict(set)
for i,(s,v,ws) in enumerate(V):
    for j in range(len(ws)-2): idx[tuple(ws[j:j+3])].add(i)
cand=collections.Counter()
for g,ids in idx.items():
    if 2<=len(ids)<=6:
        ids=sorted(ids)
        for a in range(len(ids)):
            for b in range(a+1,len(ids)): cand[(ids[a],ids[b])]+=1
import difflib
pairs=[]
for (i,j),c in cand.items():
    wi,wj=V[i][2],V[j][2]
    if min(len(wi),len(wj))<5: continue
    r=difflib.SequenceMatcher(None,wi,wj).ratio()
    if 0.7<=r<1.0: pairs.append((i,j,r))
# fasila adaptation test: for near-duplicates in DIFFERENT surahs whose final words differ,
# does each verse's ending match the dominant rhyme of its own surah more than the other's ending would?
dom={}
by=collections.defaultdict(list)
for s,v,ws in V: by[s].append(fkey(ws))
for s,l in by.items(): dom[s]=collections.Counter(l)
def local_rhyme(i,k=3):
    s=V[i][0]; ks=[fkey(V[x][2]) for x in range(max(0,i-k),min(len(V),i+k+1)) if x!=i and V[x][0]==s]; return collections.Counter(ks)
tests=0;own=0;swap=0;diffend=0;examples=[]
for i,j,r in pairs:
    if V[i][0]==V[j][0]: continue
    ei,ej=fkey(V[i][2]),fkey(V[j][2])
    if ei==ej: continue
    diffend+=1
    li,lj=local_rhyme(i),local_rhyme(j)
    if not li or not lj: continue
    tests+=1
    own+= (li[ei]>0)+(lj[ej]>0); swap+= (li[ej]>0)+(lj[ei]>0)
    if len(examples)<10: examples.append(('%d:%d'%V[i][:2],'%d:%d'%V[j][:2],round(r,2),V[i][2][-1],V[j][2][-1]))
R['B_mutashabih']={'near_duplicate_pairs':len(pairs),'cross_surah_pairs_with_different_endings':diffend,'tested':tests,
  'own_ending_matches_local_rhyme':own,'swapped_ending_would_match':swap,'share_own':round(own/(2*tests),3) if tests else None,'share_swapped':round(swap/(2*tests),3) if tests else None,'examples':examples}
from math import comb
# McNemar-like sign test on per-verse matches
# ---------- C. long-range correlations (DFA) ----------
def dfa(x,scales=(4,8,16,32,64)):
    m=st.mean(x); y=[];c=0
    for v in x: c+=v-m; y.append(c)
    F=[]
    for n in scales:
        segs=len(y)//n; 
        if segs<4: continue
        errs=[]
        for k in range(segs):
            seg=y[k*n:(k+1)*n]; t=list(range(n)); tm=(n-1)/2; sm=st.mean(seg)
            b=sum((ti-tm)*(si-sm) for ti,si in zip(t,seg))/sum((ti-tm)**2 for ti in t); a=sm-b*tm
            errs.append(st.mean((si-(a+b*ti))**2 for ti,si in zip(t,seg)))
        F.append((math.log(n),math.log(math.sqrt(st.mean(errs)))))
    xs,ys=zip(*F); xm,ym=st.mean(xs),st.mean(ys)
    return sum((a-xm)*(b-ym) for a,b in zip(xs,ys))/sum((a-xm)**2 for a in xs)
L=[len(ws) for _,_,ws in V]
alpha=dfa(L); sh=[]
for _ in range(20):
    M=L[:]; random.shuffle(M); sh.append(dfa(M))
rh=[1 if fkey(ws)=='ن' else 0 for _,_,ws in V]
alpha_r=dfa(rh); shr=[]
for _ in range(20):
    M=rh[:]; random.shuffle(M); shr.append(dfa(M))
# hadith control: lengths of consecutive 'sentences' not available -> use matn lengths in collection order
hl=[len(clean(t)) for t in prose[:6000]]
R['C_long_range']={'dfa_alpha_verse_length':round(alpha,3),'shuffled_mean':round(st.mean(sh),3),'shuffled_max':round(max(sh),3),
  'dfa_alpha_nun_rhyme':round(alpha_r,3),'rhyme_shuffled_mean':round(st.mean(shr),3),'dfa_alpha_hadith_matn_length':round(dfa(hl),3)}
# ---------- D. formulaic repetition ----------
poems=[]
with open(sys.argv[3],encoding='utf8') as f:
    for i,line in enumerate(f):
        txt,_,lab=line.rstrip('\n').rpartition(',')
        if lab!='نثر': poems.append(txt)
        if len(poems)>=60000: break
random.shuffle(poems)
def formulaic(texts,N=60000,n=4):
    toks=[w for t in texts for w in clean(t)][:N]
    grams=collections.Counter(tuple(toks[i:i+n]) for i in range(len(toks)-n))
    covered=sum(c for g,c in grams.items() if c>=3)
    return {'tokens':len(toks),'share_4grams_repeated_3plus':round(covered/max(1,len(toks)-n),3),
            'top':[' '.join(g)+' ×%d'%c for g,c in grams.most_common(5)]}
# Quran: sample contiguous 60000 tokens? use whole Quran but cap at 60000 from shuffled surahs to avoid order bias
qs=[t for *_,t in qv]
R['D_formulaic']={'quran':formulaic(qs),'hadith':formulaic(random.sample(prose,len(prose))),'poetry':formulaic(poems)}
refrains=collections.Counter(' '.join(ws) for _,_,ws in V if len(ws)>=3)
R['D_formulaic']['verbatim_repeated_verses']=[(k,c) for k,c in refrains.most_common(8) if c>=3]
json.dump(R,open('extra_results.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(R,ensure_ascii=False,indent=1))
