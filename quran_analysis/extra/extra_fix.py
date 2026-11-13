"""Corrected versions: A (orthography-neutral stylometry), B (sign test), C (within-surah shuffle), D (honorific-free hadith).
Usage: python3 -I extra_fix.py <quran.json> <hadith_dir> <scripts_dir> <linguistics_dir>"""
import sys,os,re,csv,glob,json,math,random,collections,statistics as st
sys.path.insert(0,sys.argv[3]); sys.path.insert(0,sys.argv[4])
from load import norm
from math import comb
random.seed(21)
Q=json.load(open(sys.argv[1])); R={}
def clean(t): return re.sub('[^ء-ي ]',' ',norm(t)).split()
qv=[(s['id'],v['id'],s['type'],v['text']) for s in Q for v in s['verses']]
quotes=[];prose=[];qudsi=[]
QUDSI=re.compile(r'(قال|يقول) الله (عز وجل|تعالي|تبارك وتعالي)?')
HON=re.compile(r'صلي الله عليه وسلم|رضي الله عنهما|رضي الله عنها|رضي الله عنه|رسول الله|قال|فقال|قالت')
for f in glob.glob(os.path.join(sys.argv[2],'*','*mushakkala*.csv')):
    for r in csv.reader(open(f,encoding='utf8')):
        t=r[1].replace('‏',' ')
        for x in re.findall(r'\{([^{}]+)\}',t):
            for part in re.split('إِلَى قَوْلِهِ|الْآيَةَ',x):
                if len(part.split())>=6: quotes.append(part)
        m=re.search('وَسَل[َّ]{2}مَ',t)
        if not m: continue
        nb=' '.join(clean(re.sub(r'\{[^{}]*\}',' ',t[m.end():])))
        q=QUDSI.search(nb)
        (qudsi if q else prose).append(nb[q.end():] if q else nb)
FW="و ف ثم او ام بل لا ما لم لن ان انما الا الذي التي الذين من في علي الي عن ب ل ك قد لقد هل يا اذا اذ لو لولا كان كل بعض هو هي هم انت انتم نحن انا ذلك هذا تلك عند حتي كما يوم الله رب".split()
FWi={w:i for i,w in enumerate(FW)}
def chunks(texts,n=40,limit=1500,strip=False):
    toks=[]
    for t in texts:
        if strip: t=HON.sub(' ',' '.join(clean(t)))
        toks+=clean(t)
    out=[toks[i:i+n] for i in range(0,len(toks)-n,n)]; random.shuffle(out); return out[:limit]
def feat(ch):
    v=[0]*len(FW)
    for w in ch:
        if w in FWi: v[FWi[w]]+=1
        elif len(w)>3 and w[0] in 'وف': v[FWi[w[0]]]+=1
    return [x/len(ch) for x in v]
def model(A,B):
    rows=A+B; cols=list(zip(*rows)); mu=[st.mean(c) for c in cols]; sd=[st.pstdev(c) or 1e-9 for c in cols]
    z=lambda v:[(x-m)/s for x,m,s in zip(v,mu,sd)]
    ca=[st.mean(c) for c in zip(*[z(v) for v in A])]; cb=[st.mean(c) for c in zip(*[z(v) for v in B])]
    def pred(v):
        zv=z(v); return sum(abs(x-y) for x,y in zip(zv,ca))<sum(abs(x-y) for x,y in zip(zv,cb))
    return pred
QQ=[feat(c) for c in chunks(quotes)]; HH=[feat(c) for c in chunks(prose,strip=True)]
DD=[feat(c) for c in chunks(qudsi,limit=800,strip=True)]; UU=[feat(c) for c in chunks([t for *_,t in qv],limit=800)]
qa,qb=QQ[:len(QQ)//2],QQ[len(QQ)//2:]; ha,hb=HH[:len(HH)//2],HH[len(HH)//2:]
pred=model(qa,ha)
acc=(sum(pred(v) for v in qb)+sum(not pred(v) for v in hb))/(len(qb)+len(hb))
R['A_stylometry_orthography_neutral']={'train':'Quran quotes in hadith books vs hadith prose (same orthography, honorifics removed)',
 'n_chunks':[len(QQ),len(HH),len(DD)],'accuracy':round(acc,3),
 'hadith_qudsi_classified_as_quran':round(sum(pred(v) for v in DD)/len(DD),3),
 'uthmani_quran_classified_as_quran':round(sum(pred(v) for v in UU)/len(UU),3)}
# ---------- B sign test ----------
V=[(s,v,clean(t)) for s,v,_,t in qv]
def fkey(ws):
    w=ws[-1]; return (w[-2]+'ا') if w.endswith('ا') and len(w)>1 else w[-1]
idx=collections.defaultdict(set)
for i,(s,v,ws) in enumerate(V):
    for j in range(len(ws)-2): idx[tuple(ws[j:j+3])].add(i)
cand=set()
for ids in idx.values():
    if 2<=len(ids)<=6:
        ids=sorted(ids); cand|={(a,b) for x,a in enumerate(ids) for b in ids[x+1:]}
import difflib
def local(i,k=3):
    s=V[i][0]; return collections.Counter(fkey(V[x][2]) for x in range(max(0,i-k),min(len(V),i+k+1)) if x!=i and V[x][0]==s)
disc_own=disc_swap=both=neither=0; n=0; ex=[]
for i,j in cand:
    wi,wj=V[i][2],V[j][2]
    if min(len(wi),len(wj))<5 or V[i][0]==V[j][0]: continue
    r=difflib.SequenceMatcher(None,wi,wj).ratio()
    if not (0.7<=r<1): continue
    ei,ej=fkey(wi),fkey(wj)
    if ei==ej: continue
    for x,own,oth in ((i,ei,ej),(j,ej,ei)):
        L=local(x); a=L[own]>0; b=L[oth]>0; n+=1
        if a and not b: disc_own+=1
        elif b and not a: disc_swap+=1
        elif a and b: both+=1
        else: neither+=1
    if len(ex)<12: ex.append(('%d:%d'%V[i][:2],V[i][2][-1],'%d:%d'%V[j][:2],V[j][2][-1]))
k=disc_own; m=disc_own+disc_swap
p=min(1,2*sum(comb(m,x) for x in range(k,m+1))/2**m) if m else None
R['B_mutashabih_sign_test']={'verse_tests':n,'own_only':disc_own,'swapped_only':disc_swap,'both':both,'neither':neither,'sign_test_p':p,'examples':ex}
# ---------- C within-surah shuffle ----------
def dfa(x,scales=(4,8,16,32,64)):
    m=st.mean(x); y=[];c=0
    for v in x: c+=v-m; y.append(c)
    F=[]
    for n_ in scales:
        segs=len(y)//n_
        errs=[]
        for k_ in range(segs):
            seg=y[k_*n_:(k_+1)*n_]; tm=(n_-1)/2; sm=st.mean(seg)
            b=sum((t-tm)*(s-sm) for t,s in enumerate(seg))/sum((t-tm)**2 for t in range(n_)); a=sm-b*tm
            errs.append(st.mean((s-(a+b*t))**2 for t,s in enumerate(seg)))
        F.append((math.log(n_),0.5*math.log(st.mean(errs))))
    xs,ys=zip(*F); xm,ym=st.mean(xs),st.mean(ys)
    return sum((a-xm)*(b-ym) for a,b in zip(xs,ys))/sum((a-xm)**2 for a in xs)
def within_shuffle(vals):
    by=collections.defaultdict(list)
    for (s,_,_),x in zip(V,vals): by[s].append(x)
    out=[]
    for s in sorted(by): l=by[s][:]; random.shuffle(l); out+=l
    return out
L=[len(ws) for *_,ws in V]; Rh=[1 if fkey(ws)=='ن' else 0 for *_,ws in V]
wsL=[dfa(within_shuffle(L)) for _ in range(20)]; wsR=[dfa(within_shuffle(Rh)) for _ in range(20)]
R['C_long_range_within_surah']={'alpha_length':round(dfa(L),3),'within_surah_shuffle_mean':round(st.mean(wsL),3),'within_surah_shuffle_max':round(max(wsL),3),
  'alpha_rhyme_nun':round(dfa(Rh),3),'rhyme_within_surah_shuffle_mean':round(st.mean(wsR),3),'rhyme_within_shuffle_max':round(max(wsR),3)}
# ---------- D honorific-free hadith ----------
def formulaic(texts,N=60000,n=4,strip=False):
    toks=[]
    for t in texts:
        if strip: t=HON.sub(' ',' '.join(clean(t)))
        toks+=clean(t)
        if len(toks)>N: break
    toks=toks[:N]; g=collections.Counter(tuple(toks[i:i+n]) for i in range(len(toks)-n))
    return round(sum(c for c in g.values() if c>=3)/(len(toks)-n),3)
random.shuffle(prose)
R['D_formulaic_fair']={'quran':formulaic([t for *_,t in qv]),'hadith_without_honorifics':formulaic(prose,strip=True)}
json.dump(R,open('extra_fix_results.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(R,ensure_ascii=False,indent=1))
