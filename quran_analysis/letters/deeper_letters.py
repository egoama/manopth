"""Deeper muqatta'at tests: (1) frequency-weighted null for Zamakhshari halves; (2) Massey single-order consistency;
(3) initials followed by mention of the Book; (4) coverage of the 14 letters.
Usage: python3 -I deeper_letters.py <quran.json> <morph.txt> <scripts_dir>"""
import sys,json,random,collections,statistics as st
sys.path.insert(0,sys.argv[3]); from load import load_morph,norm
from math import comb
random.seed(114)
Q=json.load(open(sys.argv[1])); W=load_morph(sys.argv[2])
ALPH=list('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
MAP={'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ء':'ا','ؤ':'و','ئ':'ي','ى':'ي','ة':'ه'}
def letters(t): return [MAP.get(c,c) for c in t if c in MAP or c in ALPH]
tot=collections.Counter(a for s in Q for v in s['verses'] for a in letters(v['text'])); T=sum(tot.values())
M=set('المصركهيعطسحقن')
CL={'mahmusa':set('فحثهشخصسكت'),'shadida':set('اجدقطبكت'),'mutbaqa':set('صضطظ'),'musta_liya':set('خصضغطقظ'),'qalqala':set('قطبجد')}
def ok(S): return all(len(S&c) in (len(c)//2,(len(c)+1)//2) for c in CL.values())
R={}
# (1) weighted sampling without replacement proportional to frequency
w=[tot[a] for a in ALPH]
def wsample(k):
    pool=list(ALPH); ww=[tot[a] for a in pool]; out=set()
    for _ in range(k):
        x=random.choices(range(len(pool)),weights=ww)[0]; out.add(pool.pop(x)); ww.pop(x)
    return out
N=200000; hitw=sum(ok(wsample(14)) for _ in range(N))
rank={a:i+1 for i,(a,_) in enumerate(tot.most_common())}
R['1_weighted_null']={'p_uniform(previous)':0.0255,'p_frequency_weighted':round(hitw/N,4),'samples':N,
  'frequency_ranks_of_muqattaat':sorted(rank[a] for a in M),'top14_most_frequent':''.join(a for a,_ in tot.most_common(14)),
  'overlap_with_top14':len(M&{a for a,_ in tot.most_common(14)})}
# (2) Massey single-order consistency
COMBOS=['الم','المص','الر','المر','كهيعص','طه','طسم','طس','يس','ص','حم','عسق','ق','ن']
# note: 'حم عسق' treated as two groups (as written: حمٓ / عٓسٓقٓ)
def acyclic(combos):
    edges=collections.defaultdict(set); nodes=set()
    for c in combos:
        nodes|=set(c)
        for i in range(len(c)):
            for j in range(i+1,len(c)): edges[c[i]].add(c[j])
    indeg={n:0 for n in nodes}
    for a in edges:
        for b in edges[a]: indeg[b]+=1
    q=[n for n in nodes if indeg[n]==0]; seen=0
    while q:
        n=q.pop(); seen+=1
        for b in edges[n]:
            indeg[b]-=1
            if indeg[b]==0: q.append(b)
    return seen==len(nodes)
obs=acyclic(COMBOS)
multi=[c for c in COMBOS if len(c)>1]
hits=0; NN=200000
for _ in range(NN):
    sh=[''.join(random.sample(c,len(c))) for c in COMBOS]
    hits+=acyclic(sh)
# derive one consistent order (topological)
def topo(combos):
    edges=collections.defaultdict(set); nodes=set()
    for c in combos:
        nodes|=set(c)
        for i in range(len(c)):
            for j in range(i+1,len(c)): edges[c[i]].add(c[j])
    indeg={n:0 for n in nodes}
    for a in edges:
        for b in edges[a]: indeg[b]+=1
    order=[]; q=sorted([n for n in nodes if indeg[n]==0],key=lambda x:-tot[x])
    while q:
        n=q.pop(0); order.append(n)
        for b in sorted(edges[n],key=lambda x:-tot[x]):
            indeg[b]-=1
            if indeg[b]==0: q.append(b)
    return ''.join(order)
# does the order follow descending frequency? count pairwise agreements
pairs=agree=0
for c in multi:
    for i in range(len(c)):
        for j in range(i+1,len(c)):
            pairs+=1; agree+= tot[c[i]]>tot[c[j]]
R['2_massey_order']={'combos':COMBOS,'consistent_with_single_order':obs,'p_random_within_combo_orders':round(hits/NN,5),
  'one_consistent_order':topo(COMBOS),'pairs_in_descending_frequency':f'{agree}/{pairs}'}
# (3) initials followed by mention of the Book in first 3 verses
BOOK={norm(x) for x in ['كِتاب','قُرْءان','نَزَّلَ','أَنزَلَ','تَنزِيل','آيَة','ذِكْر','وَحْي','أَوْحَى']}
first3=collections.defaultdict(set)
for (s,v,w),segs in W.items():
    if v<=3:
        for d in segs:
            if 'LEM' in d: first3[s].add(norm(d['LEM']))
INIT={2,3,7,10,11,12,13,14,15,19,20,26,27,28,29,30,31,32,36,38,40,41,42,43,44,45,46,50,68}
a=sum(1 for s in INIT if first3[s]&BOOK); b=len(INIT)-a
others=[s for s in range(1,115) if s not in INIT]; c=sum(1 for s in others if first3[s]&BOOK); d=len(others)-c
def fisher(a,b,c,d):
    N=a+b+c+d;r=a+b;k=a+c
    h=lambda x:comb(r,x)*comb(N-r,k-x)/comb(N,k); p0=h(a)
    return sum(h(x) for x in range(max(0,k-(N-r)),min(r,k)+1) if h(x)<=p0*(1+1e-9))
R['3_book_after_initials']={'initial_surahs_mentioning_book_in_v1_3':f'{a}/{len(INIT)}','other_surahs':f'{c}/{len(others)}','fisher_p':fisher(a,b,c,d),
  'initial_surahs_without':[s for s in sorted(INIT) if not first3[s]&BOOK]}
# (4) coverage
cov=sum(tot[x] for x in M)/T
R['4_coverage']={'share_of_all_letters_from_14_muqattaat':round(cov,3),'expected_if_random_14':round(14/28,3),'max_possible_top14':round(sum(c for _,c in tot.most_common(14))/T,3)}
json.dump(R,open('deeper_letters_results.json','w'),ensure_ascii=False,indent=1); print(json.dumps(R,ensure_ascii=False,indent=1))
