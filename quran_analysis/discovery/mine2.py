"""Lemma-level data mining of natural entities in the Quran (no literature input).
Outputs: collocations (G2), word-order regularities (binomial), singular/plural context
splits (Fisher), and distinctive verbs predicated of each entity (G2).
Usage: python3 -I mine2.py <morph.txt> <scripts_dir>"""
import sys,collections,math,json; sys.path.insert(0,sys.argv[2]); from load import load_morph,norm
from math import comb
W=load_morph(sys.argv[1])
ENT="سماء ارض شمس قمر نجم كوكب ليل نهار صبح فجر نور ظلمه ظل جبل رواسي بحر موج مطر ماء سحاب ريح برق نبات شجر شجره زرع ثمره حب نخل زوج حياه موت نطفه علقه مضغه لحم دم طير دابه تراب طين حجاره حديد نار دخان فلك سمع بصر قلب صدر ارحام نهر سراج ضياء هواء".split()
ENT=set(ENT)
V=collections.defaultdict(list)
for (s,v,w),segs in W.items():
    for d in segs:
        if 'LEM' in d and d['tag'] in('N','V'):
            V[(s,v)].append((w,norm(d['LEM']),d['tag'],d['feat'],d['form']))
NV=len(V)
occ=collections.Counter(); pair=collections.Counter(); order=collections.Counter()
vocc=collections.Counter(); ev=collections.Counter(); evv=collections.defaultdict(list)
for k,toks in V.items():
    first={}
    for w,l,t,f,_ in toks:
        if t=='N' and l in ENT and l not in first: first[l]=w
    es=sorted(first); verbs={l for _,l,t,_,_ in toks if t=='V'}
    for e in es: occ[e]+=1
    for vb in verbs: vocc[vb]+=1
    for e in es:
        for vb in verbs: ev[(e,vb)]+=1; evv[(e,vb)].append('%d:%d'%k)
    for i,a in enumerate(es):
        for b in es[i+1:]:
            pair[(a,b)]+=1
            if first[a]<first[b]: order[(a,b)]+=1
            elif first[b]<first[a]: order[(b,a)]+=1
def G2(a,b,ab,N):
    k=[ab,a-ab,b-ab,N-a-b+ab]; r=[k[0]+k[1],k[2]+k[3]]; c=[k[0]+k[2],k[1]+k[3]]
    e=[r[0]*c[0]/N,r[0]*c[1]/N,r[1]*c[0]/N,r[1]*c[1]/N]
    return 2*sum(x*math.log(x/y) for x,y in zip(k,e) if x>0)
def binom2(x,n):
    return min(1,2*sum(comb(n,i) for i in range(0,min(x,n-x)+1))/2**n)
R={}
R['collocations']=sorted([(a,b,ab,round(math.log2(ab*NV/(occ[a]*occ[b])),2),round(G2(occ[a],occ[b],ab,NV),1))
    for (a,b),ab in pair.items() if ab>=3 and G2(occ[a],occ[b],ab,NV)>15.1 and ab*NV>occ[a]*occ[b]],key=lambda x:-x[4])
ords=[]
for (a,b),n in pair.items():
    x,y=order[(a,b)],order[(b,a)]
    if x+y>=5:
        p=binom2(x,x+y); A,B,k=(a,b,x) if x>=y else (b,a,y); ords.append((A,B,k,x+y,p))
# Bonferroni over tested pairs
m=len(ords); R['orderings']=sorted([(a,b,k,n,'%.1e'%p) for a,b,k,n,p in ords if p*m<0.05],key=lambda t:float(t[4]))
R['n_order_tests']=m
# distinctive verbs
pv=[]
for (e,vb),n in ev.items():
    if n>=3:
        g=G2(occ[e],vocc[vb],n,NV)
        if g>15.1 and n*NV>occ[e]*vocc[vb]: pv.append((e,vb,n,round(g,1),evv[(e,vb)][:6]))
R['verbs']=sorted(pv,key=lambda x:-x[3])
# singular vs plural context (punishment vocabulary)
PUN={norm(x) for x in ['عَذاب','أَهْلَكَ','دَمَّرَ','أَغْرَقَ','صاعِقَة','عَذَّبَ','عِقاب','غَضَب','عاصِف','رِجْز','حاصِب','صَرْصَر','عَقِيم','مُهْلِك']}
num=collections.defaultdict(collections.Counter)
for k,toks in V.items():
    pun=any(l in PUN for _,l,_,_,_ in toks)
    for w,l,t,f,form in toks:
        if t=='N' and l in ENT:
            n='P' if any(x in('MP','FP') for x in f) else ('D' if any(x in('MD','FD') for x in f) else 'S')
            num[(l,n)]['n']+=1; num[(l,n)]['pun']+=pun
def fisher(a,b,c,d):
    N=a+b+c+d; r1=a+b; c1=a+c
    def h(x): return comb(r1,x)*comb(N-r1,c1-x)/comb(N,c1)
    p0=h(a); return sum(h(x) for x in range(max(0,c1-(N-r1)),min(r1,c1)+1) if h(x)<=p0*(1+1e-9))
sp=[]
for e in ENT:
    s,p=num[(e,'S')],num[(e,'P')]
    if s['n']>=4 and p['n']>=4:
        sp.append((e,s['n'],s['pun'],p['n'],p['pun'],'%.2g'%fisher(s['pun'],s['n']-s['pun'],p['pun'],p['n']-p['pun'])))
R['number_split']=sorted(sp,key=lambda t:float(t[-1]))
json.dump(R,open('mined2.json','w'),ensure_ascii=False,indent=0)
print('verses',NV)
print('\n== collocations (a,b,verses,PMI,G2) top 40'); [print(x) for x in R['collocations'][:40]]
print('\n== orderings significant after Bonferroni over',m,'tests'); [print(x) for x in R['orderings']]
print('\n== distinctive verbs per entity top 60'); [print(x) for x in R['verbs'][:60]]
print('\n== singular/plural vs punishment context'); [print(x) for x in R['number_split'][:12]]
