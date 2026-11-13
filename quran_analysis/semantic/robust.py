"""A without same-root echoes; C Fisher tests on EMPH and IMPV.
Usage: python3 -I robust.py <quran.json> <morph.txt> <scripts_dir>"""
import sys,json,math,random,collections,statistics as st
sys.path.insert(0,sys.argv[3]); sys.path.insert(0,'.')
from load import load_morph,norm
from math import comb
import importlib.util
random.seed(41)
spec=importlib.util.spec_from_file_location('sem','semantic.py')
W=load_morph(sys.argv[2]); Q=json.load(open(sys.argv[1]))
root_of={}
for segs in W.values():
    for d in segs:
        if 'LEM' in d: root_of[norm(d['LEM'])]=d.get('ROOT','')
NAMES=[norm(x) for x in ['غَفُور','رَحِيم','عَزِيز','حَكِيم','عَلِيم','سَمِيع','بَصِير','خَبِير','لَطِيف','قَدِير','تَوّاب','غَنِيّ','حَمِيد','حَلِيم','واسِع','شَكُور','قَوِيّ','رَؤُوف','وَدُود','كَرِيم','عَظِيم','عَلِيّ','كَبِير','مَجِيد','قَرِيب','مُجِيب','شَهِيد','حَفِيظ','رَقِيب','وَكِيل','حَسِيب','مُحِيط']]
name_roots={root_of.get(n,'') for n in NAMES}-{''}
print('name roots excluded:',len(name_roots))
R=json.load(open('semantic_results.json'))
# rebuild A data with root exclusion
words=collections.defaultdict(list)
for (s,v,w),segs in W.items(): words[(s,v)].append((w,segs))
def content(segs): return [d for d in segs if d['tag'] in('N','V') and 'LEM' in d and 'PRON' not in d['feat']]
NS=set(NAMES); endings={}
for k in sorted(words):
    ws=sorted(words[k]); tail=[]
    for _,segs in reversed(ws):
        c=content(segs)
        if c and norm(c[0]['LEM']) in NS: tail.insert(0,norm(c[0]['LEM']))
        elif c: break
        if len(tail)==2: break
    if len(tail)==2: endings[k]=' '.join(tail)
cnt=collections.Counter(endings.values()); keep=[e for e,c in cnt.items() if c>=12]
data=[]
for k,e in endings.items():
    if e not in keep: continue
    ws=sorted(words[k])[:-3]
    f={norm(d['LEM']) for _,segs in ws for d in content(segs) if d.get('ROOT','') not in name_roots and norm(d['LEM'])!='الله'}
    if f: data.append((f,e))
V=len({l for f,_ in data for l in f})+1
def run(labels):
    cntc={c:collections.Counter() for c in keep}; tot={c:0 for c in keep}; ny={c:0 for c in keep}
    for (f,_),y in zip(data,labels):
        ny[y]+=1; tot[y]+=len(f)
        for l in f: cntc[y][l]+=1
    corr=0
    for (f,_),y in zip(data,labels):
        best=None
        for c in keep:
            n_=ny[c]-(c==y); t_=tot[c]-(len(f) if c==y else 0)
            s_=math.log(max(n_,1e-9)/(len(data)-1))+sum(math.log((cntc[c][l]-(c==y)+0.5)/(t_+0.5*V)) for l in f)
            if best is None or s_>best[0]: best=(s_,c)
        corr+=best[1]==y
    return corr/len(data)
lab=[y for _,y in data]; obs=run(lab); null=[]
for _ in range(300):
    l2=lab[:]; random.shuffle(l2); null.append(run(l2))
R['A_robust_no_same_root']={'n':len(data),'loo_acc':round(obs,3),'majority':round(max(collections.Counter(lab).values())/len(lab),3),'perm_null_mean':round(st.mean(null),3),'p':round((sum(x>=obs for x in null)+1)/301,4)}
def fisher(a,b,c,d):
    N=a+b+c+d;r=a+b;k=a+c
    h=lambda x:comb(r,x)*comb(N-r,k-x)/comb(N,k); p0=h(a)
    return sum(h(x) for x in range(max(0,k-(N-r)),min(r,k)+1) if h(x)<=p0*(1+1e-9))
g=R['C_character_voices']['feature_share_by_group']
nP,nO=g['P']['n'],g['O']['n']
out={}
for t in ('EMPH','IMPV'):
    a=round(g['P'][t]*nP); c=round(g['O'][t]*nO); out[t]={'prophets':f'{a}/{nP}','opponents':f'{c}/{nO}','fisher_p':round(fisher(a,nP-a,c,nO-c),5)}
R['C_feature_tests']=out
json.dump(R,open('semantic_results.json','w'),ensure_ascii=False,indent=1)
print(json.dumps({k:R[k] for k in ('A_robust_no_same_root','C_feature_tests')},ensure_ascii=False,indent=1))
