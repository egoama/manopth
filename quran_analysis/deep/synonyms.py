"""Near-synonym differentiation with leakage control (exact lemmas; all same-root lemmas removed from features).
Usage: python3 -I synonyms.py <morph.txt> <scripts_dir> <quran.json>"""
import sys,json,random,math,collections,statistics as st
sys.path.insert(0,sys.argv[2]); from load import load_morph,norm
from math import comb
random.seed(5); W=load_morph(sys.argv[1]); Q=json.load(open(sys.argv[3]))
txt={(s['id'],v['id']):v['text'] for s in Q for v in s['verses']}; typ={s['id']:s['type'] for s in Q}
lem=collections.defaultdict(list); lemroot={}
for (s,v,w),segs in W.items():
    for d in segs:
        if d['tag'] in('N','V') and 'LEM' in d and 'PRON' not in d['feat']:
            lem[(s,v)].append(d['LEM']); lemroot[d['LEM']]=d.get('ROOT','')
PAIRS=[('عام/سنة',['عام'],['سَنَة']),('غيث/مطر',['غَيْث'],['مَطَر']),('خوف/خشية',['خافَ','خَوْف'],['خَشِيَ','خَشْيَة']),
('قلب/فؤاد',['قَلْب'],['فُؤاد']),('إنسان/بشر',['إِنسان'],['بَشَر']),('جاء/أتى',['جاءَ'],['أَتَى']),('نظر/رأى',['نَظَرَ'],['رَأَى']),
('حلف/أقسم',['حَلَفْ'],['أَقْسَمُ','قَسَم','قَسَمْ']),('بعث/أرسل',['بَعَثَ'],['أَرْسَلَ']),('مرض/سقم',['مَرَض','مَرِيض'],['سَقِيم'])]
PUN={'عَذاب','جَهَنَّم','نار','سَعِير','جَحِيم','أَهْلَكَ','عِقاب','غَضَب','لَعْنَة','خِزْي','دَمَّرَ','أَغْرَقَ','حَمِيم','كَذَّبَ','كافِر','كَفَرَ','مُنافِق','كاذِب','فاسِق','ظالِم'}
def fisher(a,b,c,d):
    N=a+b+c+d;r=a+b;k=a+c
    h=lambda x:comb(r,x)*comb(N-r,k-x)/comb(N,k); p0=h(a)
    return sum(h(x) for x in range(max(0,k-(N-r)),min(r,k)+1) if h(x)<=p0*(1+1e-9))
out=[]
for name,A,B in PAIRS:
    ra={lemroot[x] for x in A+B if x in lemroot}
    VA={k for k,l in lem.items() if set(l)&set(A)}; VB={k for k,l in lem.items() if set(l)&set(B)}
    a_only,b_only=VA-VB,VB-VA
    feats=lambda k:{l for l in lem[k] if lemroot.get(l) not in ra}
    data=[(feats(k),0) for k in a_only]+[(feats(k),1) for k in b_only]
    res=None
    if min(len(a_only),len(b_only))>=5:
        V=len(set(l for f,_ in data for l in f))+1
        def acc(labels):
            cnt=[collections.Counter(),collections.Counter()]; tot=[0,0]; ny=[0,0]
            for (f,_),y in zip(data,labels):
                ny[y]+=1; tot[y]+=len(f)
                for l in f: cnt[y][l]+=1
            corr=0
            for (f,_),y in zip(data,labels):
                sc=[]
                for c in (0,1):
                    n_=ny[c]-(c==y); t_=tot[c]-(len(f) if c==y else 0); s_=math.log((n_+1)/(len(data)+1))
                    for l in f: s_+=math.log((cnt[c][l]-(c==y)+0.5)/(t_+0.5*V))
                    sc.append(s_)
                corr+=(sc[1]>sc[0])==(y==1)
            return corr/len(data)
        lab=[y for _,y in data]; obs=acc(lab); null=[]
        for _ in range(300):
            l2=lab[:]; random.shuffle(l2); null.append(acc(l2))
        maj=max(lab.count(0),lab.count(1))/len(lab)
        res={'loo_acc':round(obs,3),'majority_baseline':round(maj,3),'perm_null_mean':round(st.mean(null),3),'p':round((sum(x>=obs for x in null)+1)/301,4)}
    pa=sum(1 for k in a_only if set(lem[k])&PUN); pb=sum(1 for k in b_only if set(lem[k])&PUN)
    ca=collections.Counter(l for k in a_only for l in feats(k)); cb=collections.Counter(l for k in b_only for l in feats(k))
    na,nb=max(1,len(a_only)),max(1,len(b_only))
    da=sorted(ca,key=lambda l:-(ca[l]/na-cb[l]/nb)*math.sqrt(ca[l]))[:6]; db=sorted(cb,key=lambda l:-(cb[l]/nb-ca[l]/na)*math.sqrt(cb[l]))[:6]
    out.append({'pair':name,'verses':[len(a_only),len(b_only)],'classifier':res,
      'negative_context_share':[round(pa/na,2),round(pb/nb,2)],'negative_context_fisher_p':round(fisher(pa,na-pa,pb,nb-pb),4),
      'meccan_share':[round(sum(typ[k[0]]=='meccan' for k in a_only)/na,2),round(sum(typ[k[0]]=='meccan' for k in b_only)/nb,2)],
      'distinctive_ctx_a':[norm(x) for x in da],'distinctive_ctx_b':[norm(x) for x in db],
      'examples_a':['%d:%d'%k for k in sorted(a_only)[:6]],'examples_b':['%d:%d'%k for k in sorted(b_only)[:6]]})
json.dump(out,open('synonyms.json','w'),ensure_ascii=False,indent=1)
for o in out: print(json.dumps(o,ensure_ascii=False))
