"""Deep letter-count analysis (two counting definitions):
  RASM  = written skeleton letters of the Uthmani (Tanzil) text; hamza forms and alif-wasla map to alif; ة->ه, ى->ي; dagger/small letters ignored.
  PRON  = RASM + dagger alif (U+0670) counted as alif, small waw/ya (U+06E5/6) counted.
Tests: Zamakhshari half-classes; initial-letter enrichment; Ha-Mim & other groups; 19-claims; pangram verses; al-Fatiha missing letters; middle letter.
Usage: python3 -I letters.py <quran.json>"""
import sys,json,random,math,collections,statistics as st
random.seed(19)
Q=json.load(open(sys.argv[1]))
ALPH=list('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
MAP={'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ء':'ا','ؤ':'و','ئ':'ي','ى':'ي','ة':'ه'}
def letters(t,mode='RASM'):
    out=[]
    for ch in t:
        if ch in MAP: out.append(MAP[ch])
        elif ch in ALPH: out.append(ch)
        elif mode=='PRON' and ch=='ٰ': out.append('ا')
        elif mode=='PRON' and ch=='ۥ': out.append('و')
        elif mode=='PRON' and ch=='ۦ': out.append('ي')
    return out
INIT={2:'الم',3:'الم',7:'المص',10:'الر',11:'الر',12:'الر',13:'المر',14:'الر',15:'الر',19:'كهيعص',20:'طه',26:'طسم',27:'طس',28:'طسم',29:'الم',30:'الم',31:'الم',32:'الم',36:'يس',38:'ص',40:'حم',41:'حم',42:'حمعسق',43:'حم',44:'حم',45:'حم',46:'حم',50:'ق',68:'ن'}
R={}
# ---------- 1. Zamakhshari half-classes ----------
M=set('المصركهيعطسحقن')
CLASSES={'mahmusa':set('فحثهشخصسكت'),'shadida':set('اجدقطبكت'),'mutbaqa':set('صضطظ'),'musta_liya':set('خصضغطقظ'),'qalqala':set('قطبجد')}
def halves(S):
    res={}
    for c,cl in CLASSES.items():
        k=len(S&cl); n=len(cl); res[c]=(k,n,k in (n//2,(n+1)//2))
    return res
obs=halves(M)
N=1_000_000; hit=0
for _ in range(N):
    S=set(random.sample(ALPH,14))
    if all(v[2] for v in halves(S).values()): hit+=1
R['1_zamakhshari']={'muqattaat_letters':''.join(sorted(M)),'n_distinct':len(M),'classes':{c:f'{k}/{n}' for c,(k,n,_) in obs.items()},
  'all_halves':all(v[2] for v in obs.values()),'p_random_14_subset_all_halves':hit/N,'monte_carlo':N}
# also: prob that a random 14-subset hits halves in mahmusa AND shadida only (the two main partitions)
# ---------- 2. initial-letter enrichment in own surah ----------
def surah_counts(mode):
    out={}
    for s in Q:
        L=[]
        for v in s['verses']: L+=letters(v['text'],mode)
        # remove the initials themselves (first verse prefix) for surahs with initials
        if s['id'] in INIT:
            ini=letters(INIT[s['id']]); 
            if L[:len(ini)]==ini: L=L[len(ini):]
        out[s['id']]=collections.Counter(L), len(L)
    return out
SC=surah_counts('RASM')
freq={s:{a:c[a]/n for a in ALPH} for s,(c,n) in SC.items()}
pct={}
for s,ini in INIT.items():
    for a in set(ini):
        vals=[freq[x][a] for x in freq if SC[x][1]>=500]
        pct[(s,a)]=sum(v<freq[s][a] for v in vals)/len(vals)
mean_pct=st.mean(pct.values())
# null: replace each surah's initial-letter set by a random letter set of same size
null=[]
pairs=[(s,len(set(ini))) for s,ini in INIT.items()]
big=[x for x in freq if SC[x][1]>=500]
pre={a:sorted(freq[x][a] for x in big) for a in ALPH}
import bisect
def pr(s,a): return bisect.bisect_left(pre[a],freq[s][a])/len(pre[a])
for _ in range(20000):
    vals=[]
    for s,k in pairs:
        for a in random.sample(ALPH,k): vals.append(pr(s,a))
    null.append(st.mean(vals))
p2=(sum(x>=mean_pct for x in null)+1)/(len(null)+1)
top=sorted(pct.items(),key=lambda x:-x[1])
R['2_initial_enrichment']={'mean_percentile_of_initial_letters_in_own_surah':round(mean_pct,3),'null_mean':round(st.mean(null),3),'p':round(p2,5),
  'highest':[('%d:%s'%k,round(v,2)) for k,v in top[:10]],'lowest':[('%d:%s'%k,round(v,2)) for k,v in top[-8:]],
  'examples_rate_per_1000':{'ق in 50':round(freq[50]['ق']*1000,1),'ق median all':round(st.median(freq[x]['ق'] for x in big)*1000,1),
                            'ص in 38':round(freq[38]['ص']*1000,1),'ص median':round(st.median(freq[x]['ص'] for x in big)*1000,1),
                            'ن in 68':round(freq[68]['ن']*1000,1),'ن median':round(st.median(freq[x]['ن'] for x in big)*1000,1)}}
# ---------- 3. group letter-profile similarity ----------
def vec(s): return [freq[s][a] for a in ALPH]
mu=[st.mean(freq[x][a] for x in big) for a in ALPH]; sd=[st.pstdev(freq[x][a] for x in big) for a in ALPH]
Z={s:[(v-m)/d for v,m,d in zip(vec(s),mu,sd)] for s in freq}
def cosz(a,b):
    na=math.sqrt(sum(x*x for x in Z[a])); nb=math.sqrt(sum(x*x for x in Z[b])); return sum(x*y for x,y in zip(Z[a],Z[b]))/(na*nb)
groups={'حم':[40,41,42,43,44,45,46],'الم':[2,3,29,30,31,32],'الر':[10,11,12,14,15],'طس/طسم':[26,27,28]}
G={}
lens={s:SC[s][1] for s in freq}
for g,ss in groups.items():
    obsg=st.mean(cosz(a,b) for i,a in enumerate(ss) for b in ss[i+1:])
    # null: random surah sets matched on length (nearest-length pool excluding initial surahs)
    pool=[x for x in big if x not in INIT]
    nl=[]
    for _ in range(3000):
        pick=[]
        for s in ss:
            cand=sorted(pool,key=lambda x:abs(math.log(lens[x]/lens[s])))[:12]
            c=random.choice(cand)
            while c in pick: c=random.choice(cand)
            pick.append(c)
        nl.append(st.mean(cosz(a,b) for i,a in enumerate(pick) for b in pick[i+1:]))
    G[g]={'mean_within_cosine':round(obsg,3),'length_matched_null':round(st.mean(nl),3),'p':round((sum(x>=obsg for x in nl)+1)/(len(nl)+1),4)}
R['3_group_profiles']=G
# ---------- 4. 19-claims ----------
def cnt(s,a,mode):
    c=collections.Counter()
    for v in Q[s-1]['verses']: c.update(letters(v['text'],mode))
    return c[a]
claims={}
for mode in ('RASM','PRON'):
    claims[mode]={'ق in 50 (claim 57)':cnt(50,'ق',mode),'ق in 42 (claim 57)':cnt(42,'ق',mode),'ن in 68 (claim 133)':cnt(68,'ن',mode),
                  'ص in 7+19+38 (claim 152)':cnt(7,'ص',mode)+cnt(19,'ص',mode)+cnt(38,'ص',mode)}
R['4_nineteen_claims']={m:{k:[v,'÷19' if v%19==0 else 'not ÷19'] for k,v in d.items()} for m,d in claims.items()}
# ---------- 5. pangram verses ----------
pang={}
for mode in ('RASM','PRON'):
    pang[mode]=['%d:%d'%(s['id'],v['id']) for s in Q for v in s['verses'] if set(ALPH)<=set(letters(v['text'],mode))]
R['5_pangram_verses']=pang
# ---------- 6. al-Fatiha missing letters ----------
fat=set()
for v in Q[0]['verses']: fat|=set(letters(v['text']))
R['6_fatiha_missing']=''.join(a for a in ALPH if a not in fat)
# ---------- 7. middle letter ----------
mid={}
for mode in ('RASM','PRON'):
    seq=[]
    for s in Q:
        for v in s['verses']:
            for w in v['text'].split():
                for a in letters(w,mode): seq.append((a,s['id'],v['id'],w))
    m=len(seq)//2; a,s,v,w=seq[m]
    mid[mode]={'total_letters':len(seq),'middle_letter':a,'verse':f'{s}:{v}','word':w}
R['7_middle']=mid
# ---------- 8. overall letter frequencies ----------
tot=collections.Counter()
for s in Q:
    for v in s['verses']: tot.update(letters(v['text']))
T=sum(tot.values()); H=-sum(c/T*math.log2(c/T) for c in tot.values())
R['8_letter_stats']={'total_letters_RASM':T,'entropy_bits':round(H,3),'top10':[(a,round(c/T*100,2)) for a,c in tot.most_common(10)],'rarest5':[(a,round(c/T*100,3)) for a,c in tot.most_common()[-5:]]}
json.dump(R,open('letters_results.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(R,ensure_ascii=False,indent=1))
