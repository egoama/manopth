"""Five deeper tests with controls / null models.
A orthography-controlled prosody; B sound-meaning congruence; C rhyme switch vs topic boundary;
D near-synonym context differentiation; E ring composition & opening-closing correspondence.
Usage: python3 -I deep.py <quran.json> <morph.txt> <hadith_dir> <scripts_dir> <linguistics_dir>"""
import sys,os,json,csv,re,glob,random,math,collections,statistics as st
sys.path.insert(0,sys.argv[4]); sys.path.insert(0,sys.argv[5])
from load import load_morph,norm
from prosody import encode
random.seed(11)
Q=json.load(open(sys.argv[1])); W=load_morph(sys.argv[2]); R={}
qverses=[(s['id'],v['id'],s['type'],v['text']) for s in Q for v in s['verses']]
# ---------- corpora from 9 hadith books ----------
quotes=[]; prose=[]
for f in glob.glob(os.path.join(sys.argv[3],'*','*mushakkala*.csv')):
    for r in csv.reader(open(f,encoding='utf8')):
        t=r[1].replace('‏',' ')
        for x in re.findall(r'\{([^{}]+)\}',t):
            for part in re.split('إِلَى قَوْلِهِ|إِلَى آخِرِ الْآيَةِ|الْآيَةَ',x):
                if len(part.split())>=4: quotes.append(part)
        m=re.search('وَسَل[َّ]{2}مَ',t)
        if m:
            body=re.sub(r'\{[^{}]*\}',' ',t[m.end():])
            w=body.split()
            for i in range(0,len(w)-7,10): prose.append(' '.join(w[i:i+10]))
random.shuffle(prose)
# ---------- A. orthography-controlled prosody ----------
def prosody_stats(texts,pause='sakin',N=200000):
    s=''.join(encode(t,pause) for t in texts)[:N]
    return {'symbols':len(s),'share_mutaharrik':round(s.count('1')/len(s),3),
            'runs5_per1000':round(len(re.findall('(?=11111)',s))/len(s)*1000,2)}
A={'quran_uthmani':prosody_stats([t for *_,t in qverses]),
   'quran_quoted_in_hadith_orthography':prosody_stats(quotes),
   'hadith_prose_same_orthography':prosody_stats(prose[:40000])}
# bootstrap CI for runs5 difference quotes vs prose (by unit)
def per_unit(texts):
    out=[]
    for t in texts:
        e=encode(t); 
        if len(e)>=10: out.append((len(re.findall('(?=11111)',e)),len(e)))
    return out
uq=per_unit(quotes); up=per_unit(prose[:20000])
def rate(u): return sum(a for a,_ in u)/sum(b for _,b in u)*1000
boots=[]
for _ in range(300):
    bq=[random.choice(uq) for _ in uq]; bp=[random.choice(up) for _ in up]; boots.append(rate(bp)/max(rate(bq),1e-9))
boots.sort(); A['ratio_prose_over_quran_runs5']={'point':round(rate(up)/rate(uq),2),'ci95':[round(boots[7],2),round(boots[292],2)],'n_quote_units':len(uq)}
R['A_orthography_control']=A
# ---------- verse lemma bags ----------
lem=collections.defaultdict(list); vpos={}
for (s,v,w),segs in W.items():
    for d in segs:
        if d['tag'] in('N','V') and 'LEM' in d and 'PRON' not in d['feat']: lem[(s,v)].append(norm(d['LEM']))
df=collections.Counter(l for k in lem for l in set(lem[k])); NV=len(lem)
idf={l:math.log(NV/c) for l,c in df.items()}
def vec(k):
    c=collections.Counter(lem.get(k,[])); return {l:n*idf[l] for l,n in c.items()}
VEC={k:vec(k) for k in lem}
NORM={k:math.sqrt(sum(x*x for x in v.values())) or 1 for k,v in VEC.items()}
def cos(a,b):
    va,vb=VEC.get(a,{}),VEC.get(b,{})
    if len(va)>len(vb): va,vb=vb,va
    return sum(x*vb.get(l,0) for l,x in va.items())/(NORM.get(a,1)*NORM.get(b,1))
# ---------- B. sound-meaning congruence ----------
PUN={norm(x) for x in ['عَذاب','جَهَنَّم','نار','سَعِير','جَحِيم','أَهْلَكَ','عِقاب','غَضَب','لَعْنَة','خِزْي','دَمَّرَ','أَغْرَقَ','صاعِقَة','حَمِيم','سَقَر','ويل','شَدِيد','أَلِيم','مُهِين']}
MER={norm(x) for x in ['رَحْمَة','جَنَّة','نَعِيم','غَفُور','رَحِيم','مَغْفِرَة','رِضْوان','فَوْز','سَلام','بُشْرَى','أَجْر','فَضْل','نِعْمَة','رِزْق','أَنْهار','حَسَن','كَرِيم','طَيِّب']}
QALQ=set('قطبجد'); EMPH=set('صضطظق'); SON=set('منلرويا')
def phon(text,exclude):
    w=[x for x in re.sub('[^ء-ي ]',' ',norm(text)).split() if x not in exclude]
    s=''.join(w); n=len(s)
    if n<10: return None
    return (sum(c in QALQ|EMPH for c in s)/n, sum(c in SON for c in s)/n)
# exclude surface forms of tag words: use normalized word forms containing tag lemma stem roughly -> exclude any word sharing tag lemma string
def tag_verse(k):
    ls=set(lem.get(k,[])); p=bool(ls&PUN); m=bool(ls&MER); return 'P' if p and not m else ('M' if m and not p else None)
txt={(s,v):t for s,v,_,t in qverses}
def excl_words(text,tagset):
    return {w for w in re.sub('[^ء-ي ]',' ',norm(text)).split() if any(t in w for t in tagset if len(t)>=3)}
grp={'P':[],'M':[]}
for k in lem:
    g=tag_verse(k)
    if g:
        ph=phon(txt[k],excl_words(txt[k],PUN|MER))
        if ph: grp[g].append(ph)
def perm_diff(a,b,it=3000):
    obs=st.mean(a)-st.mean(b); pool=a+b; n=len(a); c=0
    for _ in range(it):
        random.shuffle(pool); c+= abs(st.mean(pool[:n])-st.mean(pool[n:]))>=abs(obs)
    return obs,(c+1)/(it+1)
hq=[x[0] for x in grp['P']]; mq=[x[0] for x in grp['M']]
d,p=perm_diff(hq,mq)
B={'quran':{'n_punish':len(hq),'n_mercy':len(mq),'harsh_share_punish':round(st.mean(hq),4),'harsh_share_mercy':round(st.mean(mq),4),'diff':round(d,4),'p':round(p,4),
   'sonorant_punish':round(st.mean(x[1] for x in grp['P']),4),'sonorant_mercy':round(st.mean(x[1] for x in grp['M']),4)}}
# hadith control: tag windows by same lexicons (normalized surface word contains lemma)
PUNs={norm(x) for x in ['عذاب','جهنم','النار','السعير','الجحيم','اهلك','عقاب','غضب','لعنه','خزي','حميم','ويل','شديد','اليم']}
MERs={norm(x) for x in ['رحمه','الجنه','نعيم','غفور','رحيم','مغفره','رضوان','الفوز','سلام','بشري','اجر','فضل','نعمه','رزق','الانهار','كريم','طيب']}
hp=[];hm=[]
for t in prose[:60000]:
    ws=set(re.sub('[^ء-ي ]',' ',norm(t)).split())
    p_=any(any(x in w for x in PUNs) for w in ws); m_=any(any(x in w for x in MERs) for w in ws)
    if p_!=m_:
        ph=phon(t,excl_words(t,PUNs|MERs))
        if ph: (hp if p_ else hm).append(ph[0])
d2,p2=perm_diff(hp,hm)
B['hadith_control']={'n_punish':len(hp),'n_mercy':len(hm),'harsh_share_punish':round(st.mean(hp),4),'harsh_share_mercy':round(st.mean(hm),4),'diff':round(d2,4),'p':round(p2,4)}
# effect sizes (Cohen d)
def cohen(a,b): return (st.mean(a)-st.mean(b))/math.sqrt((st.pvariance(a)+st.pvariance(b))/2)
B['quran']['cohen_d']=round(cohen(hq,mq),3); B['hadith_control']['cohen_d']=round(cohen(hp,hm),3)
R['B_sound_meaning']=B
# ---------- C. rhyme switch vs topic boundary ----------
def fkey(t):
    t=re.sub('[^ء-ي]','',norm(t)); return (t[-2]+'ا') if t.endswith('ا') and len(t)>1 else (t[-1] if t else '')
rows=[]
for i in range(len(qverses)-1):
    a,b=qverses[i],qverses[i+1]
    if a[0]!=b[0]: continue
    ka,kb=(a[0],a[1]),(b[0],b[1])
    rows.append((a[0],fkey(a[3])!=fkey(b[3]),cos(ka,kb),len(lem.get(ka,[]))+len(lem.get(kb,[]))))
sw=[c for _,s,c,_ in rows if s]; ns=[c for _,s,c,_ in rows if not s]
# length-stratified comparison
strata=collections.defaultdict(lambda:{'s':[],'n':[]})
for _,s,c,L in rows: strata[min(L//10,8)]['s' if s else 'n'].append(c)
wdiff=[];wts=[]
for g in strata.values():
    if len(g['s'])>=20 and len(g['n'])>=20: wdiff.append(st.mean(g['n'])-st.mean(g['s'])); wts.append(len(g['s']))
d3,p3=perm_diff(ns,sw,2000)
R['C_rhyme_vs_topic']={'cohesion_same_rhyme':round(st.mean(ns),4),'cohesion_at_rhyme_switch':round(st.mean(sw),4),'n_switch':len(sw),'n_same':len(ns),
  'p':round(p3,4),'length_stratified_mean_drop':round(sum(a*b for a,b in zip(wdiff,wts))/sum(wts),4)}
# ---------- D. near-synonyms ----------
PAIRS=[('عام','سنه','N'),('غيث','مطر','N'),('خوف','خشي','R'),('قلب','فواد','N'),('انسان','بشر','N'),('جاء','اتي','V'),('نظر','راي','R'),('حلف','قسم','R'),('بعث','رسل','R'),('ضياء','نور','N'),('عباد','عبيد','N'),('مرض','سقم','R')]
roots=collections.defaultdict(list)
for (s,v,w),segs in W.items():
    for d in segs:
        if 'ROOT' in d: roots[(s,v)].append(d['ROOT'])
def verses_with(x,mode):
    out=set()
    for k in lem:
        if mode=='N' and x in lem[k]: out.add(k)
        if mode=='V' and x in lem[k]: out.add(k)
        if mode=='R' and any(norm(r)==x for r in roots.get(k,[])): out.add(k)
    return out
def nb_loo(A_,B_):
    data=[(k,0) for k in A_-B_]+[(k,1) for k in B_-A_]
    if min(sum(1 for _,y in data if y==0),sum(1 for _,y in data if y==1))<4: return None
    def acc(labels):
        cnt=[collections.Counter(),collections.Counter()]; tot=[0,0]; ny=[0,0]
        for (k,_),y in zip(data,labels):
            ny[y]+=1; 
            for l in set(lem[k]): cnt[y][l]+=1; tot[y]+=1
        V=len(df); correct=0
        for (k,_),y in zip(data,labels):
            sc=[]
            for c in (0,1):
                n_=ny[c]-(c==y); t_=tot[c]-(len(set(lem[k])) if c==y else 0)
                s_=math.log((n_+1)/(len(data)+1))
                for l in set(lem[k]):
                    s_+=math.log((cnt[c][l]-(c==y)+0.5)/(t_+0.5*V))
                sc.append(s_)
            correct+= (sc[1]>sc[0])==(y==1)
        return correct/len(data)
    labels=[y for _,y in data]; obs=acc(labels)
    null=[]
    for _ in range(200):
        l2=labels[:]; random.shuffle(l2); null.append(acc(l2))
    p=(sum(x>=obs for x in null)+1)/201
    return obs,st.mean(null),p,len(data)
def G2(a,b,ab,N):
    k=[ab,a-ab,b-ab,N-a-b+ab]; r=[k[0]+k[1],k[2]+k[3]]; c=[k[0]+k[2],k[1]+k[3]]
    e=[r[0]*c[0]/N,r[0]*c[1]/N,r[1]*c[0]/N,r[1]*c[1]/N]; return 2*sum(x*math.log(x/y) for x,y in zip(k,e) if x>0)
D=[]
for a,b,mode in PAIRS:
    A_,B_=verses_with(a,mode),verses_with(b,mode)
    res=nb_loo(A_,B_)
    pun_a=sum(1 for k in A_ if set(lem[k])&PUN)/max(1,len(A_)); pun_b=sum(1 for k in B_ if set(lem[k])&PUN)/max(1,len(B_))
    mer_a=sum(1 for k in A_ if set(lem[k])&MER)/max(1,len(A_)); mer_b=sum(1 for k in B_ if set(lem[k])&MER)/max(1,len(B_))
    # distinctive context lemmas
    ca=collections.Counter(l for k in A_-B_ for l in set(lem[k]) if l not in(a,b)); cb=collections.Counter(l for k in B_-A_ for l in set(lem[k]) if l not in(a,b))
    na,nb=len(A_-B_),len(B_-A_)
    top_a=sorted([(l,n) for l,n in ca.items() if n>=2 and n/na>cb[l]/max(nb,1)],key=lambda x:-G2(na,ca[x[0]]+cb[x[0]],x[1],na+nb))[:5]
    top_b=sorted([(l,n) for l,n in cb.items() if n>=2 and n/max(nb,1)>ca[l]/max(na,1)],key=lambda x:-G2(nb,ca[x[0]]+cb[x[0]],x[1],na+nb))[:5]
    D.append({'pair':f'{a}/{b}','n':[len(A_),len(B_)],'loo_acc':None if not res else round(res[0],3),'null_acc':None if not res else round(res[1],3),'p':None if not res else round(res[2],4),
              'punish_share':[round(pun_a,2),round(pun_b,2)],'mercy_share':[round(mer_a,2),round(mer_b,2)],'ctx_a':[x for x,_ in top_a],'ctx_b':[x for x,_ in top_b],
              'meccan_share':[round(sum(1 for k in A_ if Q[k[0]-1]['type']=='meccan')/max(1,len(A_)),2),round(sum(1 for k in B_ if Q[k[0]-1]['type']=='meccan')/max(1,len(B_)),2)]})
R['D_near_synonyms']=D
# ---------- E. ring composition ----------
E=[];beg_end=[]
for s in Q:
    n=s['total_verses']
    if n<15: continue
    ks=[(s['id'],i) for i in range(1,n+1)]
    bydist=collections.defaultdict(list)
    for i in range(n):
        for j in range(i+1,n): bydist[j-i].append(cos(ks[i],ks[j]))
    mir=[];base=[]
    for i in range(n//2):
        j=n-1-i
        if j-i<2: continue
        mir.append(cos(ks[i],ks[j])); base.append(st.mean(bydist[j-i]))
    E.append((s['id'],st.mean(mir)-st.mean(base)))
    # opening (first 3) vs closing (last 3) block similarity vs all 3-blocks at same distance
    def blk(a): return [ks[a+t] for t in range(3)]
    def bsim(a,b): return st.mean(cos(x,y) for x in blk(a) for y in blk(b))
    dist=n-3
    obs=bsim(0,n-3)
    null=[bsim(a,a+dist) for a in range(0,n-3-dist+1)] if n-3-dist>=0 else []
    # null over random block pairs separated by >= n/2
    nulls=[bsim(a,b) for a in range(0,n-2) for b in range(a+max(3,n//2),n-2)]
    beg_end.append((s['id'],obs,st.mean(nulls) if nulls else None,(sum(x>=obs for x in nulls)+1)/(len(nulls)+1) if nulls else None))
pos=sum(1 for _,d in E if d>0)
from math import comb
def sign_p(k,n): return min(1,2*sum(comb(n,i) for i in range(k,n+1))/2**n) if k>n/2 else 1
be=[x for x in beg_end if x[2] is not None]
R['E_ring']={'surahs':len(E),'mirror_above_distance_baseline':pos,'sign_test_p':round(sign_p(pos,len(E)),5),
  'top_mirror_surahs':sorted([(s,round(d,4)) for s,d in E],key=lambda x:-x[1])[:8],
  'opening_closing':{'surahs':len(be),'obs_above_null_mean':sum(1 for x in be if x[1]>x[2]),'sign_p':round(sign_p(sum(1 for x in be if x[1]>x[2]),len(be)),6),
     'significant_p_lt_0.05':[(s,round(o,3),round(p,3)) for s,o,m,p in be if p<0.05]}}
json.dump(R,open('deep_results.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(R,ensure_ascii=False,indent=1))
