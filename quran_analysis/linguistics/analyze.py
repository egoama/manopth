"""Comparative prosodic, rhyme, grammatical and lexical analysis: Quran vs Hadith (Bukhari) vs Classical poetry (APCD).
Usage: python3 -I analyze.py <quran.json> <morph.txt> <bukhari_vocalized.csv> <apcd.csv> <scripts_dir>"""
import sys,os,json,csv,re,random,math,collections,statistics as st
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,sys.argv[5])
from prosody import encode,diacritic_coverage
from load import load_morph,norm
random.seed(7)
Q=json.load(open(sys.argv[1])); MORPH=sys.argv[2]
R={}
# ---------- corpora ----------
qverses=[(s['id'],v['id'],s['type'],v['text']) for s in Q for v in s['verses']]
had=[]
for row in csv.reader(open(sys.argv[3],encoding='utf8')):
    t=row[1].replace('‏','').replace('{','').replace('}','')
    mm=re.search('وَسَل[\u0651\u064E]{2}مَ',t)
    if mm:
        m=t[mm.end():].strip()
        if len(m.split())>=8: had.append(m)
poems=[]
with open(sys.argv[4],encoding='utf8') as f:
    for line in f:
        txt,_,lab=line.rstrip('\n').rpartition(',')
        if lab!='نثر' and diacritic_coverage(txt)>=0.85 and len(txt.split())>=6: poems.append((lab,txt))
random.shuffle(poems); poems=poems[:30000]
R['sizes']={'quran_verses':len(qverses),'hadith_matns':len(had),'poetry_lines_vocalized':len(poems)}
# ---------- prosody ----------
qenc=[encode(t,'sakin') for *_,t in qverses]
henc=[]
for m in had:
    w=m.split()
    for i in range(0,len(w)-5,10): henc.append(encode(' '.join(w[i:i+10]),'sakin'))
penc=[encode(t,'ishba') for _,t in poems]
def pooled(seqs,N=250000):
    s=''.join(seqs); return s[:N]
def condH(s,k):
    c=collections.Counter(s[i:i+k+1] for i in range(len(s)-k)); ctx=collections.Counter()
    for g,n in c.items(): ctx[g[:-1]]+=n
    T=sum(c.values()); return -sum(n/T*math.log2(n/ctx[g[:-1]]) for g,n in c.items())
pros={}
for name,seqs in [('quran',qenc),('hadith',henc),('poetry',penc)]:
    s=pooled(seqs)
    run5=len(re.findall('(?=11111)',s))/len(s)*1000
    z2=len(re.findall('(?=00)',s))/len(s)*1000
    pros[name]={'symbols':len(s),'share_mutaharrik':round(s.count('1')/len(s),3),
                'H':[round(condH(s,k),3) for k in (1,2,4,6,8)],
                'runs_of_5plus_harakat_per1000':round(run5,2),'adjacent_sakin_per1000':round(z2,2)}
# periodicity: excess self-similarity at best lag 4..14 vs shuffled
def period_excess(x):
    if len(x)<20: return None
    def best(y): return max(sum(y[i]==y[i+L] for i in range(len(y)-L))/(len(y)-L) for L in range(4,15))
    y=list(x); random.shuffle(y); return best(x)-best(y)
for name,seqs in [('quran',qenc),('hadith',henc),('poetry',penc)]:
    sample=[x for x in seqs if len(x)>=20]; random.shuffle(sample); vals=[period_excess(x) for x in sample[:3000]]
    pros[name]['periodicity_excess']=round(st.mean(vals),4)
# n-gram classifier poetry vs hadith-prose on 24-symbol windows
K=6
def train(seqs):
    c=collections.Counter(); ctx=collections.Counter()
    for x in seqs:
        for i in range(len(x)-K): c[x[i:i+K+1]]+=1; ctx[x[i:i+K]]+=1
    return c,ctx
def ll(model,x):
    c,ctx=model; return sum(math.log((c[x[i:i+K+1]]+1)/(ctx[x[i:i+K]]+2)) for i in range(len(x)-K))
ph=penc[:len(penc)//2]; pt=penc[len(penc)//2:]; random.shuffle(henc); hh=henc[:len(henc)//2]; ht=henc[len(henc)//2:]
MP,MH=train(ph),train(hh)
def windows(seqs,W=24,limit=20000):
    out=[]
    for x in seqs:
        for i in range(0,len(x)-W+1,W): out.append(x[i:i+W])
    random.shuffle(out); return out[:limit]
cls={}
for name,seqs in [('poetry_test',pt),('hadith_test',ht),('quran',qenc)]:
    w=windows(seqs); cls[name]=round(sum(ll(MP,x)>ll(MH,x) for x in w)/len(w),3)
R['prosody']=pros; R['poetic_window_share']=cls
# verse-level: Quran verses classified as poetic (whole verse)
qv_llr=[(ll(MP,x)-ll(MH,x))/max(1,len(x)-K) for x in qenc]
top=sorted(range(len(qverses)),key=lambda i:-qv_llr[i])
R['most_poetic_verses']=[('%d:%d'%qverses[i][:2],round(qv_llr[i],3),qverses[i][3][:80],qenc[i]) for i in top if len(qenc[i])>=20][:12]
# Meccan vs Medinan
for typ in ('meccan','medinan'):
    idx=[i for i,v in enumerate(qverses) if v[2]==typ]
    R.setdefault('meccan_medinan',{})[typ]={'verses':len(idx),'median_symbols':st.median(len(qenc[i]) for i in idx),
        'poetic_llr_mean':round(st.mean(qv_llr[i] for i in idx),4)}
# ---------- rhyme (fasila) ----------
LET=re.compile('[ء-ي]')
def final_key(text):
    t=norm(text); t=re.sub('[^ء-ي]','',t)
    if not t: return None
    # at pause, tanwin-fath alif -> rhyme on preceding consonant + alif
    if t.endswith('ا') and len(t)>1: return t[-2]+'ا'
    return t[-1]
keys=[(s,final_key(t)) for s,v,_,t in qverses]
same=sum(1 for i in range(len(keys)-1) if keys[i][0]==keys[i+1][0] and keys[i][1]==keys[i+1][1])
pairs=sum(1 for i in range(len(keys)-1) if keys[i][0]==keys[i+1][0])
allk=[k for _,k in keys]; random.shuffle(allk)
base=sum(allk[i]==allk[i+1] for i in range(len(allk)-1))/(len(allk)-1)
# within-surah shuffle baseline
ws=[]
for _ in range(20):
    by=collections.defaultdict(list)
    for s,k in keys: by[s].append(k)
    tot=hit=0
    for s,l in by.items():
        l=l[:]; random.shuffle(l); tot+=len(l)-1; hit+=sum(l[i]==l[i+1] for i in range(len(l)-1))
    ws.append(hit/tot)
fin=collections.Counter(k for _,k in keys)
nasal=sum(1 for *_,t in qverses if re.search('(ون|ين|يم|ير|ار|ال|يل|ام)$',re.sub('[^ء-ي]','',norm(t))))
dom=[]
by=collections.defaultdict(list)
for s,k in keys: by[s].append(k)
for s,l in by.items():
    if len(l)>=10: dom.append(collections.Counter(l).most_common(1)[0][1]/len(l))
switches=sum(1 for i in range(len(keys)-1) if keys[i][0]==keys[i+1][0] and keys[i][1]!=keys[i+1][1])
R['rhyme']={'consecutive_same_final':round(same/pairs,3),'random_baseline':round(base,3),'within_surah_shuffle_baseline':round(st.mean(ws),3),
            'top_finals':fin.most_common(8),'share_ending_un_in_im_ir_etc':round(nasal/len(qverses),3),
            'median_dominant_rhyme_share_per_surah':round(st.median(dom),3),'surahs_100pct_single_rhyme':sum(d==1 for d in dom),'surahs_considered':len(dom)}
# ---------- grammar (Quran morphology) ----------
W=load_morph(MORPH)
pos=collections.Counter(); feat=collections.Counter(); ntok=0
person=collections.defaultdict(collections.Counter)
allah_1p=[]
for (s,v,w),segs in W.items():
    ntok+=1
    for d in segs:
        pos[d['tag']]+=1
        for x in d['feat']:
            if ':' not in x: feat[x]+=1
for (s,v,w),segs in W.items():
    for d in segs:
        if d['tag']=='V':
            for x in d['feat']:
                if re.fullmatch('[123](M|F)?(S|D|P)',x): person[(s,v)][x[0]]+=1
for (s,v),c in person.items(): pass
verse_has_allah=set((s,v) for (s,v,w),segs in W.items() for d in segs if norm(d.get('LEM',''))=='الله')
verse_1p=set(k for k,c in person.items() if c['1'])
R['grammar']={'tokens':ntok,'segments_by_tag':dict(pos),
  'per1000words':{k:round(feat[k]/ntok*1000,1) for k in ['PERF','IMPF','IMPV','PASS','EMPH','CERT','VOC','INTG','NEG','COND','PRO','RES','SUP','FUT','1P','1S','2MS','2MP','3MS','3MP']}}
# iltifat proxy: change of dominant verbal person between consecutive verses (same surah) vs within-surah shuffle
dpers={}
for k,c in person.items(): dpers[k]=c.most_common(1)[0][0]
seq=[(s,v,dpers.get((s,v))) for s,v,_,_ in qverses]
seq=[x for x in seq if x[2]]
obs=sum(1 for i in range(len(seq)-1) if seq[i][0]==seq[i+1][0] and seq[i][2]!=seq[i+1][2]); n=sum(1 for i in range(len(seq)-1) if seq[i][0]==seq[i+1][0])
sh=[]
for _ in range(20):
    by=collections.defaultdict(list)
    for s,v,p in seq: by[s].append(p)
    t=h=0
    for l in by.values():
        l=l[:]; random.shuffle(l); t+=len(l)-1; h+=sum(l[i]!=l[i+1] for i in range(len(l)-1))
    sh.append(h/t)
both=sorted(verse_has_allah & verse_1p)
# adjacent-verse shift: verse with divine 1P verb followed/preceded by verse naming Allah with 3rd-person verbs
adj=[]
for i in range(len(qverses)-1):
    a=qverses[i][:2]; b=qverses[i+1][:2]
    if a[0]!=b[0]: continue
    if (a in verse_1p and b in verse_has_allah and person[b]['3'] and not person[b]['1']) or (b in verse_1p and a in verse_has_allah and person[a]['3'] and not person[a]['1']): adj.append('%d:%d-%d'%(a[0],a[1],b[1]))
R['iltifat']={'adjacent_verse_we_he_shifts':len(adj),'adjacent_examples':adj[:10],'person_switch_rate_consecutive':round(obs/n,3),'within_surah_shuffle':round(st.mean(sh),3),
              'verses_with_Allah_and_1st_person_plural_or_singular_verb':len(both),'examples':['%d:%d'%k for k in both[:15]]}
# oath formula: verse-initial و + genitive noun (Meccan)
oath=[]
for s,v,_,t in qverses:
    segs=W.get((s,v,1),[])
    if len(segs)>=2 and segs[0].get('LEM')=='و' and any('GEN' in d['feat'] for d in segs[1:]) : oath.append('%d:%d'%(s,v))
R['oath_like_openings']=len(oath)
# ---------- lexical & function words (all corpora, normalized, equal-size samples) ----------
def toks(texts):
    out=[]
    for t in texts: out+= [w for w in re.sub('[^ء-ي ]',' ',norm(t)).split() if w]
    return out
qt=toks(t for *_,t in qverses); ht_=toks(had); pt_=toks(t for _,t in poems)
N=min(len(qt),len(ht_),len(pt_),60000)
def lex(ts):
    ts=ts[:N]; c=collections.Counter(ts)
    return {'tokens':N,'types':len(c),'hapax_share_of_types':round(sum(1 for v in c.values() if v==1)/len(c),3),'mean_word_len':round(st.mean(len(w) for w in ts),2)}
FW={'إن/أن(ان)':'ان','الذين':'الذين','قد':'قد','لقد':'لقد','يا':'يا','لا':'لا','ما':'ما','لم':'لم','لن':'لن','هل':'هل','اذا':'اذا','لو':'لو','كلا':'كلا','ثم':'ثم','بل':'بل','قال':'قال','الله':'الله','ربك/ربكم':'رب'}
def fw(ts):
    ts=ts[:N]; c=collections.Counter(ts); out={}
    for lab,w in FW.items():
        if w=='رب': n=sum(v for k,v in c.items() if k in('ربك','ربكم','ربنا','ربه','ربهم','ربي'))
        else: n=c[w]+ (c['و'+w] if w not in('ان',) else c['وان']+c['فان']+c['بان']+c['لان'])
        out[lab]=round(n/N*1000,1)
    return out
R['lexical']={'quran':lex(qt),'hadith':lex(ht_),'poetry':lex(pt_)}
R['function_words_per1000']={'quran':fw(qt),'hadith':fw(ht_),'poetry':fw(pt_)}
R['function_words_per1000']['quran']['يا']=R['grammar']['per1000words']['VOC']  # Uthmani joins ya- to noun; use morphology VOC tag
json.dump(R,open('results.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(R,ensure_ascii=False,indent=1)[:9000])
