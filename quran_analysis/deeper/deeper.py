"""A chronology smoothness test; B euphony (tanafur) at word boundaries; C internal balance via waqf marks; D rhyme pressure on rare words.
Usage: python3 -I deeper.py <quran.json> <morph.txt> <HafsLists.js> <hadith_dir> <apcd.csv> <scripts_dir> <linguistics_dir>"""
import sys,os,re,csv,glob,json,math,random,collections,statistics as st
sys.path.insert(0,sys.argv[6]); sys.path.insert(0,sys.argv[7])
from load import load_morph,norm
from prosody import encode,diacritic_coverage
random.seed(29)
Q=json.load(open(sys.argv[1])); W=load_morph(sys.argv[2]); R={}
# revelation order (Egyptian standard) from quran-meta Hafs SurahList: [start,count,revOrder,rukus,name,isMeccan]
js=open(sys.argv[3],encoding='utf8').read(); i=js.find('SurahList'); seg=js[i:js.find(']\n\t]',i)+4]
rows=re.findall(r'\[\s*(-?\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*"([^"]*)",\s*(true|false)',seg)
rev={k:int(r[2]) for k,r in enumerate(rows) if 1<=k<=114}
assert len(rev)==114 and sorted(rev.values())==list(range(1,115)), len(rev)
typ={s['id']:s['type'] for s in Q}
def clean(t): return re.sub('[^ء-ي ]',' ',norm(t)).split()
# ---------- A. chronology ----------
feat={}
lem=collections.defaultdict(list); person=collections.defaultdict(collections.Counter)
for (s,v,w),segs in W.items():
    for d in segs:
        if 'LEM' in d: lem[s].append(norm(d['LEM']))
        if d['tag']=='V':
            for x in d['feat']:
                if re.fullmatch('[123](M|F)?(S|D|P)',x): person[s][x[0]]+=1
for s in Q:
    vs=[v['text'] for v in s['verses']]; enc=[encode(t) for t in vs]
    ends=[]
    for t in vs:
        w=clean(t)[-1]; ends.append(w[-1])
    c=collections.Counter(ends); H=-sum(n/len(ends)*math.log2(n/len(ends)) for n in c.values())
    L=lem[s['id']]; n=len(L)
    pc=person[s['id']]; tot=sum(pc.values()) or 1
    feat[s['id']]=[st.median(len(e) for e in enc), sum(e.count('1') for e in enc)/sum(len(e) for e in enc), H,
                   L.count('الذي')/n*1000, pc['2']/tot, sum(1 for e in ends if e=='ن')/len(ends), len(set(L))/math.sqrt(n)]
names=['median_verse_len','share_mutaharrik','rhyme_entropy','alladhi_rate','2nd_person_share','nun_rhyme_share','lexical_richness']
cols=list(zip(*feat.values())); mu=[st.mean(c) for c in cols]; sd=[st.pstdev(c) for c in cols]
Z={s:[(x-m)/d for x,m,d in zip(f,mu,sd)] for s,f in feat.items()}
def rough(order): return sum(sum((a-b)**2 for a,b in zip(Z[order[i]],Z[order[i+1]])) for i in range(len(order)-1))
trad=sorted(rev,key=lambda s:rev[s]); mush=list(range(1,115))
def perm_test(order,groups=None,it=5000):
    obs=rough(order); c=0
    for _ in range(it):
        if groups:
            o=[]; 
            for g in groups: g2=g[:]; random.shuffle(g2); o+=g2
        else: o=order[:]; random.shuffle(o)
        c+= rough(o)<=obs
    return obs,(c+1)/(it+1)
mec=[s for s in trad if typ[s]=='meccan']; med=[s for s in trad if typ[s]=='medinan']
o1,p1=perm_test(trad); o2,p2=perm_test(trad,[mec,med]); o3,p3=perm_test(mush)
# outliers: surahs whose features are far from their chronological neighbours
dev={}
for k,s in enumerate(trad):
    nb=[trad[j] for j in range(max(0,k-3),min(114,k+4)) if j!=k]
    cen=[st.mean(Z[x][i] for x in nb) for i in range(len(names))]
    dev[s]=math.sqrt(sum((a-b)**2 for a,b in zip(Z[s],cen)))
# Spearman of each feature with revelation order
def spearman(x,y):
    rx={v:i for i,v in enumerate(sorted(range(len(x)),key=lambda i:x[i]))}; ry={v:i for i,v in enumerate(sorted(range(len(y)),key=lambda i:y[i]))}
    n=len(x); return 1-6*sum((rx[i]-ry[i])**2 for i in range(n))/(n*(n*n-1))
ro=[rev[s] for s in range(1,115)]
R['A_chronology']={'roughness_traditional':round(o1,1),'p_vs_random':p1,'p_within_meccan_medinan_shuffle':p2,'roughness_mushaf':round(o3,1),'p_mushaf_vs_random':p3,
  'spearman_with_revelation_order':{n_:round(spearman([feat[s][i] for s in range(1,115)],ro),3) for i,n_ in enumerate(names)},
  'outliers_top8':[(s,Q[s-1]['name'],rev[s],round(dev[s],2)) for s in sorted(dev,key=lambda s:-dev[s])[:8]]}
# ---------- B. euphony at word boundaries ----------
PLACE={}
for ch in 'بمف': PLACE[ch]='labial'
for ch in 'ثذظ': PLACE[ch]='interdental'
for ch in 'تدطنلرسزصض': PLACE[ch]='alveolar'
for ch in 'جشي': PLACE[ch]='palatal'
for ch in 'ك': PLACE[ch]='velar'
for ch in 'قخغ': PLACE[ch]='uvular'
for ch in 'حع': PLACE[ch]='pharyngeal'
for ch in 'ءاه': PLACE[ch]='glottal'
for ch in 'و': PLACE[ch]='labial'
def cons(w):
    w=w.replace('ة','ت')
    return [c for c in w if c in PLACE and c!='ا']
def boundary_pairs(texts):
    pairs=[]
    for t in texts:
        ws=[cons(w) for w in clean(t)]
        ws=[w for w in ws if w]
        for a,b in zip(ws,ws[1:]):
            first=b[0]
            if len(b)>1 and b[0]=='ل' and clean_al: pass
            pairs.append((a[-1],first))
    return pairs
clean_al=False
def oe(pairs):
    n=len(pairs); fa=collections.Counter(a for a,_ in pairs); fb=collections.Counter(b for _,b in pairs)
    pa={k:v/n for k,v in fa.items()}; pb={k:v/n for k,v in fb.items()}
    obs_same=sum(1 for a,b in pairs if PLACE[a]==PLACE[b])/n
    exp_same=sum(pa[a]*pb[b] for a in pa for b in pb if PLACE[a]==PLACE[b])
    obs_ident=sum(1 for a,b in pairs if a==b)/n; exp_ident=sum(pa[a]*pb.get(a,0) for a in pa)
    gut={'pharyngeal','uvular','glottal'}
    obs_gut=sum(1 for a,b in pairs if PLACE[a] in gut and PLACE[b] in gut and PLACE[a]==PLACE[b])/n
    exp_gut=sum(pa[a]*pb[b] for a in pa for b in pb if PLACE[a] in gut and PLACE[a]==PLACE[b])
    return {'pairs':n,'same_place_O/E':round(obs_same/exp_same,3),'identical_O/E':round(obs_ident/exp_ident,3),'guttural_same_place_O/E':round(obs_gut/exp_gut,3) if exp_gut else None}
quotes=[];prose=[]
for f in glob.glob(os.path.join(sys.argv[4],'*','*mushakkala*.csv')):
    for r in csv.reader(open(f,encoding='utf8')):
        t=r[1].replace('‏',' ')
        for x in re.findall(r'\{([^{}]+)\}',t):
            if len(x.split())>=4: quotes.append(x)
        m=re.search('وَسَل[َّ]{2}مَ',t)
        if m: prose.append(re.sub(r'\{[^{}]*\}',' ',t[m.end():]))
poems=[]
with open(sys.argv[5],encoding='utf8') as f:
    for line in f:
        txt,_,lab=line.rstrip('\n').rpartition(',')
        if lab!='نثر': poems.append(txt)
        if len(poems)>=80000: break
qtexts=[v['text'] for s in Q for v in s['verses']]
def bootstrap_ratio(texts,it=200):
    vals=[]
    for _ in range(it):
        smp=[random.choice(texts) for _ in range(len(texts))]
        vals.append(oe(boundary_pairs(smp))['same_place_O/E'])
    vals.sort(); return [vals[int(0.025*it)],vals[int(0.975*it)]]
random.shuffle(prose); random.shuffle(poems)
B={'quran_uthmani':oe(boundary_pairs(qtexts)),'quran_quotes_hadith_orthography':oe(boundary_pairs(quotes)),
   'hadith_prose':oe(boundary_pairs(prose[:8000])),'poetry':oe(boundary_pairs(poems[:30000]))}
B['ci95_same_place']={'quran_quotes':bootstrap_ratio(quotes[:3000]),'hadith_prose':bootstrap_ratio(prose[:3000]),'poetry':bootstrap_ratio(poems[:8000])}
# within-word baseline (root constraint, OCP) for reference
def within(texts):
    pairs=[]
    for t in texts:
        for w in clean(t):
            c=cons(w); pairs+=list(zip(c,c[1:]))
    return oe(pairs)['same_place_O/E']
B['within_word_same_place_O/E']={'quran':within(qtexts),'hadith':within(prose[:8000]),'poetry':within(poems[:30000])}
R['B_euphony']=B
# ---------- C. internal balance by waqf marks ----------
WAQF='[ۖ-ۛ]'
bal=[];rnd=[]
for s in Q:
    for v in s['verses']:
        parts=[p.strip() for p in re.split(WAQF,v['text']) if p.strip() and clean(p)]
        if len(parts)!=2: continue
        a,b=encode(parts[0]),encode(parts[1])
        if not a or not b: continue
        bal.append(min(len(a),len(b))/max(len(a),len(b)))
        ws=v['text'].replace('ۖ','').replace('ۗ','').replace('ۚ','').replace('ۛ','').split()
        ws=[w for w in ws if clean(w)]
        if len(ws)<4: continue
        k=random.randint(1,len(ws)-1); x,y=encode(' '.join(ws[:k])),encode(' '.join(ws[k:]))
        if x and y: rnd.append(min(len(x),len(y))/max(len(x),len(y)))
R['C_internal_balance']={'two_segment_verses':len(bal),'mean_balance_waqf':round(st.mean(bal),3),'mean_balance_random_cut':round(st.mean(rnd),3),
  'share_balance_ge_0.8_waqf':round(sum(x>=0.8 for x in bal)/len(bal),3),'share_ge_0.8_random':round(sum(x>=0.8 for x in rnd)/len(rnd),3)}
# ---------- D. rhyme pressure on rare words ----------
freq=collections.Counter(); pos=[]
verse_words=collections.defaultdict(list)
for (s,v,w),segs in W.items():
    content=[d for d in segs if d['tag'] in('N','V') and 'LEM' in d and 'PRON' not in d['feat']]
    if content: verse_words[(s,v)].append((w,content[0]['LEM'],content[0]['tag']))
for k,ws in verse_words.items():
    for w,l,t in ws: freq[l]+=1
fin=[0,0]; non=[0,0]; finN=[0,0]; nonN=[0,0]
for k,ws in verse_words.items():
    last=max(w for w,_,_ in ws)
    for w,l,t in ws:
        h=freq[l]==1
        if w==last: fin[0]+=h; fin[1]+=1
        else: non[0]+=h; non[1]+=1
        if t=='N':
            if w==last: finN[0]+=h; finN[1]+=1
            else: nonN[0]+=h; nonN[1]+=1
# do final hapaxes fit local rhyme?
def fk(t):
    w=clean(t)[-1]; return w[-1]
txt={(s['id'],v['id']):v['text'] for s in Q for v in s['verses']}
keys=sorted(txt); fit=tot=0
for i,k in enumerate(keys):
    ws=verse_words.get(k)
    if not ws: continue
    last=max(w for w,_,_ in ws); lw=[l for w,l,_ in ws if w==last][0]
    if freq[lw]!=1: continue
    nb=[keys[j] for j in range(max(0,i-3),min(len(keys),i+4)) if j!=i and keys[j][0]==k[0]]
    if not nb: continue
    tot+=1; fit+= fk(txt[k]) in {fk(txt[x]) for x in nb}
R['D_rhyme_pressure']={'hapax_rate_final_word':round(fin[0]/fin[1],4),'hapax_rate_other_positions':round(non[0]/non[1],4),'ratio':round((fin[0]/fin[1])/(non[0]/non[1]),2),
  'nouns_only_ratio':round((finN[0]/finN[1])/(nonN[0]/nonN[1]),2),'final_hapax_verses':tot,'final_hapax_matching_local_rhyme':round(fit/tot,3) if tot else None}
json.dump(R,open('deeper_results.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(R,ensure_ascii=False,indent=1))
