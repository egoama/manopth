"""Robustness: A without length features + disputed-surah stylistic classification; C conjunction-cut baseline; D poetry baseline.
Usage: python3 -I deeper2.py <quran.json> <morph.txt> <HafsLists.js> <apcd.csv> <scripts_dir> <linguistics_dir>"""
import sys,re,json,math,random,collections,statistics as st
sys.path.insert(0,sys.argv[5]); sys.path.insert(0,sys.argv[6])
from load import load_morph,norm
from prosody import encode
random.seed(31)
Q=json.load(open(sys.argv[1])); W=load_morph(sys.argv[2]); R={}
js=open(sys.argv[3],encoding='utf8').read(); i=js.find('SurahList'); seg=js[i:js.find(']\n\t]',i)+4]
rows=re.findall(r'\[\s*(-?\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*"([^"]*)",\s*(true|false)',seg)
rev={k:int(r[2]) for k,r in enumerate(rows) if 1<=k<=114}; meta_mec={k:r[5]=='true' for k,r in enumerate(rows) if 1<=k<=114}
typ={s['id']:s['type'] for s in Q}
def clean(t): return re.sub('[^ء-ي ]',' ',norm(t)).split()
lem=collections.defaultdict(list); person=collections.defaultdict(collections.Counter); tags=collections.defaultdict(collections.Counter)
for (s,v,w),segs in W.items():
    for d in segs:
        if 'LEM' in d: lem[s].append(norm(d['LEM']))
        for x in d['feat']:
            if ':' not in x: tags[s][x]+=1
        if d['tag']=='V':
            for x in d['feat']:
                if re.fullmatch('[123](M|F)?(S|D|P)',x): person[s][x[0]]+=1
F={}
for s in Q:
    sid=s['id']; vs=[v['text'] for v in s['verses']]; enc=[encode(t) for t in vs]
    ends=[clean(t)[-1][-1] for t in vs]; c=collections.Counter(ends); H=-sum(n/len(ends)*math.log2(n/len(ends)) for n in c.values())
    L=lem[sid]; n=len(L); pc=person[sid]; tot=sum(pc.values()) or 1; tg=tags[sid]; nw=sum(1 for k in W if k[0]==sid)
    F[sid]={'median_verse_len':st.median(len(e) for e in enc),'share_mutaharrik':sum(e.count('1') for e in enc)/sum(len(e) for e in enc),
      'rhyme_entropy':H,'alladhi_rate':L.count('الذي')/n,'2nd_person_share':pc['2']/tot,'nun_rhyme_share':ends.count('ن')/len(ends),
      'oath_like':sum(1 for v in s['verses'] if W.get((sid,v['id'],1),[{}])[0].get('LEM')=='و' and any('GEN' in d['feat'] for d in W.get((sid,v['id'],1),[])[1:]))/len(vs),
      'impv_rate':tg['IMPV']/nw,'cond_rate':tg['COND']/nw,'emph_rate':tg['EMPH']/nw}
NOLEN=['share_mutaharrik','rhyme_entropy','alladhi_rate','2nd_person_share','nun_rhyme_share','impv_rate','cond_rate','emph_rate']
ALL=['median_verse_len']+NOLEN+['oath_like']
def zs(keys):
    cols={k:[F[s][k] for s in F] for k in keys}; mu={k:st.mean(v) for k,v in cols.items()}; sd={k:st.pstdev(v) or 1 for k,v in cols.items()}
    return {s:[(F[s][k]-mu[k])/sd[k] for k in keys] for s in F}
def rough(order,Z): return sum(sum((a-b)**2 for a,b in zip(Z[order[i]],Z[order[i+1]])) for i in range(len(order)-1))
trad=sorted(rev,key=lambda s:rev[s]); mush=list(range(1,115))
Z=zs(NOLEN)
def ptest(order,Z,groups=None,it=5000):
    obs=rough(order,Z); c=0
    for _ in range(it):
        if groups:
            o=[]
            for g in groups: g2=g[:]; random.shuffle(g2); o+=g2
        else: o=order[:]; random.shuffle(o)
        c+=rough(o,Z)<=obs
    return round(obs,1),(c+1)/(it+1)
mec=[s for s in trad if typ[s]=='meccan']; med=[s for s in trad if typ[s]=='medinan']
R['A_chronology_no_length']={'features':NOLEN,'traditional':ptest(trad,Z),'traditional_within_groups':ptest(trad,Z,[mec,med]),'mushaf':ptest(mush,Z),
  'mushaf_within_groups':ptest(mush,Z,[[s for s in mush if typ[s]=='meccan'],[s for s in mush if typ[s]=='medinan']])}
# ordered within groups? test traditional order inside Meccan only vs shuffled Meccan
R['A_chronology_no_length']['meccan_only_traditional']=ptest(mec,Z)
# ---------- disputed surahs: nearest-centroid on ALL features, trained on undisputed ----------
DISPUTED=[1,13,22,47,55,57,61,64,76,83,97,98,99,107,110,112,113,114]
Za=zs(ALL)
trainM=[s for s in F if typ[s]=='meccan' and s not in DISPUTED]; trainD=[s for s in F if typ[s]=='medinan' and s not in DISPUTED]
cm=[st.mean(Za[s][i] for s in trainM) for i in range(len(ALL))]; cd=[st.mean(Za[s][i] for s in trainD) for i in range(len(ALL))]
def score(s): 
    dm=math.dist(Za[s],cm); dd=math.dist(Za[s],cd); return dd-dm   # >0 => Meccan-like
# LOO accuracy on undisputed
corr=0; und=trainM+trainD
for s in und:
    M=[x for x in trainM if x!=s]; D=[x for x in trainD if x!=s]
    m_=[st.mean(Za[x][i] for x in M) for i in range(len(ALL))]; d_=[st.mean(Za[x][i] for x in D) for i in range(len(ALL))]
    pred='meccan' if math.dist(Za[s],m_)<math.dist(Za[s],d_) else 'medinan'; corr+=pred==typ[s]
R['disputed_surahs']={'loo_accuracy_undisputed':round(corr/len(und),3),'n_undisputed':len(und),
  'classification':[(s,Q[s-1]['name'],typ[s],'meccan-like' if score(s)>0 else 'medinan-like',round(score(s),2)) for s in DISPUTED]}
# ---------- C. conjunction-cut baseline ----------
WAQF='[ۖ-ۛ]'
bw=[];bc=[];bm=[]
for s in Q:
    for v in s['verses']:
        parts=[p for p in re.split(WAQF,v['text']) if clean(p)]
        if len(parts)!=2: continue
        a,b=encode(parts[0]),encode(parts[1])
        if not a or not b: continue
        bw.append(min(len(a),len(b))/max(len(a),len(b)))
        ws=[w for w in re.sub(WAQF,' ',v['text']).split() if clean(w)]
        cuts=[k for k in range(1,len(ws)) if clean(ws[k])[0][0] in 'وف' and len(clean(ws[k])[0])>2]
        if cuts:
            k=random.choice(cuts); x,y=encode(' '.join(ws[:k])),encode(' '.join(ws[k:]))
            if x and y: bc.append(min(len(x),len(y))/max(len(x),len(y)))
R['C_balance_robust']={'waqf':round(st.mean(bw),3),'conjunction_cut':round(st.mean(bc),3),'n_waqf':len(bw),'n_conj':len(bc)}
# ---------- D. poetry baseline for rhyme pressure ----------
lines=[]
with open(sys.argv[4],encoding='utf8') as f:
    for line in f:
        txt,_,lab=line.rstrip('\n').rpartition(',')
        if lab!='نثر': lines.append(clean(txt))
        if len(lines)>=150000: break
fr=collections.Counter(w for l in lines for w in l)
fin=[0,0];non=[0,0]
for l in lines:
    if len(l)<4: continue
    for j,w in enumerate(l):
        h=fr[w]==1
        if j==len(l)-1: fin[0]+=h; fin[1]+=1
        else: non[0]+=h; non[1]+=1
R['D_poetry_baseline']={'hapax_final':round(fin[0]/fin[1],4),'hapax_other':round(non[0]/non[1],4),'ratio':round((fin[0]/fin[1])/(non[0]/non[1]),2),'note':'word-form hapax over 150k lines'}
# Quran with word-form hapax (same definition as poetry)
qf=collections.Counter(w for s in Q for v in s['verses'] for w in clean(v['text']))
fin=[0,0];non=[0,0]
for s in Q:
    for v in s['verses']:
        l=clean(v['text'])
        if len(l)<4: continue
        for j,w in enumerate(l):
            h=qf[w]==1
            if j==len(l)-1: fin[0]+=h; fin[1]+=1
            else: non[0]+=h; non[1]+=1
R['D_quran_wordform']={'hapax_final':round(fin[0]/fin[1],4),'hapax_other':round(non[0]/non[1],4),'ratio':round((fin[0]/fin[1])/(non[0]/non[1]),2)}
json.dump(R,open('deeper2_results.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(R,ensure_ascii=False,indent=1))
