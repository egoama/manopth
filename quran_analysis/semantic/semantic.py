"""Semantic-structure tests:
A divine-name verse endings predicted from content; B repeated-narrative tailoring to host surah;
C character voice differentiation; D emotional arc / surah endings; E divine 'I' vs 'We'.
Usage: python3 -I semantic.py <quran.json> <morph.txt> <scripts_dir>"""
import sys,re,json,math,random,collections,statistics as st
sys.path.insert(0,sys.argv[3]); from load import load_morph,norm
from math import comb
random.seed(37)
Q=json.load(open(sys.argv[1])); W=load_morph(sys.argv[2]); R={}
txt={(s['id'],v['id']):v['text'] for s in Q for v in s['verses']}; keys=sorted(txt)
words=collections.defaultdict(list)   # (s,v)-> list of (w, [segs])
for (s,v,w),segs in W.items(): words[(s,v)].append((w,segs))
def content(segs): return [d for d in segs if d['tag'] in('N','V') and 'LEM' in d and 'PRON' not in d['feat']]
def lemmas(k,skip_last=0):
    ws=sorted(words[k]); ws=ws[:len(ws)-skip_last] if skip_last else ws
    return [norm(d['LEM']) for _,segs in ws for d in content(segs)]
def nb_loo(data,classes,perm=200):
    V=len({l for f,_ in data for l in f})+1
    def run(labels):
        cnt={c:collections.Counter() for c in classes}; tot={c:0 for c in classes}; ny={c:0 for c in classes}
        for (f,_),y in zip(data,labels):
            ny[y]+=1; tot[y]+=len(f)
            for l in f: cnt[y][l]+=1
        corr=0; preds=[]
        for (f,_),y in zip(data,labels):
            best=None
            for c in classes:
                n_=ny[c]-(c==y); t_=tot[c]-(len(f) if c==y else 0)
                if n_<=0: continue
                s_=math.log(n_/(len(data)-1))
                for l in f: s_+=math.log((cnt[c][l]-(c==y and l in f)+0.5)/(t_+0.5*V))
                if best is None or s_>best[0]: best=(s_,c)
            preds.append(best[1]); corr+=best[1]==y
        return corr/len(data),preds
    lab=[y for _,y in data]; obs,preds=run(lab); null=[]
    for _ in range(perm):
        l2=lab[:]; random.shuffle(l2); null.append(run(l2)[0])
    maj=max(collections.Counter(lab).values())/len(lab)
    return {'n':len(data),'classes':len(classes),'loo_acc':round(obs,3),'majority':round(maj,3),'perm_null_mean':round(st.mean(null),3),'p':round((sum(x>=obs for x in null)+1)/(perm+1),4)},preds
# ---------- A. divine-name endings ----------
NAMES={norm(x) for x in ['غَفُور','رَحِيم','عَزِيز','حَكِيم','عَلِيم','سَمِيع','بَصِير','خَبِير','لَطِيف','قَدِير','تَوّاب','غَنِيّ','حَمِيد','حَلِيم','واسِع','شَكُور','قَوِيّ','رَؤُوف','وَدُود','كَرِيم','عَظِيم','عَلِيّ','كَبِير','مَجِيد','قَرِيب','مُجِيب','شَهِيد','حَفِيظ','رَقِيب','وَكِيل','حَسِيب','مُحِيط']}
endings={}
for k in keys:
    ws=sorted(words[k]); tail=[]
    for _,segs in reversed(ws):
        c=content(segs)
        if c and norm(c[0]['LEM']) in NAMES: tail.insert(0,norm(c[0]['LEM']))
        elif c: break
        if len(tail)==2: break
    if len(tail)==2: endings[k]=' '.join(tail)
cnt=collections.Counter(endings.values()); keep=[e for e,c in cnt.items() if c>=12]
dataA=[]; kA=[]
for k,e in endings.items():
    if e in keep:
        f=set(l for l in lemmas(k,skip_last=3) if l not in NAMES and l!='الله')
        if f: dataA.append((f,e)); kA.append(k)
resA,predA=nb_loo(dataA,keep)
# distinctive content per class
cl=collections.defaultdict(collections.Counter)
for (f,e) in dataA:
    for l in f: cl[e][l]+=1
glob=collections.Counter(l for f,_ in dataA for l in f); N=len(dataA)
dist={e:[l for l in sorted(cl[e],key=lambda l:-(cl[e][l]/cnt[e]-glob[l]/N)*math.sqrt(cl[e][l]))[:6]] for e in keep}
ex538=None
if (5,38) in kA: ex538=predA[kA.index((5,38))]
R['A_divine_name_endings']={**resA,'ending_counts':{e:cnt[e] for e in keep},'distinctive_context':dist,'prediction_for_5_38':ex538,'true_5_38':endings.get((5,38))}
# ---------- B. repeated-narrative tailoring ----------
docf=collections.Counter(); VL={}
for k in keys:
    L=lemmas(k,skip_last=1); VL[k]=L
    for l in set(L): docf[l]+=1
idf={l:math.log(len(keys)/c) for l,c in docf.items()}
def vec(Ls):
    c=collections.Counter(l for L in Ls for l in L); return {l:n*idf[l] for l,n in c.items()}
def cos(a,b):
    if len(a)>len(b): a,b=b,a
    num=sum(x*b.get(l,0) for l,x in a.items()); da=math.sqrt(sum(x*x for x in a.values())); db=math.sqrt(sum(x*x for x in b.values()))
    return num/(da*db) if da and db else 0
MOSES={norm('مُوسَى'),norm('فِرْعَوْن')}
mv=collections.defaultdict(list)
for k in keys:
    if set(VL[k])&MOSES: mv[k[0]].append(k)
hosts=[s for s,l in mv.items() if len(l)>=5]
host_ctx={s:vec([VL[k] for k in keys if k[0]==s and k not in set(mv[s])]) for s in hosts}
own=0;tot=0;margins=[]
for s in hosts:
    for k in mv[s]:
        v=vec([[l for l in VL[k] if l not in MOSES]])
        sims={h:cos(v,host_ctx[h]) for h in hosts}
        o=sims[s]; others=[sims[h] for h in hosts if h!=s]
        tot+=1; own+= o>max(others); margins.append(o-st.mean(others))
pos=sum(m>0 for m in margins)
R['B_narrative_tailoring']={'host_surahs':len(hosts),'moses_pharaoh_verses':tot,'own_host_most_similar':round(own/tot,3),'chance_level':round(1/len(hosts),3),
  'positive_margin_share':round(pos/tot,3),'sign_test_p':min(1,2*sum(comb(tot,x) for x in range(pos,tot+1))/2**tot)}
# ---------- C. character voices ----------
PROPH={norm(x) for x in ['مُوسَى','نُوح','إِبْراهِيم','هُود','صالِح','شُعَيْب','عِيسَى','يُوسُف','لُوط','هارُون']}
OPP={norm(x) for x in ['فِرْعَوْن','إِبْلِيس','مَلَأ']}
speech=[]
for k in keys:
    ws=sorted(words[k]); flat=[(w,d) for w,segs in ws for d in segs]
    for i,(w,d) in enumerate(flat):
        if d['tag']=='V' and norm(d.get('LEM',''))=='قال':
            nxt=[x for x in flat[i+1:i+4] if x[1]['tag']=='N' and 'LEM' in x[1]]
            if not nxt: continue
            sp=norm(nxt[0][1]['LEM'])
            grp='P' if sp in PROPH else ('O' if sp in OPP else None)
            if not grp: continue
            after=[x for x in flat if x[0]>nxt[0][0]]
            feats=[]
            for _,dd in after:
                if 'LEM' in dd and dd['tag'] in('N','V','P'): feats.append(norm(dd['LEM']))
                for f in dd['feat']:
                    if f in('INTG','IMPV','VOC','NEG','EMPH','COND','1S','1P','2MS','2MP','PRO'): feats.append('#'+f)
            feats=set(x for x in feats if x not in PROPH|OPP)
            if len(feats)>=3: speech.append((feats,grp,'%d:%d'%k,sp)); break
dataC=[(f,g) for f,g,_,_ in speech]
resC,_=nb_loo(dataC,['P','O'])
gstat={}
for g in 'PO':
    S=[f for f,gg,_,_ in speech if gg==g]
    gstat[g]={t:round(sum(1 for f in S if '#'+t in f)/len(S),3) for t in ('INTG','IMPV','VOC','NEG','EMPH','1S','1P','2MP','PRO')}
    gstat[g]['n']=len(S)
R['C_character_voices']={**resC,'feature_share_by_group':gstat,'speakers':dict(collections.Counter(sp for *_,sp in speech))}
# ---------- D. emotional arc ----------
PUN={norm(x) for x in ['عَذاب','جَهَنَّم','نار','سَعِير','جَحِيم','أَهْلَكَ','عِقاب','غَضَب','لَعْنَة','خِزْي','دَمَّرَ','أَغْرَقَ','حَمِيم','سَقَر','وَيْل','أَلِيم','مُهِين']}
MER={norm(x) for x in ['رَحْمَة','جَنَّة','نَعِيم','غَفُور','رَحِيم','مَغْفِرَة','رِضْوان','فَوْز','سَلام','بُشْرَى','أَجْر','فَضْل','نِعْمَة','رِزْق','حَسَن','كَرِيم','طَيِّب','غَفَرَ']}
def tone(k):
    L=set(lemmas(k)); return (len(L&MER)>0)-(len(L&PUN)>0)
first=[];last=[];mid=[]
for s in Q:
    n=s['total_verses']
    if n<10: continue
    ks=[(s['id'],i) for i in range(1,n+1)]; t=[tone(k) for k in ks]
    first.append(st.mean(t[:3])); last.append(st.mean(t[-3:])); mid.append(st.mean(t[3:-3]))
def perm_pair(a,b,it=5000):
    d=[x-y for x,y in zip(a,b)]; obs=st.mean(d); c=0
    for _ in range(it):
        c+= abs(st.mean([x*random.choice((1,-1)) for x in d]))>=abs(obs)
    return round(obs,3),round((c+1)/(it+1),4)
R['D_emotional_arc']={'surahs':len(first),'mean_tone_first3':round(st.mean(first),3),'mean_tone_middle':round(st.mean(mid),3),'mean_tone_last3':round(st.mean(last),3),
  'last_minus_middle':perm_pair(last,mid),'first_minus_middle':perm_pair(first,mid)}
# ---------- E. divine I vs We ----------
# divine 1S: verses with 1S pronoun attached to عباد/رحمة/عذاب/رب? use explicit divine-speech markers: 'عبادي','رحمتي','عذابي','آياتي','ذكري' and 'إني أنا الله'
D1S_forms=re.compile(r'(عبادي|رحمتي|عذابي|اياتي|ءاياتي|ذكري|رسلي|سبيلي|لقائي|نعمتي)')
D1P_verbs={norm(x) for x in ['أَنزَلَ','نَزَّلَ','خَلَقَ','جَعَلَ','أَرْسَلَ','آتَى','أَوْحَى','أَهْلَكَ','نَجَّى','أَنجَى','كَتَبَ','رَزَقَ','بَعَثَ','أَغْرَقَ','أَمْطَرَ','دَمَّرَ','أَخْرَجَ','أَنبَتَ']}
s1=[];p1=[]
for k in keys:
    t=' '.join(re.sub('[^ء-ي ]',' ',norm(txt[k])).split())
    one=bool(D1S_forms.search(t)); we=any(d['tag']=='V' and '1P' in d['feat'] and norm(d.get('LEM','')) in D1P_verbs for _,segs in words[k] for d in segs)
    if one and not we: s1.append(k)
    elif we and not one: p1.append(k)
def prof(K):
    n=len(K); return {'n':n,'mercy':round(sum(1 for k in K if set(lemmas(k))&MER)/n,3),'punish':round(sum(1 for k in K if set(lemmas(k))&PUN)/n,3),
                      'meccan':round(sum(1 for k in K if Q[k[0]-1]['type']=='meccan')/n,3)}
R['E_divine_I_vs_We']={'I':prof(s1),'We':prof(p1)}
json.dump(R,open('semantic_results.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(R,ensure_ascii=False,indent=1))
