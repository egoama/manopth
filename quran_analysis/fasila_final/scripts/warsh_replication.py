"""Phase F: replication of the primary analysis on the Warsh text (quran-ws/quran-text, KFGQPC UthmanicWarsh v3.0, Madani-last numbering).
Aligns the 30 eligible Hafs pairs via data/ayah-map.csv; recomputes D1/D3 endings and k=3 neighbourhoods in Warsh numbering;
also re-extracts candidate pairs independently on the Warsh text."""
import sys,os,json,csv,collections,difflib,unicodedata,re
H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(H,'..','..','revision','audit'))
from common import *
from math import comb
QW=os.environ.get('QWS',os.path.join(D,'ext','qwstext'))
w=json.load(open(os.path.join(QW,'data/mushaf/warsh.json'),encoding='utf8'))
words=[unicodedata.normalize('NFC',x) for x in w['words']]
print('keys sample:',{k:(type(w[k]).__name__, (w[k][:3] if isinstance(w[k],list) else '')) for k in ('ayah_starts','surah_starts','numbering')})
ast=w['ayah_starts']; sst=w['surah_starts']
# build verse list: ayah_starts is a list of word indices (one per ayah, in order); numbering gives (surah,ayah) per index?
num=w['numbering']
if isinstance(num,dict): print('numbering keys',list(num.keys())[:5])
verses=[]  # (surah, ayah, text)
if isinstance(ast,list) and isinstance(ast[0],int):
    # ayah_starts and surah_starts are WORD indices; surah of an ayah = number of surah starts <= its start word index
    import bisect
    cnt=collections.Counter()
    for i,a in enumerate(ast):
        end=ast[i+1] if i+1<len(ast) else len(words); s=bisect.bisect_right(sst,a); cnt[s]+=1
        verses.append((s,cnt[s],' '.join(words[a:end])))
else:
    verses=[(x['surah'],x['ayah'],' '.join(words[x['start']:x['end']])) for x in ast]
print('warsh verses',len(verses),'first',verses[0][:2],verses[0][2][:40],'| 2:1',[v for v in verses if v[0]==2 and v[1]==1][0][2][:40])
# Warsh writes hamzat al-wasl with a special seat (U+0874) -> normalise to ٱ for our functions; strip Warsh-specific small marks
def wnorm(t): return re.sub('[ࡰ-࢟]','ٱ',t).replace('۬','')
verses=[(s,v,wnorm(t)) for s,v,t in verses]
# basmala unnumbered in Warsh (ayah 0) is not in list if counting starts at 1; check 1:1
# ---- ending definitions (D1, D3 only; same code as pipeline) ----
MARKS=set('ًٌٍَُِّْٰۥۦ'); LET=lambda c:'ء'<=c<='ي' or c=='ٱ'
def lws(t):
    ws=[x for x in t.split() if any(LET(c) for c in x)]
    if not ws: return None
    base=[]
    for c in ws[-1]:
        if LET(c): base.append([c,set()])
        elif base and (c in MARKS or 'ۖ'<=c<='ۭ' or 'ࣰ'<=c<='ࣲ'): base[-1][1].add(c)
    return base
def nrm(c): return {'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ء':'ا','ؤ':'و','ئ':'ي','ى':'ي','ة':'ه'}.get(c,c)
def d1d3(t):
    b=lws(t)
    if not b: return None,None
    L=[nrm(c) for c,_ in b]; last=L[-1]; tf=bool(b[-1][1]&{'ً','ࣰ'})
    d1=(L[-2]+'ا') if (last=='ا' and len(L)>1) else ('@'+last if tf else last)
    if last=='ا' and len(L)>1: d3=('longa',L[-2])
    elif tf: d3=('longa',last)
    else:
        vow='none'
        if len(b)>1:
            p,pm=b[-2]
            if nrm(p) in 'اوي' and not (pm&{'َ','ُ','ِ'}): vow={'ا':'longa','و':'longu','ي':'longi'}[nrm(p)]
            elif 'ٰ' in pm: vow='longa'
            else: vow={'َ':'a','ُ':'u','ِ':'i'}.get(next(iter(pm&{'َ','ُ','ِ'}),'x'),'none')
        d3=(vow,last)
    return d1,d3
WK=[d1d3(t) for *_,t in verses]; WIDX={(s,v):i for i,(s,v,_) in enumerate(verses)}
def neigh(i,k,which):
    s=verses[i][0]; ids=[y for y in range(max(0,i-k),min(len(verses),i+k+1)) if y!=i and verses[y][0]==s]
    return collections.Counter(WK[y][which] for y in ids),len(ids)
# ---- alignment map ----
amap={}
for r in csv.DictReader(open(os.path.join(QW,'data/ayah-map.csv'),encoding='utf8')): amap[(int(r['surah']),int(r['ayah']))]=(r['warsh'],r['warsh_relation'])
def warsh_ref(hs,hv):
    ref,rel=amap[(hs,hv)]
    if '-' in ref: s,rng=ref.split(':'); a,b=rng.split('-'); return int(s),int(b),rel,ref     # split: Hafs verse end = end of last Warsh verse
    s,a=ref.split(':'); return int(s),int(a),rel,ref
# ---- the 30 eligible Hafs pairs ----
elig=list(csv.DictReader(open(os.path.join(H,'..','ELIGIBLE_PAIRS.csv'),encoding='utf8')))
Q=quran(); HT={(s['id'],v['id']):v['text'] for s in Q for v in s['verses']}
def sign_p(a,b):
    n=a+b; return min(1,2*sum(comb(n,x) for x in range(max(a,b),n+1))/2**n) if n else 1.0
rows=[]; own=sw=tie=0; excluded=0; rel_count=collections.Counter(); end_diff=0
for e in elig:
    hi=(int(e['surah_i']),int(e['verse_i'])); hj=(int(e['surah_j']),int(e['verse_j']))
    (si,ai,ri,refi),(sj,aj,rj,refj)=warsh_ref(*hi),warsh_ref(*hj); rel_count[ri]+=1; rel_count[rj]+=1
    wi,wj=WIDX.get((si,ai)),WIDX.get((sj,aj))
    note=''
    # merged: Hafs verse end may not be a Warsh verse end -> check whether Warsh verse ends with the same final word as Hafs
    hfi=clean(HT[hi])[-1]; hfj=clean(HT[hj])[-1]; wfi=clean(verses[wi][2])[-1] if wi is not None else None; wfj=clean(verses[wj][2])[-1] if wj is not None else None
    if ri=='merged' and wfi!=hfi: note+='i: Hafs ending not verse-final in Warsh; '
    if rj=='merged' and wfj!=hfj: note+='j: Hafs ending not verse-final in Warsh; '
    if wfi!=hfi or wfj!=hfj: end_diff+=1
    if note or wi is None or wj is None:
        excluded+=1; rows.append({'pair':e['pair_id'],'warsh_i':refi,'warsh_j':refj,'relation':f'{ri}/{rj}','status':'excluded: '+(note or 'unmapped'),'warsh_final_i':wfi,'warsh_final_j':wfj}); continue
    ei,ej=WK[wi][0],WK[wj][0]
    if ei==ej: rows.append({'pair':e['pair_id'],'warsh_i':refi,'warsh_j':refj,'relation':f'{ri}/{rj}','status':'excluded: same D1 in Warsh','warsh_final_i':wfi,'warsh_final_j':wfj}); excluded+=1; continue
    li,ni=neigh(wi,3,0); lj,nj=neigh(wj,3,0); Dv=((li[ei]>0)+(lj[ej]>0))-((li[ej]>0)+(lj[ei]>0))
    own+=Dv>0; sw+=Dv<0; tie+=Dv==0
    rows.append({'pair':e['pair_id'],'warsh_i':refi,'warsh_j':refj,'relation':f'{ri}/{rj}','status':'tested','D':Dv,'warsh_final_i':wfi,'warsh_final_j':wfj,'hafs_D':e['pair_outcome_D1_k3'],'n_i':ni,'n_j':nj})
# D3 as well
own3=sw3=0
for r in rows:
    if r['status']!='tested': continue
    si,ai=map(int,r['warsh_i'].split(':')[0]),None
res={'warsh_text':{'source':'github.com/quran-ws/quran-text data/mushaf/warsh.json (CC BY 4.0; text per KFGQPC terms)','package':w['provenance']['text']['package'],'sha256':w['provenance']['text']['sha256'],'counting':w['counting']['system'],'ayah_count':w['counting']['ayah_count'],'generated':w['generated']},
     'alignment_relations_of_60_verses':dict(rel_count),'pairs_tested':own+sw+tie,'excluded':excluded,'favour_own':own,'favour_swapped':sw,'ties':tie,'sign_p':sign_p(own,sw),
     'pairs_with_final_word_differing_hafs_vs_warsh':end_diff,'per_pair':rows,
     'caveats':['Pausal pronunciation assumed identical (sukun at pause) for D1; Warsh-specific features (imala, taqlil, naql, long madd) do not alter the final consonant used by D1.','Only D1 recomputed; D3 depends on vowel quality where imala/taqlil could matter and is not reported.','Warsh writes hamzat al-wasl with a distinct seat; normalised before processing.']}
json.dump(res,open(os.path.join(H,'..','results','warsh_replication.json'),'w'),ensure_ascii=False,indent=1,default=str)
print(json.dumps({k:v for k,v in res.items() if k!='per_pair'},ensure_ascii=False,indent=1,default=str))
for r in rows:
    if r['status']!='tested' or r.get('hafs_D','').replace('favours_','')!=('own' if r['D']>0 else 'swapped' if r['D']<0 else 'tie'): print(r)
