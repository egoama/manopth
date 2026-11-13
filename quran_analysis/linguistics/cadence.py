"""Verse-final cadence and length symmetry (tawazun) of consecutive Quranic verses.
Usage: python3 -I cadence.py <quran.json>"""
import sys,json,random,collections,statistics as st
sys.path.insert(0,'.'); from prosody import encode
random.seed(3); Q=json.load(open(sys.argv[1]))
cad=collections.Counter(); res={}
lens={}
for s in Q:
    L=[]
    for v in s['verses']:
        e=encode(v['text']); cad[e[-4:]]+=1; L.append(len(e))
    lens[s['id']]=(s['type'],L)
tot=sum(cad.values()); res['final4_cadence_top']=[(k,round(n/tot,3)) for k,n in cad.most_common(6)]
def adjdiff(L): return st.mean(abs(a-b)/((a+b)/2) for a,b in zip(L,L[1:]))
for typ in ('meccan','medinan'):
    obs=[];sh=[]
    for t,L in lens.values():
        if t!=typ or len(L)<5: continue
        obs.append(adjdiff(L))
        for _ in range(20):
            M=L[:]; random.shuffle(M); sh.append(adjdiff(M))
    res['length_symmetry_'+typ]={'mean_rel_diff_adjacent':round(st.mean(obs),3),'shuffled':round(st.mean(sh),3)}
json.dump(res,open('cadence.json','w'),ensure_ascii=False); print(res)
