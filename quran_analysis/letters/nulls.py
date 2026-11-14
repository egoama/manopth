"""Null models for pangram verses, al-Fatiha missing letters, and 19-divisibility base rate.
Usage: python3 -I nulls.py <quran.json>"""
import sys,json,random,collections,statistics as st
sys.path.insert(0,'.')
random.seed(7)
Q=json.load(open(sys.argv[1]))
ALPH=list('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
MAP={'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ء':'ا','ؤ':'و','ئ':'ي','ى':'ي','ة':'ه'}
def letters(t):
    return [MAP.get(c,c) for c in t if c in MAP or c in ALPH]

V=[letters(v['text']) for s in Q for v in s['verses']]
allL=[a for L in V for a in L]
lens=[len(L) for L in V]
obs=sum(1 for L in V if set(ALPH)<=set(L))
# null A: letters shuffled across Quran, verse lengths kept
null=[]
for _ in range(300):
    random.shuffle(allL); i=0; c=0
    for n in lens:
        if set(ALPH)<=set(allL[i:i+n]): c+=1
        i+=n
    null.append(c)
# null B: natural text windows (contiguous) of the same lengths, sliding across Quran (keeps word structure)
flat=[a for L in V for a in L]
nat=[]
for _ in range(300):
    c=0
    for n in random.sample(lens,len(lens)):
        j=random.randrange(0,len(flat)-n); 
        if set(ALPH)<=set(flat[j:j+n]): c+=1
    nat.append(c)
# pangram probability vs verse length: lengths of the two pangram verses and rank
plen=sorted(((len(L),i) for i,L in enumerate(V)),reverse=True)
idx={i:r for r,(n,i) in enumerate(plen)}
keys=[(s['id'],v['id']) for s in Q for v in s['verses']]
pv=[i for i,L in enumerate(V) if set(ALPH)<=set(L)]
# Fatiha: missing letters in contiguous windows of same letter length
F=sum(len(L) for L in V[:7])
miss=[]
for _ in range(20000):
    j=random.randrange(0,len(flat)-F); miss.append(28-len(set(flat[j:j+F])))
R={'pangram':{'observed':obs,'null_shuffled_letters_mean':round(st.mean(null),2),'null_contiguous_windows_mean':round(st.mean(nat),2),
   'p_contiguous_ge_obs':round(sum(x>=obs for x in nat)/len(nat),3),'p_contiguous_le_obs':round(sum(x<=obs for x in nat)/len(nat),3),
   'pangram_verse_length_rank':[('%d:%d'%keys[i],len(V[i]),idx[i]+1) for i in pv],'longest_verse_letters':plen[0][0]},
   'fatiha':{'letters':F,'missing_observed':7,'missing_null_mean':round(st.mean(miss),2),'share_windows_missing_ge7':round(sum(m>=7 for m in miss)/len(miss),3),
   'missing_distribution':dict(sorted(collections.Counter(miss).items()))},
   'nineteen_base_rate':{'P(divisible by 19)':round(1/19,4),'claims_tested':4,'hits':2,'P(>=2 of 4 by chance)':round(1-((18/19)**4+4*(1/19)*(18/19)**3),4)}}
json.dump(R,open('nulls_results.json','w'),ensure_ascii=False,indent=1); print(json.dumps(R,ensure_ascii=False,indent=1))
