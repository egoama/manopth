"""Sampling frame: verses with natural-phenomena lemmas, and rare lemmas (freq<=3) inside them.
Usage: python3 -I frame.py <morph.txt> <scripts_dir> <quran.json>"""
import sys,json,collections; sys.path.insert(0,sys.argv[2]); from load import load_morph,norm
W=load_morph(sys.argv[1]); Q=json.load(open(sys.argv[3])); txt={(s['id'],v['id']):v['text'] for s in Q for v in s['verses']}
NAT=set(norm(x) for x in """سَماء أَرْض شَمْس قَمَر نَجْم كَوْكَب لَيْل نَهار صُبْح فَجْر نُور ظُلُمات ظِلّ جَبَل رَواسِي بَحْر مَوْج مَطَر ماء سَحاب رِيح بَرْق رَعْد بَرَد
نَبات شَجَر شَجَرَة زَرْع ثَمَرَة حَبّ نَخْل زَوْج نُطْفَة عَلَقَة مُضْغَة عَظْم لَحْم دَم لَبَن عَسَل نَحْل نَمْل طَيْر دابَّة تُراب طِين حِجارَة حَدِيد نار دُخان فَلَك
سَمْع بَصَر قَلْب فُؤاد جِلْد صَدْر أَرْحام بَطْن عَيْن أُذُن لِسان يَد رِجْل ناصِيَة بَنان صُلْب تَرائِب رِيح بَحْرَيْن بَرْزَخ نَهَر عَيْن يَنْبُوع ثَلْج بُرُوج مَصابِيح شُهُب
مَشْرِق مَغْرِب ضُحَى شَفَق غَسَق سِراج ضِياء نُحاس قِطْر ذَرَّة فُطُور فُرُوج حُبُك رَجْع صَدْع دَحا طَحا مِهاد أَوْتاد كِفات سَبْع طِباق عَرْش""".split())
freq=collections.Counter(); vl=collections.defaultdict(set)
for (s,v,w),segs in W.items():
    for d in segs:
        if 'LEM' in d and d['tag'] in('N','V'):
            L=d['LEM']; freq[L]+=1; vl[(s,v)].add(L)
rows=[]
for k,ls in vl.items():
    nat=[l for l in ls if norm(l) in NAT]
    if not nat: continue
    rare=[l for l in ls if freq[l]<=3]
    if rare: rows.append(('%d:%d'%k,sorted(set(norm(x) for x in nat)),sorted(rare,key=lambda l:freq[l]),txt[k]))
rows.sort(key=lambda r:(int(r[0].split(':')[0]),int(r[0].split(':')[1])))
print('nature verses:',sum(1 for k in vl if any(norm(l) in NAT for l in vl[k])),' with rare lemmas:',len(rows))
with open('frame.tsv','w') as f:
    for r in rows: f.write('\t'.join([r[0],' '.join(r[1]),' '.join(norm(x)+'(%d)'%freq[x] for x in r[2]),r[3]])+'\n')
