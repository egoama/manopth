"""Intra-verse iltifat candidates: divine first-person-plural action verbs + 'Allah' (3rd person) in the same verse.
Usage: python3 -I iltifat.py <morph.txt> <scripts_dir> <quran.json>"""
import sys,json,collections; sys.path.insert(0,sys.argv[2]); from load import load_morph,norm
W=load_morph(sys.argv[1]); Q=json.load(open(sys.argv[3])); txt={(s['id'],v['id']):v['text'] for s in Q for v in s['verses']}
DIV={norm(x) for x in ['أَنزَلَ','نَزَّلَ','خَلَقَ','جَعَلَ','أَرْسَلَ','آتَى','أَوْحَى','أَهْلَكَ','نَجَّى','أَنجَى','كَتَبَ','رَزَقَ','بارَكَ','أَرَيْ','قَصَّ','صَرَّفَ','زَيَّنَ','سَخَّرَ','أَحْيا','بَعَثَ','وَصَّى','مَكَّنَ','أَغْرَقَ','أَمْطَرَ','نَسَخَ','فَصَّلَ','بَيَّنَ','يَسَّرَ','أَنشَأَ','دَمَّرَ','أَخْرَجَ','أَنبَتَ','أَسْقَى','وَهَبَ','فَضَّلَ','هَدَى']}
div1p=collections.defaultdict(list); allah=set()
for (s,v,w),segs in W.items():
    for d in segs:
        if d['tag']=='V' and '1P' in d['feat'] and norm(d.get('LEM','')) in DIV: div1p[(s,v)].append(norm(d['LEM']))
        if norm(d.get('LEM',''))=='الله': allah.add((s,v))
both=sorted(set(div1p)&allah)
print('verses with divine-1P verbs:',len(div1p),' with also Allah named in 3rd person:',len(both))
for k in both[:12]: print('%d:%d'%k, txt[k][:120])
# adjacent verse shift We<->Allah
keys=[(s['id'],v['id']) for s in Q for v in s['verses']]
adj=[(a,b) for a,b in zip(keys,keys[1:]) if a[0]==b[0] and ((a in div1p and b in allah and b not in div1p) or (b in div1p and a in allah and a not in div1p))]
print('adjacent-verse We<->Allah shifts:',len(adj))
json.dump({'divine_1p_verses':len(div1p),'intra_verse_we_allah':len(both),'intra_examples':['%d:%d'%k for k in both],'adjacent_shifts':len(adj)},open('iltifat.json','w'),ensure_ascii=False)
