"""Root concordance for candidate key words: all Quranic uses of each root (semantic range check).
Usage: python3 -I roots.py <morph.txt> <scripts_dir> <quran.json>"""
import sys,json,collections; sys.path.insert(0,sys.argv[2]); from load import load_morph,norm
W=load_morph(sys.argv[1]); Q=json.load(open(sys.argv[3])); txt={(s['id'],v['id']):v['text'] for s in Q for v in s['verses']}
C={'A1':'وسق','A2':'كشط','A3':'وهي','A4':'طوي','A5':'كسف','A6':'نثر','A7':'فوت','A8':'حثث','A9':'شوظ','A10':'حرس',
'B1':'قدر','B2':'صلد','B3':'هور','B4':'شمخ','B5':'بسط','B6':'عرض','B7':'بلع','B8':'هيل','B9':'هبط','B10':'سنو','B11':'غور','B12':'عصر',
'C1':'نطق','C2':'لزب','C3':'أسن','C4':'سفح','C6':'ضوأ','C7':'نبت','C11':'ذرو','C12':'شطأ',
'D1':'قشعر','D2':'شعل','D3':'نعس','D4':'وسن','D5':'كلح','D6':'جلد','D8':'نشأ','D9':'رقد'}
occ=collections.defaultdict(list)
for (s,v,w),segs in W.items():
    for d in segs:
        if 'ROOT' in d: occ[d['ROOT']].append(((s,v),d['form']))
out={}
for cid,r in C.items():
    vs=[]; seen=set()
    for k,f in occ.get(r,[]):
        if k not in seen: seen.add(k); vs.append('%d:%d'%k)
    out[cid]={'root':r,'n_occurrences':len(occ.get(r,[])),'verses':vs[:12]}
    print(cid,r,len(occ.get(r,[])),' '.join(vs[:12]))
json.dump(out,open('roots.json','w'),ensure_ascii=False,indent=0)
