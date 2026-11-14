"""Discourse-type classification of nature verses using morphological lemmas (robust to Uthmani spelling).
Priority: eschatological > parable/simile > sign/invitation to observe > narrative > other.
Usage: python3 -I discourse.py <scripts_dir> <quran.json> <morph.txt>"""
import sys,json,collections
sys.path.insert(0,sys.argv[1]); from load import load_morph,norm
Q=json.load(open(sys.argv[2])); W=load_morph(sys.argv[3]); typ={s['id']:s['type'] for s in Q}
N=lambda xs:{norm(x) for x in xs}
NAT=N("""سَماء أَرْض شَمْس قَمَر نَجْم كَوْكَب لَيْل نَهار نُور ظُلُمات ظِلّ جَبَل رَواسِي بَحْر مَوْج مَطَر ماء سَحاب رِيح بَرْق رَعْد بَرَد
نَبات شَجَر شَجَرَة زَرْع ثَمَرَة حَبّ نَخْل نُطْفَة عَلَقَة مُضْغَة عَظْم لَحْم دَم لَبَن عَسَل نَحْل نَمْل طَيْر دابَّة تُراب طِين حِجارَة حَدِيد دُخان فَلَك""".split())
ESCH=N(['قِيامَة','ساعَة','صُور','حَشَرَ','جَهَنَّم','آخِرَة','حِساب','نَفَخَ','يَوْمَئِذٍ','قارِعَة','واقِعَة','حاقَّة','سَعِير','جَحِيم','فِرْدَوْس','خالِد'])
PARA=N(['مَثَل','كَأَنَّ','كَأَنَّما'])
SIGN=N(['آيَة','تَفَكَّرَ','عَقَلَ','تَذَكَّرَ','اعْتَبَرَ','نَظَرَ','أَبْصَرَ'])
lem=collections.defaultdict(set); pn=collections.defaultdict(set); first=collections.defaultdict(str)
for (s,v,w),segs in W.items():
    for d in segs:
        if 'LEM' in d:
            lem[(s,v)].add(norm(d['LEM']))
            if 'PN' in d['feat']: pn[(s,v)].add(norm(d['LEM']))
        if w==1 and 'LEM' in d and not first[(s,v)]: first[(s,v)]=norm(d['LEM'])
NARR_PN=N(['مُوسَى','فِرْعَوْن','نُوح','عاد','ثَمُود','لُوط','إِبْراهِيم','صالِح','شُعَيْب','هُود','سُلَيْمان','داوُۥد','يُوسُف','عِيسَى','مَرْيَم','إِسْرائِيل','قارُون','هامان','زَكَرِيّا'])|N(['قَوْم'])
cats=collections.Counter(); bytype=collections.defaultdict(collections.Counter); ex=collections.defaultdict(list)
nat=[k for k in lem if lem[k]&NAT]
for k in sorted(nat):
    L=lem[k]
    if L&ESCH or (first[k]==norm('إِذا') and k[0] in (56,69,77,78,79,80,81,82,84,99,101)): c='eschatological'
    elif L&PARA: c='parable_simile'
    elif L&SIGN: c='sign_invitation'
    elif (pn[k]|L)&NARR_PN: c='narrative'
    else: c='other'
    cats[c]+=1; bytype[typ[k[0]]][c]+=1
    if len(ex[c])<6: ex[c].append('%d:%d'%k)
n=len(nat)
out={'nature_verses':n,'categories':{c:[v,round(v/n,3)] for c,v in cats.most_common()},'by_type':{t:dict(c) for t,c in bytype.items()},'examples':ex}
json.dump(out,open('discourse.json','w'),ensure_ascii=False,indent=1); print(json.dumps(out,ensure_ascii=False,indent=1))
import random; random.seed(5)
txt={(s['id'],v['id']):v['text'] for s in Q for v in s['verses']}
lab={k:None for k in nat}
for k in sorted(nat):
    L=lem[k]
    if L&ESCH or (first[k]==norm('إِذا') and k[0] in (56,69,77,78,79,80,81,82,84,99,101)): c='E'
    elif L&PARA: c='P'
    elif L&SIGN: c='S'
    elif (pn[k]|L)&NARR_PN: c='N'
    else: c='O'
    lab[k]=c
for k in random.sample(sorted(nat),40): print('%d:%d'%k, lab[k], txt[k][:95])
