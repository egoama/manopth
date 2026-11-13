import sys,collections,json; sys.path.insert(0,sys.argv[2]); from load import load_morph,norm
W=load_morph(sys.argv[1]); Q=json.load(open(sys.argv[3]))
txt={(s['id'],v['id']):v['text'] for s in Q for v in s['verses']}
V=collections.defaultdict(list)
for (s,v,w),segs in W.items():
    for d in segs:
        if 'LEM' in d and d['tag']=='N': V[(s,v)].append((w,norm(d['LEM']),d['feat']))
def first(k,l):
    for w,x,_ in V[k]:
        if x==l: return w
for a,b in [('سماء','ارض'),('ليل','نهار'),('شمس','قمر'),('سمع','بصر'),('سمع','قلب'),('بصر','قلب'),('بر','بحر'),('نور','ظلمه'),('حياه','موت')]:
    both=[k for k in V if first(k,a) and first(k,b)]
    ab=[k for k in both if first(k,a)<first(k,b)]; ba=[k for k in both if first(k,b)<first(k,a)]
    print(f'\n### {a} before {b}: {len(ab)}/{len(both)}; exceptions ({b} first): {len(ba)}')
    if len(ba)<=25:
        for k in ba: print('  %d:%d'%k, txt[k][:110])
# wind singular in punishment
print('\n### ريح singular/plural verses')
for k,toks in sorted(V.items()):
    for w,l,f in toks:
        if l=='ريح': print('  %d:%d'%k, 'PLUR' if 'FP' in f or 'MP' in f else 'SING', txt[k][:90])
