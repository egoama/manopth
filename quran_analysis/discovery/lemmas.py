import sys,collections; sys.path.insert(0,sys.argv[2]); from load import load_morph,norm
W=load_morph(sys.argv[1]); c=collections.Counter()
for segs in W.values():
    for d in segs:
        if d['tag']=='N' and 'LEM' in d: c[(norm(d['LEM']),d.get('ROOT',''))]+=1
want=sys.argv[3].split(',')
for w in want:
    hits=[(k,v) for k,v in c.items() if k[0]==w]
    print(w, hits if hits else 'MISSING')
