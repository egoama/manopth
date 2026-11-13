import sys,collections; sys.path.insert(0,sys.argv[2]); from load import load_morph
W=load_morph(sys.argv[1])
lem=collections.Counter(); lemnum=collections.Counter(); root=collections.Counter()
for k,segs in W.items():
    for d in segs:
        if 'LEM' in d and d['tag']!='P':
            L=d['LEM']; lem[L]+=1
            num='S'
            for x in d['feat']:
                if x in('MD','FD') or (len(x)<=3 and x.endswith('D') and x[0] in'MF'): num='D'
                if x in('MP','FP','P'): num='P'
            lemnum[(L,num)]+=1
        if 'ROOT' in d: root[d['ROOT']]+=1
def show(L):
    print(L, 'total',lem[L],'sing',lemnum[(L,'S')],'dual',lemnum[(L,'D')],'plur',lemnum[(L,'P')])
for L in sys.argv[3:]: show(L)
print('words',len(W),'verses',len({(s,v) for s,v,w in W}),'lemmas',len(lem))
