import sys; sys.path.insert(0,sys.argv[2]); from load import *
W=load_morph(sys.argv[1]); T=lemma_table(W)
idx={}
for L in T: idx.setdefault(norm(L),[]).append(L)
for q in sys.argv[3:]:
    for L in idx.get(norm(q),[]): print(q,'->',L,dict(T[L]))
    if norm(q) not in idx: print(q,'NOT FOUND')
