import re, collections, json, sys
D=sys.argv[1] if len(sys.argv)>1 else None
def load_morph(p):
    words=collections.OrderedDict()  # (s,v,w)->list of segments
    for line in open(p,encoding='utf8'):
        loc,form,tag,feat=line.rstrip('\n').split('\t')
        s,v,w,k=map(int,loc.split(':'))
        f=feat.split('|'); d={'form':form,'tag':tag,'feat':f}
        for x in f:
            if ':' in x: a,b=x.split(':',1); d[a]=b
        words.setdefault((s,v,w),[]).append(d)
    return words
DIAC=re.compile('[ً-ٰٟۖ-ۭـ]')
def norm(s):
    s=DIAC.sub('',s); s=s.replace('ٱ','ا').replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ى','ي').replace('ة','ه').replace('ـ','')
    return s
def lemma_table(W):
    import collections
    T=collections.defaultdict(lambda: collections.Counter())
    for k,segs in W.items():
        for d in segs:
            if 'LEM' in d and d['tag']!='P':
                num='S'
                for x in d['feat']:
                    if x in('MD','FD'): num='D'
                    if x in('MP','FP'): num='P'
                T[d['LEM']][num]+=1; T[d['LEM']]['ALL']+=1
    return T
