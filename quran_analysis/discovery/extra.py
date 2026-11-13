import sys,collections; sys.path.insert(0,sys.argv[2]); from load import load_morph,norm
W=load_morph(sys.argv[1]); V=collections.defaultdict(list)
for (s,v,w),segs in W.items():
    for d in segs:
        if 'LEM' in d: V[(s,v)].append((norm(d['LEM']),d['feat'],d['tag']))
def num(f): return 'P' if ('MP' in f or 'FP' in f) else 'S'
for L in ['ظلمه','نور']:
    c=collections.Counter(num(f) for k in V for l,f,t in V[k] if l==L and t=='N'); print(L,dict(c))
RAIN={'سحاب','ماء','مطر','ودق','غيث','بشري','لواقح','رحمه','ارسل'}
for n in 'SP':
    tot=hit=0
    for k,toks in V.items():
        if any(l=='ريح' and num(f)==n for l,f,t in toks):
            tot+=1; ls={l for l,_,_ in toks}; hit+=bool(ls&{'سحاب','ماء','مطر','بشر','لواقح','رحمه'})
    print('wind',n,'verses',tot,'with rain/cloud/glad-tidings/mercy vocab',hit)
