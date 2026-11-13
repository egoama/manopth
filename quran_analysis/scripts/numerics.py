import sys,json,random,collections; sys.path.insert(0,sys.argv[2]); from load import *
Q=json.load(open(sys.argv[3]))
def letters(t): return [ch for ch in norm(t) if 'ء'<=ch<='ي']
b=Q[0]['verses'][0]['text']; print('Basmala:',norm(b),'letters=',len(letters(b)),'(classic 19 claim; depends on orthography)')
print('114 =',114/19,'x19')
s96=Q[95]; print('96 verses:',s96['total_verses'],' words in 96:1-5 =',sum(len(norm(v['text']).split()) for v in s96['verses'][:5]))
nv=sum(s['total_verses'] for s in Q); print('total verses',nv, nv/19)
# abjad
AB=dict(zip('ابجدهوزحطيكلمنسعفصقرشتثخذضظغ',[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]))
AB.update({'ء':1,'ؤ':6,'ئ':10,'ه':5})
def abjad(t): return sum(AB.get(ch,0) for ch in norm(t))
print('abjad(حديد)=',abjad('حديد'),' abjad(الحديد)=',abjad('الحديد'))
# Look-elsewhere experiment: for an arbitrary word, how many "meaningful" numbers can we derive?
targets={26:'Fe atomic no.',56:'Fe-56 mass',57:'Fe-57 isotope',8:'O',7:'N',6:'C',1:'H',2:'He',79:'Au',47:'Ag',29:'Cu',82:'Pb',92:'U',
 12:'months',365:'days',354:'lunar yr',24:'hours',60:'minutes',30:'days/month',29:'synodic month',9:'planets(old)',8.0:'planets',23:'chromosome pairs',46:'chromosomes',
 206:'bones',33:'vertebrae',360:'joints(hadith)',299792:'c km/s',1400:'',19:'code 19',7.0:'heavens'}
tset=set(int(k) for k in targets if k>=10)
idx_s={norm(s['name']):s for s in Q}
def derived(word, sura):
    vals=set(); vals.add(abjad(word)); vals.add(abjad('ال'+word)); vals.add(sura['id']); vals.add(sura['total_verses'])
    for v in sura['verses']:
        if norm(word) in norm(v['text']): vals.add(v['id']); vals.add(sura['id']+v['id'])
    vals.add(len(letters(word)))
    return vals
hits=[]
for s in Q:
    w=norm(s['name']).replace('ال','',1) if norm(s['name']).startswith('ال') else norm(s['name'])
    m=derived(w,s)&tset
    if m: hits.append((s['id'],s['name'],sorted(m)))
print(f'\nSurahs whose name yields >=1 "meaningful constant" via simple derivations: {len(hits)}/114')
for h in hits[:200]: print(h[0],h[1],[ (x,targets.get(x,targets.get(float(x),''))) for x in h[2]])
