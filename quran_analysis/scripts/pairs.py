import sys,random,collections; sys.path.insert(0,sys.argv[2]); from load import *
W=load_morph(sys.argv[1]); T=lemma_table(W)
idx=collections.defaultdict(list)
for L in T: idx[norm(L)].append(L)
def c(q,mode='ALL'):
    return sum(T[L][mode] for L in idx.get(norm(q),[]))
pairs=[('ليل','نهار'),('شمس','قمر'),('حر','برد'),('نور','ظلمه'),('خير','شر'),('صيف','شتاء'),('جنه','نار'),('حياه','موت'),
('سماء','ارض'),('غني','فقير'),('عسر','يسر'),('ذكر','انثي'),('حسنه','سيئه'),('ثواب','عقاب'),('رحمه','عذاب'),('ايمان','كفر'),
('هدي','ضلال'),('علم','جهل'),('عدل','ظلم'),('صدق','كذب'),('حق','باطل'),('قليل','كثير'),('شرق','غرب'),('مشرق','مغرب'),('يمين','شمال'),
('فوق','تحت'),('اول','اخر'),('بصر','سمع'),('قلب','عقل'),('ماء','نار'),('بر','بحر'),('جبل','بحر'),('ظل','حرور'),('رجل','امرات'),
('ملك','شيطان'),('دنيا','اخر'),('قيامه','بعث'),('نفع','ضر'),('صلاه','زكاه'),('مومن','كافر'),('مسلم','مشرك'),('خوف','رجاء'),
('حزن','فرح'),('ضحك','بكي'),('عزه','ذله'),('سر','علن'),('غيب','شهاده'),('جن','انس'),('ولد','والد'),('اب','ام'),('ابن','بنت'),
('ذهب','فضه'),('لؤلؤ','مرجان'),('نخل','زيتون'),('عنب','رمان'),('بقره','غنم'),('شجر','نبات'),('مطر','سحاب'),('ريح','سحاب'),('نجم','كوكب'),
('صبر','شكر'),('توبه','مغفره'),('لسان','شفه'),('عين','اذن'),('يد','رجل'),('سبع','سبعه'),('نوح','ابراهيم'),('موسي','فرعون'),('عيسي','مريم'),('محمد','احمد')]
rows=[]
for a,b in pairs:
    ca,cb=c(a),c(b); sa,sb=c(a,'S'),c(b,'S'); rows.append((a,b,ca,cb,sa,sb))
    flag='EQUAL(all)' if ca==cb and ca>0 else ('EQUAL(sing)' if sa==sb and sa>0 else '')
    print(f'{a}/{b}: all {ca}/{cb}  sing {sa}/{sb} {flag}')
# baseline: random lemma pairs with counts within same range as a tested pair
counts=[T[L]['ALL'] for L in T]
random.seed(1); tests=0; hits=0
for a,b,ca,cb,sa,sb in rows:
    if ca==0 or cb==0: continue
    pool=[x for x in counts if 0.5*min(ca,cb)<=x<=2*max(ca,cb)]
    for _ in range(2000):
        x,y=random.sample(pool,2); tests+=1; hits+= (x==y)
print('baseline P(equal | random lemmas of similar frequency) =',hits/tests)
