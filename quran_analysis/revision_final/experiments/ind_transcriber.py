"""Independent prosodic transcriber implemented from PHONOLOGICAL_SPEC.md (syllable-structured; separate code path from prosody.encode)."""
import re
V={'َ':'a','ُ':'u','ِ':'i'}; TAN={'ً':'an','ٌ':'un','ٍ':'in','ࣰ':'an','ࣱ':'un','ࣲ':'in'}
SUK={'ْ','ۡ'}; SH='ّ'; DAG={'ٰ','ۥ','ۦ'}; MAD='ٓ'
LET=lambda c:('ء'<=c<='ي') or c=='ٱ'
def parse(word):
    out=[]
    for c in word:
        if LET(c): out.append([c,set()])
        elif out and (('ً'<=c<='ْ') or c in DAG or c==MAD or ('ࣰ'<=c<='ࣲ') or c=='ۡ'): out[-1][1].add(c)
    return out
def word_units(letters,start):
    """Return list of units: ('C',vowel|None) consonant with/without vowel, ('L',) lengthening, ('N',vowel) tanwin nasal pending."""
    U=[]; prev_v=None
    n=len(letters)
    for i,(c,m) in enumerate(letters):
        nxt=letters[i+1][1] if i+1<n else set()
        if c=='ٱ' or (c=='ا' and i==0 and not (m&set(V)) and not (m&set(TAN)) and not (m&SUK)):
            # hamzat al-wasl: Uthmani wasla, or plain unmarked initial alif in standard orthography (S8)
            if start and i==0: U.append(('C','a')); prev_v='a'
            continue
        if c=='آ' or (MAD in m and c in 'أاء' and not (m&set(V))):
            # S7: alif-madda after a fatha-bearing letter is pure lengthening (Uthmani لَآ, يَشَآءُ); otherwise hamza + long a
            if c=='آ' and prev_v=='a': U.append(('L',))
            else: U.append(('C','a')); U.append(('L',))
            prev_v=None; continue
        if c=='ا' and (m&SUK): continue
        v=[V[x] for x in m if x in V]; t=[TAN[x] for x in m if x in TAN]
        if SH in m: U.append(('C',None))
        if t: U.append(('C',t[0][0])); U.append(('N',t[0])); prev_v=None
        elif v: U.append(('C',v[0])); prev_v=v[0]
        elif m&SUK: U.append(('C',None)); prev_v=None
        elif (c in 'اى' and prev_v=='a') or (c=='و' and prev_v=='u') or (c=='ي' and prev_v=='i'): U.append(('L',)); prev_v=None
        elif c in 'اى': prev_v=None
        elif c in 'وي' and (m&DAG): pass            # orthographic carrier of a dagger-alif long vowel (الصلوٰة, الزكوٰة)
        else:
            if SH in nxt: pass
            elif c=='ل' and i>0 and letters[i-1][0] in 'ٱا': pass
            else: U.append(('C',None)); prev_v=None
        if m&DAG and not (U and U[-1]==('L',)): U.append(('L',)); prev_v=None
        if c=='ل' and SH in m and 'a' in [V[x] for x in m if x in V] and i+1<n and letters[i+1][0]=='ه' and i>0 and letters[i-1][0]=='ل' and not (m&DAG): U.append(('L',))
    return U
def to_code(U):
    s=[]
    for u in U:
        if u[0]=='C': s.append('1' if u[1] else '0')
        elif u[0]=='L': s.append('0')
        elif u[0]=='N': s.append('0')
    return s
def transcribe(text,pause='sakin'):
    words=[w for w in text.split() if any(LET(c) for c in w)]
    if not words: return ''
    units=[]
    for wi,w in enumerate(words):
        L=parse(w); U=word_units(L,wi==0)
        if not U: continue
        NL=parse(words[wi+1]) if wi+1<len(words) else []
        nxt_wasl = bool(NL) and (NL[0][0]=='ٱ' or (NL[0][0]=='ا' and not (NL[0][1]&set(V)) and not (NL[0][1]&set(TAN)) and not (NL[0][1]&SUK)))
        nxt_shadda = bool(NL) and (SH in NL[0][1])
        if nxt_shadda and U and U[-1][0]=='C' and U[-1][1] is None: U=U[:-1]      # cross-word idgham (S10)
        if nxt_wasl and units is not None and U:
            last=U[-1]
            if last==('L',): U=U[:-1]                      # S12
            elif last[0]=='C' and last[1] is None: U[-1]=('C','i')   # S13
            elif last[0]=='N': U[-1]=('C','i')               # S14 nasal vowelled
        units.append(U)
    if not units: return ''
    last=units[-1]
    if pause=='sakin':
        if last and last[-1][0]=='N':
            if last[-1][1]=='an': last[-1]=('L',)           # -an -> long a
            else: last.pop(); 
            if last and last[-1][0]=='C' and last[-1][1] and units[-1][-1][0]!='L' and last[-1][1] in('u','i') and len(last)>=1 and last[-1]!=('L',):
                # damm/kasr tanwin at pause: consonant becomes sakin
                last[-1]=('C',None)
        elif last and last[-1][0]=='C' and last[-1][1]: last[-1]=('C',None)
    elif pause=='ishba':
        if last and last[-1][0]=='C' and last[-1][1]: last.append(('L',))
    return ''.join(''.join(to_code(U)) for U in units)
if __name__=='__main__':
    for t in ['قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ','قُلْ هُوَ ٱللَّهُ أَحَدٌ','مِنْ ٱلْكِتَابِ','فِي ٱلْأَرْضِ','ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَالَمِينَ','كِتَابٌ أَنزَلْنَاهُ']:
        print(t,transcribe(t,'ishba' if 'قِفَا' in t else 'sakin'))
