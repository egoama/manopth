"""Arudi (Khalilian) prosodic encoding: vocalized Arabic -> string of 1 (mutaharrik) / 0 (sakin).
Handles standard vocalization (hadith, poetry) and Tanzil Uthmani marks (Quran)."""
import re
FATHA,DAMMA,KASRA='َ','ُ','ِ'
SUKUN={'ْ','ۡ'}; SHADDA='ّ'
TANWIN={'ً','ٌ','ٍ','ࣰ','ࣱ','ࣲ'}
LONG_MARK={'ٰ','ۥ','ۦ'}  # superscript alif, small waw, small ya
MADDA='ٓ'
VOWELS={FATHA,DAMMA,KASRA}
LETTER=re.compile('[ء-يٱ]')
def letters(text):
    out=[]
    for ch in text:
        if LETTER.match(ch): out.append([ch,set()])
        elif out and 'ً'<=ch<='ۭ' or (out and 'ࣰ'<=ch<='ࣲ'): out[-1][1].add(ch)
        elif ch==' ': out.append([' ',set()])
    return out
def encode_raw(text, pause='sakin'):
    s=[]; prev_v=None; last_tanwin=None
    L=[x for x in letters(text) if x[0]!=' ']
    for i,(ch,m) in enumerate(L):
        if ch=='\u0671': continue                      # hamzat wasl: silent in connected speech
        if ch=='\u0622':                               # alif madda
            s+= ['0'] if prev_v==FATHA else ['1','0']; prev_v=None; last_tanwin=None; continue
        v=m&VOWELS; t=m&TANWIN; last_tanwin=None
        if ch=='ا' and m&SUKUN: continue                # silent alif (e.g. after waw al-jama'a)
        if SHADDA in m: s.append('0')
        if t: s+=['1','0']; prev_v=None; last_tanwin=('fath' if ('\u064B' in t or '\u08F0' in t) else 'other')
        elif v: s.append('1'); prev_v=next(iter(v))
        elif m&SUKUN: s.append('0'); prev_v=None
        elif (ch in 'اى' and prev_v==FATHA) or (ch=='و' and prev_v==DAMMA) or (ch=='ي' and prev_v==KASRA):
            s.append('0'); prev_v=None
        elif ch in 'اى': pass                           # silent alif (wasl / orthographic)
        else:
            nxt=L[i+1][1] if i+1<len(L) else set()
            if SHADDA in nxt: pass                       # assimilated (idgham / shamsi lam)
            elif ch=='ل' and i>0 and L[i-1][0] in 'ٱا' : pass   # lam of al- (Uthmani unmarked when shamsi handled above)
            else: s.append('0'); prev_v=None              # hidden/unmarked sakin (ikhfa etc.)
        if m&LONG_MARK and not (s and s[-1]=='0' and ch in 'اىوي' and not v): s.append('0'); prev_v=None
        # lafz al-jalala: (ا)لـلَّـه -> long a not written
        if ch=='ل' and SHADDA in m and FATHA in m and i+1<len(L) and L[i+1][0]=='ه' and i>0 and L[i-1][0]=='ل' and not (m&LONG_MARK):
            s.append('0')
    if s and pause=='sakin':
        if last_tanwin=='other': s=s[:-2]+['0']
        elif last_tanwin=='fath': pass                   # -an -> long a at pause
        elif s[-1]=='1': s[-1]='0'
    elif s and pause=='ishba' and s[-1]=='1': s.append('0')
    return ''.join(s)
def diacritic_coverage(text):
    L=sum(1 for c in text if 'ء'<=c<='ي'); M=sum(1 for c in text if 'ً'<=c<='ْ')
    return M/L if L else 0

def encode(text, pause='sakin'):
    words=[w for w in text.split() if LETTER.search(w)]
    if not words: return ''
    parts=[encode_raw(w,pause='none') for w in words]
    out=''
    for p_ in parts:
        if out and p_.startswith('0') and out.endswith('0'): out=out[:-1]
        out+=p_
    # apply pause on the final word using full rule
    last=encode_raw(words[-1],pause)
    lastraw=parts[-1]
    if out.endswith(lastraw) and lastraw: out=out[:len(out)-len(lastraw)]+last
    return out
