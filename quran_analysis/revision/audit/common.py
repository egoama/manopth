"""Shared, audited data loaders for the revision. Paths come from env var QDATA (scratch data dir)."""
import os,sys,re,csv,glob,json,collections
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,os.path.join(ROOT,'scripts')); sys.path.insert(0,os.path.join(ROOT,'linguistics'))
from load import load_morph,norm            # original helpers (unchanged)
from prosody import encode                  # original prosody engine (unchanged)
D=os.environ.get('QDATA','/tmp/claude-0/-home-user-manopth/934b0512-6583-5a49-9798-78126e4f0360/scratchpad')
QJSON=D+'/data/package/dist/quran.json'; MORPH=D+'/data/morph.txt'; HAD=D+'/ext/hadith'; APCD=D+'/ext/apcd_x/APCD_plus_porse_all.csv'
def quran(): return json.load(open(QJSON))
SALLA=re.compile('وَسَل[َّ]{2}مَ')
def clean(t): return re.sub('[^ء-ي ]',' ',norm(t)).split()
def hadith_records():
    """One record per hadith row: collection, id, matn (after first salla formula, quotes removed), quotes list."""
    out=[]
    for f in sorted(glob.glob(os.path.join(HAD,'*','*mushakkala*.csv'))):
        col=os.path.basename(os.path.dirname(f))
        for r in csv.reader(open(f,encoding='utf8')):
            t=r[1].replace('‏',' ')
            qs=[x for x in re.findall(r'\{([^{}]+)\}',t)]
            m=SALLA.search(t)
            matn=re.sub(r'\{[^{}]*\}',' ',t[m.end():]) if m else None
            out.append({'col':col,'id':r[0],'matn':matn,'quotes':qs})
    return out
