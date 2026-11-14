import sys,os,json,unittest
H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(H,'..','experiments')); sys.path.insert(0,os.path.join(H,'..','..','revision','audit'))
from ind_transcriber import transcribe
R=os.path.join(H,'..','results')
class T(unittest.TestCase):
    def test_transcriber_junctions(self):
        for t,e in [('مِنْ ٱلْكِتَابِ','1101100'),('عَنْ الْيَمِينِ','1101100'),('ٱنظُرۡ كَيۡفَ','1010100'),('فِي ٱلۡأٓخِرِينَ','10101100'),('وَأَقِيمُواْ ٱلصَّلَوٰةَ','1110101100'),('بَل لَّا يُوقِنُونَ','1010101100')]:
            self.assertEqual(transcribe(t),e,t)
    def test_expA_selection(self):
        A=json.load(open(os.path.join(R,'expA_endings.json'))); self.assertEqual((A['candidates'],A['cross_surah'],A['analysed']),(193,159,30))
    def test_expA_direction_all_defs(self):
        A=json.load(open(os.path.join(R,'expA_endings.json')))
        for d,v in A['by_definition'].items():
            for k,x in v.items(): self.assertGreater(x['favour_own'],x['favour_swapped'],(d,k))
    def test_ratio_direction_independent(self):
        B=json.load(open(os.path.join(R,'expB_benchmark.json')))
        for k,v in B['main_ratio_sensitivity'].items(): self.assertGreater(v['ratio'],2.0,k)
    def test_macro_f1_consistent(self):
        S=json.load(open(os.path.join(R,'expS_stylometry_metrics.json'))); self.assertTrue(S['macro_f1_audit']['consistent'])
if __name__=='__main__': unittest.main()
