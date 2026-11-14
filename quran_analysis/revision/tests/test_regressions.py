"""Regression tests for confirmed audit issues. Run: QDATA=<data dir> python3 -m unittest discover -s tests -v"""
import sys,os,json,unittest,collections,re
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','audit'))
from common import *
RES=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results')
class T(unittest.TestCase):
    @classmethod
    def setUpClass(c):
        c.Q=quran(); c.W=load_morph(MORPH)
    def test_dataset_counts(self):
        self.assertEqual(len(self.Q),114); self.assertEqual(sum(len(s['verses']) for s in self.Q),6236)
        self.assertEqual(len(self.W),77429)
    def test_norm_collision_kaana(self):
        # AUDIT-07: normalization makes كَأَنَّ identical to كان -> lexicons must use raw lemmas
        self.assertEqual(norm('كَأَنَّ'),norm('كانَ'))
    def test_norm_merges_ins(self):
        # AUDIT-06: إِنس and آنَسَ collide after normalization -> equal-count artefact (جن/انس)
        self.assertEqual(norm('إِنس'),norm('آنَسَ'))
    def test_prosody_gold(self):
        g=[('قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ','ishba','11010110101011010110110'),('قُلْ هُوَ ٱللَّهُ أَحَدٌ','sakin','10110101110')]
        for t,p,e in g: self.assertEqual(encode(t,p),e)
    def test_issue_B_192_vs_190(self):
        d=json.load(open(os.path.join(RES,'audit_issues_ABC.json')))['issue_B']
        self.assertEqual(d['verses_with_kept_endings'],192); self.assertEqual(len(d['dropped_empty_feature_verses']),2)
    def test_issue_C_quote_duplication(self):
        d=json.load(open(os.path.join(RES,'audit_issues_ABC.json')))['issue_C']
        self.assertGreater(d['instances_that_are_repeats'],0)   # leakage risk confirmed
    def test_fuad_all_meccan(self):
        d=json.load(open(os.path.join(RES,'exp4_synonyms.json')))['fuad']
        self.assertTrue(d['all_in_meccan_surahs']); self.assertEqual(d['n_verses'],15)
    def test_pairs_list_is_70_not_74(self):
        d=json.load(open(os.path.join(RES,'exp6c_pairs.json')))
        self.assertEqual(d['ALL']['pairs_listed'],70)
    def test_mutashabih_counts(self):
        d=json.load(open(os.path.join(RES,'exp3_mutashabih.json')))
        self.assertEqual(d['candidate_pairs'],193); self.assertEqual(d['analysed_pairs_diff_endings'],30)
if __name__=='__main__': unittest.main()
