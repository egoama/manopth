"""Build RESULTS_REVISED.csv from result JSON files (+ documented original values). Applies Holm and BH within family and globally."""
import os,json,csv
from statsmodels.stats.multitest import multipletests
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results')
J=lambda f:json.load(open(os.path.join(R,f)))
e1=J('exp1_phonology.json'); e1b=J('exp1b_engine_validation.json'); e2=J('exp2_fasila_topic.json'); e3=J('exp3_mutashabih.json'); e4=J('exp4_synonyms.json')
e5=J('exp5_stylometry.json'); e6=J('exp6_structure.json'); e6c=J('exp6c_pairs.json'); e6d=J('exp6d_divine_names.json')
L=json.load(open(os.path.join(os.path.dirname(R),'..','letters','letters_results.json'))); LD=json.load(open(os.path.join(os.path.dirname(R),'..','letters','deeper_letters_results.json')))
LN=json.load(open(os.path.join(os.path.dirname(R),'..','letters','nulls_results.json')))
rows=[]
def add(fam,exp,metric,orig,rev,eff,ci,p,status,src,note=''):
    rows.append({'family':fam,'experiment':exp,'metric':metric,'original_result':orig,'revised_result':rev,'effect_size':eff,'ci95':ci,'p_raw':p,'status':status,'source_file':src,'note':note})
r=e1['runs5']; add('phonology','E1','runs>=5 harakat rate ratio Hadith/Quran-quotes','3.18 (7.29 vs 22.77)',f"{r['ratio_hadith_over_quotes']} ({r['quotes_dedup_rate']} vs {r['hadith_dedup_rate']})",'rate ratio',str(r['ci95_cluster']),'','Corrected','revision/results/exp1_phonology.json','dedup + cluster bootstrap; conditional on engine validity')
r=e1['runs5_length_matched_40']; add('phonology','E1','same, length-matched 40-symbol windows','—',str(r['ratio']),'rate ratio',str(r['ci95_cluster']),'','Robust','revision/results/exp1_phonology.json')
for k,v in e1['other_rhythm_features'].items(): add('phonology','E1',k,'61.8% vs 65.0% (share)' if k=='share_mutaharrik' else '—',f"{v['quotes']} vs {v['hadith']}",f"d={v['cohen_d']}",str(v['ci95']),'','Robust','revision/results/exp1_phonology.json')
add('phonology','E1','engine validation (meter id, balanced acc)','—',str(e1b['balanced_accuracy']),f"chance {e1b['chance']}",'','','Preliminary','revision/results/exp1b_engine_validation.json','below pre-set 0.5 threshold')
iv=e1['incremental_value']; add('phonology','E1','AUC lexical vs lexical+prosodic','—',f"{iv['auc_lexical']} -> {iv['auc_lexical_plus_prosodic']}",f"+{round(iv['auc_lexical_plus_prosodic']-iv['auc_lexical'],3)} AUC",'','','Preliminary','revision/results/exp1_phonology.json','grouped CV; no CI')
for m in ('tfidf','root','lsa'):
    v=e2[m]; add('structure','E2',f'cohesion diff changed-same ending ({m})','0.043 vs 0.053 p=0.0005' if m=='tfidf' else '—',f"{v['mean_changed']} vs {v['mean_same']}",f"std diff {v['standardized_diff']}; OLS {v['ols_coef_change_FE_clustered']}",str(v['ols_ci95']),v['perm_p_within_surah'],'Robust','revision/results/exp2_fasila_topic.json','effect very small; no incremental predictive value')
v=e3['by_neighbourhood']['3']; add('structure','E3','pairs favouring own vs swapped ending (k=3)','38 vs 5 verses p=2.5e-7',f"{v['pairs_favour_own']} vs {v['pairs_favour_swapped']} pairs",f"own match {v['own_match_rate']} vs freq null {v['frequency_null_match_rate']}",str(v['own_match_ci95']),v['pair_sign_p'],'Robust','revision/results/exp3_mutashabih.json','robust for k=1..7')
for k,v in e6['6a_dfa'].items():
    add('structure','E6a',f'DFA alpha {k}','0.702 / 0.922 (20 perms)',str(v['alpha']),f"null mean {v['null_mean']}",f"null p99 {v['null_p99']}",v['p'],'Robust' if v['alpha']>v['null_p99'] else 'Preliminary','revision/results/exp6_structure.json')
s=e6['6b_order_sensitivity']['all']; add('structure','E6b','mushaf order smoother than type-preserving random','p=0.0008',str(s['mushaf_p_type_preserving']),'','','%s'%s['mushaf_p_type_preserving'],'Robust','revision/results/exp6_structure.json','holds leaving out any single feature')
add('structure','E6b','chronological order smoother than type-preserving random','p=0.13',str(s['chronological_p_type_preserving']),'','',s['chronological_p_type_preserving'],'Not supported','revision/results/exp6_structure.json')
for r_ in e4['pairs']:
    if 'p_raw' in r_: add('lexical','E4',f"near-synonym {r_['pair']} (balanced acc, grouped by surah)",'see DEEP_AR',str(r_['balanced_acc']),f"baseline Meccan-only {r_['baseline_meccan_only']}",'',r_['p_raw'],r_['status'],'revision/results/exp4_synonyms.json')
    else: add('lexical','E4',f"near-synonym {r_['pair']}",'5/5 vs 0/3','—','','','','Not evaluated','revision/results/exp4_synonyms.json','too few verses')
fu=e4['fuad']; add('lexical','E4','fu\'ad only in Meccan surahs','14/14, p=3.7e-7 (Fisher vs qalb)',f"{fu['n_verses']}/{fu['n_verses']} verses; P by word share {fu['P_all_meccan_if_random_by_words']:.4f}",'','','','Reproduced','revision/results/exp4_synonyms.json','prior descriptive report (Avcı 2020); contextual classifier does not beat Meccan-only baseline')
ls=e5['quran_vs_hadith_leave_one_collection_out']['lr']; add('stylometry','E5','Quran quotes vs Hadith (LR, leave-one-collection-out, balanced acc)','81.5% vs 50% baseline',str(ls['balanced_accuracy']),f"macro-F1 {ls['macro_f1']}",str(ls['ci95']),'','Corrected','revision/results/exp5_stylometry.json','original had imbalance + quote leakage')
add('stylometry','E5','Hadith Qudsi classified as Quran','6.5%',f"{e5['qudsi_threshold_sensitivity']['0.5']} pooled; range by collection 0.08-0.33",'vs 0.18 for ordinary hadith','','','Preliminary','revision/results/exp5_stylometry.json','threshold-sensitive (0.09-0.53)')
add('semantic','E6d','divine-name ending predicted from content (grouped)','43.6% acc, p=0.003',str(e6d['balanced_acc_grouped']),'chance 0.2','',e6d['p'],e6d['decision'],'revision/results/exp6d_divine_names.json','repeated refrains leaked in original')
add('semantic','orig','Moses narrative tailoring','77.6%, p=8.6e-9','not re-evaluated','','','8.6e-9','Not evaluated','semantic/semantic_results.json','verse dependence not modelled')
add('semantic','orig','surah endings more positive','+0.093 p=0.0026','not re-evaluated','','','0.0026','Preliminary','semantic/semantic_results.json','surah-level sign-flip; lexicon-based')
add('semantic','orig','opponents use emphasis more','50% vs 12%','not re-evaluated','','','0.00053','Preliminary','semantic/semantic_results.json','n=82')
add('letters','orig','Zamakhshari halves (frequency-weighted null)','p=0.025 uniform',str(LD['1_weighted_null']['p_frequency_weighted']),'','','%s'%LD['1_weighted_null']['p_frequency_weighted'],'Preliminary','letters/deeper_letters_results.json','post-hoc pattern; Kashshaf passage not re-verified from an edition')
add('letters','orig','Massey single order','—',str(LD['2_massey_order']['p_random_within_combo_orders']),'','','%s'%LD['2_massey_order']['p_random_within_combo_orders'],'Reproduced','letters/deeper_letters_results.json','replication of Massey 1996')
add('letters','orig','initials followed by Book mention','26/29 vs 22/85',LD['3_book_after_initials']['initial_surahs_mentioning_book_in_v1_3'],'','','%s'%LD['3_book_after_initials']['fisher_p'],'Robust','letters/deeper_letters_results.json','classical observation; lexicon-dependent')
add('letters','orig','initial-letter enrichment in own surah','—',str(L['2_initial_enrichment']['mean_percentile_of_initial_letters_in_own_surah']),'null 0.494','','%s'%L['2_initial_enrichment']['p'],'Preliminary','letters/letters_results.json')
add('letters','orig','two pangram verses','true',f"obs 2 vs null {LN['pangram']['null_contiguous_windows_mean']}",'','','%s'%LN['pangram']['p_contiguous_ge_obs'],'Not supported','letters/nulls_results.json','true but expected by chance')
add('letters','orig','Fatiha missing 7 letters','true',f"null mean {LN['fatiha']['missing_null_mean']}",'','','%s'%LN['fatiha']['share_windows_missing_ge7'],'Not supported','letters/nulls_results.json','true but expected by chance')
v=e6c['ALL']; add('numerical','E6c','equal-count opposite pairs (consistent procedure)','6/74 vs 4.8 expected',f"{v['observed_equal']}/{v['pairs_with_both_nonzero']} vs null {v['null_mean']}",'',str(v['null_95_interval']),v['p_ge_observed'],'Not supported','revision/results/exp6c_pairs.json','list has 70 pairs; جن/انس equality is a normalization artefact')
# multiplicity
def num(x):
    try: return float(x)
    except: return None
for fam in sorted({r['family'] for r in rows}):
    ix=[i for i,r in enumerate(rows) if r['family']==fam and num(r['p_raw']) is not None]
    if ix:
        ps=[num(rows[i]['p_raw']) for i in ix]; h=multipletests(ps,method='holm')[1]; b=multipletests(ps,method='fdr_bh')[1]
        for i,a,c in zip(ix,h,b): rows[i]['p_holm_family']=f"{a:.3g}"; rows[i]['p_bh_family']=f"{c:.3g}"
ix=[i for i,r in enumerate(rows) if num(r['p_raw']) is not None]; ps=[num(rows[i]['p_raw']) for i in ix]
h=multipletests(ps,method='holm')[1]; b=multipletests(ps,method='fdr_bh')[1]
for i,a,c in zip(ix,h,b): rows[i]['p_holm_global']=f"{a:.3g}"; rows[i]['p_bh_global']=f"{c:.3g}"
cols=['family','experiment','metric','original_result','revised_result','effect_size','ci95','p_raw','p_holm_family','p_bh_family','p_holm_global','p_bh_global','status','source_file','note']
with open(os.path.join(os.path.dirname(R),'RESULTS_REVISED.csv'),'w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({c:r.get(c,'') for c in cols})
print(len(rows),'rows'); 
for r in rows: print(r['experiment'],'|',r['metric'][:55],'|',r['revised_result'][:35],'|',r.get('p_holm_global',''),'|',r['status'])
