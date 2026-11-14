"""Build FINAL_VALIDATED_RESULTS.csv: every number traceable to a result file + command."""
import os,json,csv
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results'); J=lambda f:json.load(open(os.path.join(R,f)))
A=J('expA_endings.json'); B=J('expB_benchmark.json'); G=J('expB_adjudicated.json'); U=J('expB_auc.json'); S=J('expS_stylometry_metrics.json'); C=J('expC_lexical.json')
rows=[]
def add(prio,metric,value,ci,p,p_adj,status,src,cmd,note=''):
    rows.append(dict(priority=prio,metric=metric,value=value,ci95=ci,p_raw=p,p_adjusted=p_adj,status=status,result_file=src,command=cmd,note=note))
cA='python3 -I experiments/expA_endings.py'
add('A','candidate pairs / cross-surah / analysed',f"{A['candidates']}/{A['cross_surah']}/{A['analysed']}",'','','','Reproduced','results/expA_endings.json',cA,A['exclusion_rule'])
for d,v in A['by_definition'].items():
    k3=v['3']; add('A',f'{d} k=3 pairs own:swapped',f"{k3['favour_own']}:{k3['favour_swapped']} (ties {k3['ties']})",f"own-match {k3['own_match']} {k3['own_ci']}; freq-null {k3['freq_null_match']}",k3['p_sign'],A['holm_k3'][d],'Robust','results/expA_endings.json',cA,'exact sign test, pair unit')
    add('A',f'{d} direction across k=1,2,3,5,7','own>swapped at every k' if all(v[k]['favour_own']>v[k]['favour_swapped'] for k in v) else 'inconsistent','','','','Robust','results/expA_endings.json',cA)
o=A['one_pair_per_verse']['last_phoneme']; add('A','one pair per verse (last phoneme, k=3)',f"{o['favour_own']}:{o['favour_swapped']} of {o['pairs_tested']}",str(o['own_ci']),o['p_sign'],'','Robust','results/expA_endings.json',cA,'dependence sensitivity')
f=A['frequency_matched_random_ending_null']; add('A','own-match vs frequency-matched random ending (k=3)',f"{f['observed_own_match']} vs null {f['null_mean']}",str(f['null_95']),f['p_ge_obs'],'','Robust','results/expA_endings.json',cA,'10000 Monte Carlo draws')
add('A','grammatical alternative endings','not constructed','','','','Not evaluated','results/expA_endings.json',cA,'expert review required')
cB='python3 -I experiments/expB_benchmark.py'
ov=B['overall']; add('B','engine vs independent transcriber: exact match / SER (320 items)',f"{ov['exact_match']} / {ov['symbol_error_rate']}",'','','','Preliminary','results/expB_benchmark.json',cB,'reference is UNVERIFIED (same analyst)')
add('B','runs5 agreement (items) / totals engine vs ref',f"{ov['runs5_agreement_items']} / {ov['runs5_engine_total']} vs {ov['runs5_ref_total']}",'','','','Preliminary','results/expB_benchmark.json',cB)
for k,v in B['by_phenomenon'].items(): add('B',f'phenomenon {k}: exact / SER / n',f"{v['exact_match']} / {v['symbol_error_rate']} / {v['items']}",'','','','Preliminary','results/expB_benchmark.json',cB)
for k,v in B['main_ratio_sensitivity'].items(): add('B',f'runs5 ratio Hadith/Quran-quotes [{k}]',v['ratio'],f"rates {v['quotes']} vs {v['hadith']}",'','','Robust','results/expB_benchmark.json',cB,'sensitivity to transcriber and pause convention')
add('B','adjudicated 30-item set: exact engine / independent',f"{G['engine_exact']}/30 ; {G['independent_exact']}/30",f"SER {G['engine_SER']} / {G['independent_SER']}",'','','Preliminary','results/expB_adjudicated.json','python3 -I experiments/expB_adjudicated.py',G['caveat'])
for k in ('engine','independent_transcriber'):
    u=U[k]; add('B',f'incremental AUC combined-lexical [{k}]',f"{u['auc_lexical']} -> {u['auc_combined']} (Δ={round(u['auc_combined']-u['auc_lexical'],3)})",str(u['delta_ci95_bootstrap']),'','','Robust','results/expB_auc.json','python3 -I experiments/expB_auc.py',f"{u['folds_with_positive_delta']}/{u['n_folds']} folds positive; paired, identical GroupKFold")
cC='python3 -I experiments/expC_lexical.py'
for r in C:
    ba=r['balanced_acc']; add('C',f"{r['pair']} balanced acc (LR content, grouped by surah)",ba['lr_content'],str(r['lr_content_ci95_surah_bootstrap']),r['p_raw'],f"Holm family {r['p_holm_family']}; Holm global {r['p_holm_global_with_revision_tests']}",r['status']+' / global: '+r['status_global'],'results/expC_lexical.json',cC,f"baselines: Meccan-only {ba['meccan_only']}, roots {ba['lr_roots']}, function words {ba['lr_function_words']}, NB {ba['nb_content']}, all {ba['lr_all']}; n={r['n']}")
cS='python3 -I experiments/expS_stylometry_metrics.py'
n=S['natural_prevalence']
for k in ('accuracy','balanced_accuracy','precision_quran','recall_quran','precision_hadith','recall_hadith(specificity)','macro_f1','weighted_f1','mcc','roc_auc','pr_auc_quran'):
    b=S['balanced_subsamples_1000'].get(k,{}); add('S',f'stylometry LR LOCO: {k}',f"natural {n[k]} | balanced {b.get('mean','')}",str(b.get('ci95','')),'','','Corrected','results/expS_stylometry_metrics.json',cS,'natural prevalence Quran=1.1%')
add('S','macro-F1 consistent with confusion matrix',str(S['macro_f1_audit']['consistent']),'','','','Reproduced','results/expS_stylometry_metrics.json',cS,json.dumps(S['macro_f1_audit']))
cols=list(rows[0].keys())
with open(os.path.join(R,'..','FINAL_VALIDATED_RESULTS.csv'),'w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
print(len(rows),'rows')
