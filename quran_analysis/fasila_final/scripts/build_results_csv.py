import os,json,csv
H=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(H,'..','results'); J=lambda f:json.load(open(os.path.join(R,f)))
F=J('fasila_results.json'); Wr=J('warsh_replication.json')
rows=[]
def add(sec,metric,num,den,point,ci,test,p,padj,unit,src,cmd,note=''):
    rows.append(dict(section=sec,metric=metric,numerator=num,denominator=den,point_estimate=point,ci95=ci,test=test,p_raw=p,p_holm=padj,unit=unit,source_path=src,command=cmd,note=note))
C='python3 -I scripts/fasila_pipeline.py'; S='fasila_final/results/fasila_results.json'
a=F['audit']; add('A audit','candidate pairs',a['candidates'],'','','','','','','pair',S,C,'3-word seed, 0.70<=sim<1.0, >=5 words')
add('A audit','cross-surah pairs',a['cross_surah'],a['candidates'],'','','','','','pair',S,C)
add('A audit','eligible pairs',a['eligible'],a['cross_surah'],'','','','','','pair',S,C,json.dumps(a['exclusions'],ensure_ascii=False))
P=F['primary']
add('B primary','decisive pairs favouring original (D1,k=3)',P['favour_own'],P['favour_own']+P['favour_swapped'],P['share_own_of_decisive'],P['share_own_ci95_component_bootstrap'],'exact sign test',P['sign_test_p_pair_iid'],F['definitions_grid']['D1_final_phoneme']['3']['p_holm_primary_family'],'pair (CI: component bootstrap 10000)',S,C,f"ties {P['ties']}; components {P['n_components']}")
add('B primary','component sign-flip randomization',P['favour_own']-P['favour_swapped'],'','','','sign-flip of whole components',P['component_signflip_p'],'','component',S,C,P['component_signflip_mode'])
add('B primary','verse own-ending matches local rhyme',P['verse_own_match'].split('/')[0],P['verse_own_match'].split('/')[1],P['verse_own_rate'],P['verse_own_ci95_component_bootstrap'],'','','','verse occurrence',S,C,f"Wilson iid {P['verse_own_wilson_iid']}")
add('B primary','verse swapped-ending matches local rhyme',P['verse_swapped_match'].split('/')[0],P['verse_swapped_match'].split('/')[1],P['verse_swapped_rate'],'','','','','verse occurrence',S,C)
add('B primary','own minus swapped match difference','','',round(P['verse_own_rate']-P['verse_swapped_rate'],3),P['own_minus_swapped_ci95'],'','','','verse occurrence (component bootstrap)',S,C)
O=F['one_pair_per_verse']; add('B dependence','one pair per verse: decisive favouring original',O['favour_own'],O['favour_own']+O['favour_swapped'],O['share_own_of_decisive'],O['share_own_ci95_component_bootstrap'],'exact sign test',O['sign_test_p_pair_iid'],'','pair',S,C,f"ties {O['ties']}; signflip p {O['component_signflip_p']}")
for d,g in F['definitions_grid'].items():
    for k,v in g.items(): add('E definitions',f'{d} k={k}',v['own'],v['own']+v['swapped'],round(v['own']/(v['own']+v['swapped']),3) if v['own']+v['swapped'] else '','','exact sign test',v['p'],v.get('p_holm_primary_family',''),'pair',S,C,f"ties {v['ties']}; tested {v['tested']}")
for n,v in F['null_models'].items():
    if n in('N3_pool_sizes',): continue
    if 'observed_own_match' in v: add('C null models',n,'','',v['observed_own_match'],f"null {v['null_mean']} {v['null_95']}",'Monte Carlo 10000',v['p_ge_obs'],'','distinct verse',S,C,v.get('matched_on','')+(f"; insufficient pool {v.get('verses_insufficient_pool')}" if 'verses_insufficient_pool' in v else ''))
    else: add('C null models',n,v['mean_pairs_favour_own'],'',v['share_own_mean'],v['share_own_95'],'Monte Carlo 5000','','','pair',S,C,'original vs random alternative ending')
for r in F['robustness_grid']: add('E extraction robustness',f"seq={r['shared_seq']} thr={r['threshold']} minw={r['min_words']}",r['own'],r['own']+r['swapped'],'','','exact sign test',r['p'],'','pair',S,C,f"candidates {r['candidates']}; eligible {r['eligible']}; ties {r['ties']}")
add('F Warsh','Warsh replication: decisive pairs favouring original (D1,k=3)',Wr['favour_own'],Wr['favour_own']+Wr['favour_swapped'],round(Wr['favour_own']/(Wr['favour_own']+Wr['favour_swapped']),3),'','exact sign test',Wr['sign_p'],'','pair','fasila_final/results/warsh_replication.json','python3 -I scripts/warsh_replication.py',f"{Wr['pairs_tested']} pairs tested; relations {Wr['alignment_relations_of_60_verses']}; KFGQPC UthmanicWarsh v3.0 sha256 {Wr['warsh_text']['sha256'][:12]}…; D1 only")
add('D expert review','grammatical/semantic alternatives (N4)','','','pending','','blinded expert review','','','pair','fasila_final/EXPERT_REVIEW_PACKAGE.csv','—','30 items prepared; exploratory model notes: 9 viable, 1 viable with caveat, 5 not substitutable (not validation)')
with open(os.path.join(H,'..','FASILA_FINAL_RESULTS.csv'),'w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
print(len(rows),'rows')
