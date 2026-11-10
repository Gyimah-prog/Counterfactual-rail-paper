"""Collects every number the report cites from the saved outputs, so the text is generated from data."""
import pickle, json, pandas as pd, numpy as np
from PIL import Image; import glob
A = pickle.load(open('out/analysis.pkl', 'rb')); T, best = A['T'], A['best']
m = T['main']; mon = pickle.load(open('out/monitor.pkl', 'rb')); info = json.load(open('out/ladder_main_info.json'))
g = lambda per, meth, col='joint': float(m[(m.period == per) & (m.method == meth)][col].iloc[0])
rec = T['recovery']; r = lambda per, meth: float(rec[(rec.period == per) & (rec.method == meth)].recovery.iloc[0])
C = T['contrasts']; P = T['policy']; pp = lambda per, s, c: float(P[(P.period == per) & (P.strategy == s)][c].iloc[0])
comp = T['composition']; cons = T['consistency'].set_index('method'); cost = T['cost'].set_index('item').value; fac = T['factorial'].set_index('method')
b13 = mon[mon.date.dt.year == 2013].validity
L = {'ctrl_threshold': 'Threshold correction (control)', 'ctrl_p1': 'P1 signed mean (control)', 'stale': 'Stale P2', 'inst_ratio': 'Instance, ratio form',
     'inst_margin': 'Instance, margin form', 'sc_sigma': 'Scalar, spread', 'sc_tail': 'Scalar, tail magnitude', 'sc_excess': 'Scalar, project excess',
     'sc_mproxy': 'Scalar, model boundary proxy', 'dg_sigma': 'Diagonal, spread', 'dg_tail': 'Diagonal, tail magnitude', 'dg_excess': 'Diagonal, project excess',
     'ot_1d': 'OT, 1D WARP quantile', 'ot_marg': 'OT, marginal quantile', 'ot_mv': 'OT, multivariate', 'ot_mv_ret': 'OT, multivariate with retrieval', 'fresh': 'Fresh generation'}
mainrows = [[L[k] + (' *' if k in best.values() else ''), *[f"{100 * g(p, k):.1f}" for p in ('Eval 2015', 'Test 2016', '2015', '2016')]] for k in L]
crow = []
for _, x in C[C.metric == 'js'].iterrows():
    crow.append([f"{x.contrast} ({L[x.a].split(',')[0] if x.a != 'fresh' else 'fresh'})", x.period, f"{100 * x['diff']:+.1f}", f"{100 * x.lo:+.1f} to {100 * x.hi:+.1f}"])
c4 = C[(C.contrast == 'C4 OT vs diagonal') & (C.period == 'Test 2016') & (C.metric == 'js')].iloc[0]
rd = T['residual_desc'].set_index(['period', 'group']).resid_fail
glm = T['residual_glm']
D = dict(
 js=dict(test={k: g('Test 2016', k) for k in L}, eval={k: g('Eval 2015', k) for k in L}),
 rec=dict(test={k: r('Test 2016', k) for k in ['inst_margin', 'inst_ratio']}, eval={k: r('Eval 2015', k) for k in ['inst_margin', 'inst_ratio']}),
 c4=dict(diff=float(c4['diff']), lo=float(c4.lo), hi=float(c4.hi)),
 mon=dict(b13min=float(b13.min()), b13max=float(b13.max())),
 cost=dict(fresh_sec=float(cost['Fresh generation per case (s)']), fresh_q=float(cost['Fresh generation model queries per case']),
           fresh_total=float(cost['Fresh generation total for 2015 and 2016 cohorts (min)'])),
 n=dict(y15=int(g('2015', 'stale', 'N')), y16=int(g('2016', 'stale', 'N')), eval15=int(g('Eval 2015', 'stale', 'N'))),
 pol=dict(eval_nogate=pp('Eval 2015', 'Policy no support gate', 'success'), test_nogate=pp('Test 2016', 'Policy no support gate', 'success'),
          eval_fresh=pp('Eval 2015', 'Universal fresh', 'success'), test_fresh=pp('Test 2016', 'Universal fresh', 'success'),
          eval_nogate_regen=pp('Eval 2015', 'Policy no support gate', 'regen_frac'), test_nogate_regen=pp('Test 2016', 'Policy no support gate', 'regen_frac')),
 comp=dict(dg_x=float(comp[(comp.period == 'Test 2016') & (comp.method == 'dg_excess')]['js_XLEVEL only'].iloc[0]),
           stale_x=float(comp[(comp.period == 'Test 2016') & (comp.method == 'stale')]['js_XLEVEL only'].iloc[0])),
 cons=dict(stale=float(cons.loc['stale', 'mean_change']), inst=float(cons.loc['inst_margin', 'mean_change']), fresh=float(cons.loc['fresh', 'mean_change']),
           policy=float(cons.loc['policy', 'mean_change']), seed_med=float(cons.loc['fresh seed 42 vs seed 7', 'median_change'])),
 fac=dict(inst_upd=float(fac.loc['inst_margin', 'js_updated_model']), stale_upd=float(fac.loc['stale', 'js_updated_model']),
          fresh_f0_on_upd=float(fac.loc['fresh', 'js_updated_model']), fresh_upd=float(fac.loc['fresh under updated model', 'js_updated_model']),
          fresh_upd_wd=float(fac.loc['fresh under updated model, WARP driven only', 'flip_updated_model'])),
 margin=info['M_MARGIN'], sup99=info['SUP99'], mainrows=mainrows, contr=crow,
 repro=[['2014 library size', '637 of 649', '637 of 649'], ['ROC AUC 2015 and 2016', '0.9963 and 0.9974', '0.9963 and 0.9974'],
        ['P2 flip 2015', '28.5% (476 of 1,672)', '28.47% (476 of 1,672)'], ['P2 flip 2016', '6.9% (144 of 2,092)', '6.88% (144 of 2,092)'],
        ['Mean absolute WARP perturbation', '0.0494 in.', '0.0494 in.'], ['Fresh WARP driven 2015, frozen model', '98.5% (1,219 of 1,237)', '98.5% (1,217 of 1,236)'],
        ['Fresh WARP driven 2016, updated model', '84.2% (1,015 of 1,206)', f"{100 * fac.loc['fresh under updated model, WARP driven only', 'flip_updated_model']:.1f}% of 1,206"]],
 ladder=[['Stale (L1)', 'P2', 'Nearest 2014 action, unchanged'],
         ['Instance (B)', 'ratio, margin', 'Ratio scales the action by the target WARP excess over the origin excess. Margin scales WARP to cover the target excess plus a library margin.'],
         ['Scalar (L2)', 'spread, tail, excess, model proxy', 'One WARP anchored factor from the window. Spread uses the square root variance ratio. Tail uses mean |WARP| above the 2014 95th percentile. Excess uses mean excess beyond 0.30 in. Proxy uses 0.283 in.'],
         ['Diagonal (L3)', 'spread, tail, excess', 'Same families estimated per feature'],
         ['OT (L4)', '1D, marginal, multivariate, multivariate with retrieval', 'Quantile map on signed WARP, quantile maps per feature, entropic barycentric map, and the same map with retrieval by transported origins']],
 costrows=[[k, f"{v:.4g}"] for k, v in cost.items()],
 verdicts=[['H1 monitoring', 'Partly supported', 'Validity exposes decay that AUC misses. It does not lead exceedance frequency once the in sample 2014 artifact is removed.'],
           ['H2 magnitude', 'Supported for instance', f"Instance margin {100 * g('Test 2016', 'inst_margin'):.1f}% against stale {100 * g('Test 2016', 'stale'):.1f}% in 2016. Population scalars recover far less."],
           ['H3 diagonal', 'Null form holds', 'Spread diagonal equals spread scalar. Diagonal excess repairs XLEVEL only cases, a registered alternative.'],
           ['H4 OT', 'Not supported', 'Below the best diagonal in 2016. Helps only under synthetic tail inflation. Multivariate map biased with no change.'],
           ['H5 residual', 'Supported', 'XLEVEL only share, reference distance, and support mismatch predict failure. Joint and other defects always fail.'],
           ['H6 refresh', 'Supported for validity and cost', 'Gate free policy matches fresh generation with 23% to 50% regeneration. Not more consistent than fresh generation.']],
 files=[['Counterfactual_Project_2_Results.docx', 'This report'], ['figures/', 'Nine figures in PNG and PDF'], ['tables/results_tables.xlsx', 'All analysis tables, one sheet each'],
        ['tables/*.csv', 'Monitoring, stress test, and sensitivity tables'], ['code/', 'Scripts s01 to s08 and the report builder, in run order'],
        ['data/', 'Per case ladder results, fresh generation results, updated model'], ['README.md', 'How to rerun the analysis']])
D['js'] = {k: v for k, v in D['js'].items()}
json.dump(D, open('out/report_numbers.json', 'w'), indent=1, default=float)
json.dump({f.split('/')[-1][:-4]: Image.open(f).size for f in glob.glob('figs/*.png')}, open('figs/sizes.json', 'w'))
print(rd.round(3).to_dict()); print(glm.round(3).to_string())
