"""Matched analysis of the ladder against fresh generation. Writes every table used in the report."""
import numpy as np, pandas as pd, pickle, json, glob
import statsmodels.api as sm
from sklearn.metrics import pairwise_distances
from cf2lib import *
df, model, lib = load(); madv = mad(df); B = bounds(df)
X0 = lib[[f'x0_{f}' for f in F]].values; D0 = lib[[f'd_{f}' for f in F]].values
info = json.load(open('out/ladder_main_info.json'))
lad = pickle.load(open('out/ladder_main.pkl', 'rb'))
fr = pd.concat([pickle.load(open(f, 'rb')) for f in sorted(glob.glob('out/fresh_[0-9].pkl'))], ignore_index=True)
T = {}  # output tables
lad['date'] = pd.to_datetime(lad.date)
def period(d):
    if d.year == 2016: return 'Test 2016'
    return 'Dev 2015' if d.month <= 6 else 'Eval 2015'
lad['period'] = lad.date.map(period); lad['fvf'] = lad.vf & lad.vb
lad['block'] = lad.MILE.astype(str) + '_' + (lad.FEET // 200).astype(str)

# ---------- fresh rows on the same cases ----------
f0 = fr[(fr.model == 'f0') & (fr.seed == SEED)].set_index('rid')
base = lad[lad.method == 'stale'].copy()
Xt = df.set_index('rid').loc[base.rid, F].values
ok = f0.loc[base.rid, 'ok'].values
Zf = np.array([z if o else x for z, o, x in zip(f0.loc[base.rid, 'z'].values, ok, Xt)])
vf, vp, vb, p1 = validate(model, Xt, Zf, B); vf &= ok
fresh = base.copy(); fresh['method'] = 'fresh'; fresh['vf'] = vf; fresh['vp'] = vp; fresh['vb'] = vb
fresh['js'] = vf & vp & vb; fresh['fvf'] = vf & vb; fresh['cost'] = (np.abs(Zf - Xt) / madv).sum(1)
fresh['sec'] = f0.loc[base.rid, 'sec'].values; fresh['queries'] = f0.loc[base.rid, 'queries'].values
for j, f in enumerate(F): fresh[f'z_{f}'] = Zf[:, j]
lad = pd.concat([lad, fresh], ignore_index=True)
assert set(fresh.rid) == set(base.rid)

# ---------- development selection ----------
fam = {'Instance': ['inst_ratio', 'inst_margin'], 'Scalar': ['sc_sigma', 'sc_tail', 'sc_excess', 'sc_mproxy'],
       'Diagonal': ['dg_sigma', 'dg_tail', 'dg_excess'], 'OT': ['ot_1d', 'ot_marg', 'ot_mv', 'ot_mv_ret']}
dev = lad[lad.period == 'Dev 2015'].groupby('method').js.mean()
best = {k: dev[v].idxmax() for k, v in fam.items()}
T['dev_selection'] = pd.DataFrame([dict(level=k, method=m, dev_js=dev[m]) for k, v in fam.items() for m in v])
print('best by development JS', best)

# ---------- main comparison table ----------
order = ['ctrl_none', 'ctrl_threshold', 'ctrl_p1', 'stale'] + sum(fam.values(), []) + ['fresh']
g = lad.groupby(['period', 'method'])
main = g.agg(N=('js', 'size'), flip=('vf', 'mean'), feasible_flip=('fvf', 'mean'), project=('vp', 'mean'),
             joint=('js', 'mean'), cost_med=('cost', 'median'), clipped=('clipped', 'mean'),
             overshoot=('overshoot', 'mean')).reset_index()
yr = lad.assign(period=lad.date.dt.year.astype(str)).groupby(['period', 'method']).agg(
    N=('js', 'size'), flip=('vf', 'mean'), feasible_flip=('fvf', 'mean'), project=('vp', 'mean'),
    joint=('js', 'mean'), cost_med=('cost', 'median'), clipped=('clipped', 'mean'), overshoot=('overshoot', 'mean')).reset_index()
T['main'] = pd.concat([main, yr]); T['main']['order'] = T['main'].method.map({m: i for i, m in enumerate(order)})
T['main'] = T['main'].sort_values(['period', 'order']).drop(columns='order')

# ---------- paired block bootstrap ----------
def paired(a, b, per, metric='js', blk=200, nb=2000, seed=SEED):
    A = lad[(lad.method == a) & (lad.period.isin(per) if isinstance(per, list) else lad.date.dt.year.astype(str).isin([per]))]
    Bm = lad[(lad.method == b)]
    m = A[['rid', metric, 'MILE', 'FEET']].merge(Bm[['rid', metric]], on='rid', suffixes=('_a', '_b'))
    m['d'] = m[f'{metric}_a'].astype(float) - m[f'{metric}_b'].astype(float)
    m['blk'] = m.MILE.astype(str) + '_' + (m.FEET // blk).astype(str)
    s = m.groupby('blk').d.agg(['sum', 'size']); r = np.random.default_rng(seed); k = len(s); boot = []
    for _ in range(nb):
        i = r.integers(0, k, k); boot.append(s['sum'].values[i].sum() / s['size'].values[i].sum())
    return dict(diff=m.d.mean(), lo=np.quantile(boot, .025), hi=np.quantile(boot, .975), n=len(m), blocks=k)
contr = [('C1 instance vs stale', best['Instance'], 'stale'), ('C2 scalar vs instance', best['Scalar'], best['Instance']),
         ('C3 diagonal vs scalar', best['Diagonal'], best['Scalar']), ('C4 OT vs diagonal', best['OT'], best['Diagonal']),
         ('S1 OT vs stale', best['OT'], 'stale'), ('S2 fresh vs best instance', 'fresh', best['Instance']),
         ('S3 diagonal vs instance', best['Diagonal'], best['Instance'])]
rows = []
for per in (['Eval 2015'], ['Test 2016']):
    for name, a, b in contr:
        for metric in ('js', 'fvf'):
            r = paired(a, b, per, metric); rows.append(dict(period=per[0], contrast=name, a=a, b=b, metric=metric, **r))
T['contrasts'] = pd.DataFrame(rows)
# block size sensitivity and leave one inspection out for C1 and C4 in 2016
rows = []
for blk in (100, 200, 500):
    for name, a, b in contr[:4]:
        r = paired(a, b, ['Test 2016'], 'js', blk=blk); rows.append(dict(contrast=name, block_ft=blk, **r))
T['block_sensitivity'] = pd.DataFrame(rows)
rows = []
for name, a, b in contr[:4]:
    for per in ('Eval 2015', 'Test 2016'):
        ds = []
        for d in sorted(lad[lad.period == per].date.unique()):
            A = lad[(lad.method == a) & (lad.period == per) & (lad.date != d)]; Bm = lad[(lad.method == b) & (lad.period == per) & (lad.date != d)]
            ds.append(A.js.mean() - Bm.js.mean())
        rows.append(dict(contrast=name, period=per, n_inspections=len(ds), loio_min=min(ds), loio_max=max(ds)))
T['loio'] = pd.DataFrame(rows)

# ---------- recovery ----------
rows = []
for per in ('Eval 2015', 'Test 2016'):
    sub = lad[lad.period == per].groupby('method').fvf.mean(); den = sub['fresh'] - sub['stale']
    for m in order[3:-1]:
        rows.append(dict(period=per, method=m, fr=sub[m], fr_stale=sub['stale'], fr_fresh=sub['fresh'],
                         recovery=(sub[m] - sub['stale']) / den if den > 0.05 else np.nan))
T['recovery'] = pd.DataFrame(rows)

# ---------- composition ----------
pi0 = pd.Series({'WARP only': 511 / 649, 'XLEVEL only': 138 / 649})
rows = []
for per in ('Eval 2015', 'Test 2016'):
    s = lad[lad.period == per]
    for m in order[3:]:
        sm_ = s[s.method == m]; byg = sm_.groupby('group').js.mean(); share = sm_.group.value_counts(normalize=True)
        rows.append(dict(period=per, method=m, raw=sm_.js.mean(), standardized=float((pi0 * byg.reindex(pi0.index)).sum()),
                         **{f'js_{k}': byg.get(k, np.nan) for k in ['WARP only', 'XLEVEL only', 'WARP and XLEVEL', 'Other']},
                         **{f'share_{k}': share.get(k, 0) for k in ['WARP only', 'XLEVEL only', 'WARP and XLEVEL', 'Other']}))
T['composition'] = pd.DataFrame(rows)

# ---------- residual failure after all ladder levels ----------
ladder_methods = [best['Instance'], best['Scalar'], best['Diagonal'], best['OT']]
anyok = lad[lad.method.isin(ladder_methods)].groupby('rid').js.max()
cs = lad[lad.method == 'stale'].set_index('rid').copy(); cs['resid_fail'] = ~anyok.loc[cs.index].values
cs = cs[cs.period != 'Dev 2015']
Xd = pd.DataFrame({'xlevel_only': (cs.group == 'XLEVEL only').astype(float), 'joint': (cs.group == 'WARP and XLEVEL').astype(float),
                   'other': (cs.group == 'Other').astype(float), 'warp_excess': cs.eW, 'log_ref_dist': np.log(cs.nn_dist + 1e-3),
                   'support_mismatch': cs.mismatch.astype(float), 'year2016': (cs.date.dt.year == 2016).astype(float)})
glm = sm.GLM(cs.resid_fail.astype(float), sm.add_constant(Xd), family=sm.families.Binomial())
# perfect separation for joint and other groups (always fail) is handled by reporting them descriptively
Xd2 = Xd.drop(columns=['joint', 'other']); keep = (cs.group.isin(['WARP only', 'XLEVEL only'])).values
fit = sm.GLM(cs.resid_fail[keep].astype(float), sm.add_constant(Xd2[keep]), family=sm.families.Binomial()).fit(
    cov_type='cluster', cov_kwds={'groups': pd.factorize(cs.block[keep])[0]})
T['residual_glm'] = pd.DataFrame({'coef': fit.params, 'se': fit.bse, 'p': fit.pvalues, 'odds_ratio': np.exp(fit.params)})
T['residual_desc'] = cs.groupby(['period', 'group']).agg(n=('resid_fail', 'size'), resid_fail=('resid_fail', 'mean'),
                                                         mismatch=('mismatch', 'mean'), ref_dist_med=('nn_dist', 'median')).reset_index()
cs['dist_bin'] = pd.qcut(cs.nn_dist, 5, labels=[f'Q{i}' for i in range(1, 6)])
T['residual_heat'] = cs.groupby(['group', 'dist_bin'], observed=False).agg(fail=('resid_fail', 'mean'), n=('resid_fail', 'size'),
                                                                            mismatch=('mismatch', 'mean')).reset_index()
T['residual_dist_edges'] = pd.DataFrame({'edge': pd.qcut(cs.nn_dist, 5, retbins=True)[1]})

# ---------- refresh policy ----------
SUP99 = info['SUP99']
piv = lad[lad.method.isin(ladder_methods + ['fresh', 'stale'])].pivot_table(index='rid', columns='method', values='js', aggfunc='first')
tv = lad[lad.method.isin(ladder_methods)].groupby('rid').t_validate.mean()
meta = lad[lad.method == 'fresh'].set_index('rid')[['period', 'sec', 'queries', 'nn_dist', 'group', 'date']]
def policy(levels, gate=True):
    out = []
    for rid, r in piv.iterrows():
        m = meta.loc[rid]; tried = 0; res = None
        if not (gate and m.nn_dist > SUP99):
            for lv in levels:
                tried += 1
                if r[lv]: res = lv; break
        if res is None:
            res = 'fresh' if r['fresh'] else 'unresolved'; regen = True
        else: regen = False
        out.append(dict(rid=rid, period=m.period, outcome=res, regen=regen, success=res != 'unresolved',
                        sec=tried * tv.get(rid, 0) + (m.sec if regen else 0), queries=2 * tried + (m.queries if regen else 0)))
    return pd.DataFrame(out)
pols = {'Policy full ladder': policy(ladder_methods), 'Policy without OT': policy(ladder_methods[:3]),
        'Policy instance only': policy(ladder_methods[:1]), 'Policy no support gate': policy(ladder_methods, gate=False),
        'Policy no gate without OT': policy(ladder_methods[:3], gate=False)}
rows = []
for per in ('Eval 2015', 'Test 2016'):
    st = lad[(lad.method == 'stale') & (lad.period == per)]; fsub = lad[(lad.method == 'fresh') & (lad.period == per)]
    rows.append(dict(period=per, strategy='Stale reuse', success=st.js.mean(), regen_frac=0, unresolved=1 - st.js.mean(),
                     sec_per_case=st.t_validate.mean(), queries_per_case=2))
    rows.append(dict(period=per, strategy='Universal fresh', success=fsub.js.mean(), regen_frac=1, unresolved=1 - fsub.js.mean(),
                     sec_per_case=fsub.sec.mean(), queries_per_case=fsub.queries.mean()))
    for k, p in pols.items():
        s = p[p.period == per]
        rows.append(dict(period=per, strategy=k, success=s.success.mean(), regen_frac=s.regen.mean(), unresolved=1 - s.success.mean(),
                         sec_per_case=s.sec.mean(), queries_per_case=s.queries.mean(),
                         **{f'by_{o}': (s.outcome == o).mean() for o in ladder_methods + ['fresh']}))
T['policy'] = pd.DataFrame(rows)
pol = pols['Policy full ladder']

# ---------- consistency ----------
def actions(m):
    s = lad[lad.method == m].set_index('rid'); Xs = df.set_index('rid').loc[s.index, F].values
    return pd.DataFrame((s[[f'z_{f}' for f in F]].values - Xs) / madv, index=s.index)
loc = df.set_index('rid')[['MILE', 'FEET', 'date']]
elig_rids = lad[lad.method == 'stale'].rid.unique(); L = loc.loc[elig_rids].reset_index().sort_values(['MILE', 'FEET', 'date'])
L['prev'] = L.groupby(['MILE', 'FEET']).rid.shift(); pairs_ = L.dropna(subset=['prev'])
dlist = sorted(df.date.unique()); dpos = {d: i for i, d in enumerate(dlist)}
pairs_ = pairs_[[dpos[d] - dpos[dp] == 1 for d, dp in zip(pairs_.date, loc.loc[pairs_.prev.astype(int), 'date'])]]
act_pol = {}
pol_idx = pol.set_index('rid')
rows = []
for m in ['stale', best['Instance'], best['Diagonal'], 'fresh', 'policy']:
    if m == 'policy':
        A = pd.concat([actions(o).loc[pol_idx.index[pol_idx.outcome == o]] for o in pol_idx.outcome.unique() if o != 'unresolved'])
    else: A = actions(m)
    a = pairs_[pairs_.rid.isin(A.index) & pairs_.prev.astype(int).isin(A.index)]
    d = np.abs(A.loc[a.rid].values - A.loc[a.prev.astype(int)].values).sum(1)
    rows.append(dict(method=m, n_pairs=len(a), mean_change=d.mean(), median_change=np.median(d), share_change_gt1=(d > 1).mean()))
rep = fr[(fr.model == 'f0') & (fr.seed == 7)].set_index('rid')
z42 = f0.loc[rep.index]; both = rep.ok & z42.ok
dz = np.array([np.abs((a - b) / madv).sum() for a, b in zip(rep.z[both], z42.z[both])])
rows.append(dict(method='fresh seed 42 vs seed 7', n_pairs=int(both.sum()), mean_change=dz.mean(), median_change=np.median(dz),
                 share_change_gt1=(dz > 1).mean(), success_disagreement=float((rep.ok != z42.ok).mean())))
T['consistency'] = pd.DataFrame(rows)

# ---------- cost ----------
fi = info['finfo']; tms = pd.DataFrame({d: v['times'] for d, v in fi.items()}).T
f0all = fr[(fr.model == 'f0') & (fr.seed == SEED)]
T['cost'] = pd.DataFrame([
    dict(item='Retrieval per inspection (s)', value=tms.retrieval.mean()),
    dict(item='Factor estimation per inspection, all families (s)', value=tms.factors.mean()),
    dict(item='1D quantile map per inspection (s)', value=tms.ot_1d.mean()),
    dict(item='Marginal quantile maps per inspection (s)', value=tms.ot_marg.mean()),
    dict(item='Multivariate OT fit per inspection (s)', value=tms.ot_mv_fit.mean()),
    dict(item='Transfer and validation per case (ms)', value=1000 * lad[lad.method != 'fresh'].t_validate.mean()),
    dict(item='Fresh generation per case (s)', value=f0all.sec.mean()),
    dict(item='Fresh generation per case, 95th percentile (s)', value=f0all.sec.quantile(.95)),
    dict(item='Fresh generation model queries per case', value=f0all.queries.mean()),
    dict(item='Fresh generation total for 2015 and 2016 cohorts (min)', value=f0all.sec.sum() / 60),
    dict(item='Fresh generation success rate', value=f0all.ok.mean()),
])

# ---------- factorial with the updated 2013 to 2015 model (2016) ----------
upd = pickle.load(open('out/model_updated_2013_2015.pkl', 'rb'))
s16 = lad[(lad.date.dt.year == 2016)]; rid16 = s16.rid.unique(); X16 = df.set_index('rid').loc[rid16, F].values
eu = upd.predict(X16) == 1; rid_u = set(np.array(rid16)[eu])
fu = fr[(fr.model == 'upd')].set_index('rid')
rows = []
for m in ['stale', best['Instance'], best['Scalar'], best['Diagonal'], best['OT'], 'fresh']:
    s = s16[(s16.method == m) & s16.rid.isin(rid_u)]
    Xs = df.set_index('rid').loc[s.rid, F].values; Zs = s[[f'z_{f}' for f in F]].values
    vfu, vpu, vbu, _ = validate(upd, Xs, Zs, B)
    rows.append(dict(method=m, N=len(s), js_f0=s.js.mean(), js_updated_model=(vfu & vpu & vbu).mean()))
s = fu.loc[[r for r in fu.index if r in rid_u]]; Xs = df.set_index('rid').loc[s.index, F].values
Zs = np.array([z if o else x for z, o, x in zip(s.z, s.ok, Xs)]); vfu, vpu, vbu, _ = validate(upd, Xs, Zs, B); vfu &= s.ok.values
rows.append(dict(method='fresh under updated model', N=len(s), js_f0=np.nan, js_updated_model=(vfu & vpu & vbu).mean()))
wd = (np.abs(Xs[:, W]) > 0.30)
rows.append(dict(method='fresh under updated model, WARP driven only', N=int(wd.sum()), js_f0=np.nan,
                 js_updated_model=(vfu & vpu & vbu)[wd].mean(), flip_updated_model=vfu[wd].mean()))
T['factorial'] = pd.DataFrame(rows)

# ---------- 2013 backcast (out of library, before the evaluation period) ----------
rows = []
s13 = df[(df.year == 2013) & (df.defect == 1)]; X13 = s13[F].values; X13 = X13[model.predict(X13) == 1]
dist = pairwise_distances(X13 / madv, X0 / madv); nn = dist.argmin(1); Dn = D0[nn]
eT = np.maximum(np.abs(X13[:, W]) - .3, 0); eO = np.maximum(np.abs(X0[nn, W]) - .3, 0)
dW = np.maximum(np.abs(Dn[:, W]), 1e-4)
for m, D in {'stale': Dn, 'inst_ratio': Dn * np.clip(np.where(eT > 0, (eT + .005) / (eO + .005), 1), .25, 8)[:, None],
             'inst_margin': Dn * np.clip(np.where(eT > 0, (eT + info['M_MARGIN']) / dW, 1), .25, 8)[:, None]}.items():
    Z = apply_action(X13, D); vf, vp, vb, _ = validate(model, X13, Z, B); g = group(X13)
    rows.append(dict(method=m, N=len(X13), feasible_flip=(vf & vb).mean(), joint=(vf & vp & vb).mean(),
                     joint_warp_only=(vf & vp & vb)[g == 'WARP only'].mean(), share_other=(g == 'Other').mean()))
T['backcast_2013'] = pd.DataFrame(rows)

pickle.dump(dict(T=T, best=best, lad=lad, pol=pol, pols=pols), open('out/analysis.pkl', 'wb'))
with pd.ExcelWriter('out/results_tables.xlsx') as xw:
    for k, v in T.items(): v.to_excel(xw, sheet_name=k[:31], index=k == 'residual_glm')
for k, v in T.items():
    print('\n=====', k); print(v.round(4).to_string(max_rows=80))
