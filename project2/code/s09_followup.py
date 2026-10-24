"""Follow up analyses.
A  Recompute affected results with training only bounds (2013 to 2014 for f0, 2013 to 2015 for the updated model).
B  Compare instance margin adaptation with library free corrections that use the same margin.
C  Test whether reference distance and support mismatch add information beyond feature excess and defect group.
Run with argument A, B, C or ALL. Part A needs out/fresh_tr_*.pkl."""
import sys, glob, json, pickle, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from cf2lib import *
PARTS = sys.argv[1] if len(sys.argv) > 1 else 'ALL'
df, model, lib = load(); madv = mad(df)
def bnd(d): return np.c_[np.maximum(FRA_LO, d[F].min().values), np.minimum(FRA_HI, d[F].max().values)]
B14 = bnd(df[df.year.isin([2013, 2014])]); B15 = bnd(df[df.year.isin([2013, 2014, 2015])])
A = pickle.load(open('out/analysis.pkl', 'rb')); lad, best = A['lad'], A['best']
info = json.load(open('out/ladder_main_info.json')); M_W = info['M_MARGIN']
X0 = lib[[f'x0_{f}' for f in F]].values; XC = lib[[f'cf_{f}' for f in F]].values; D0 = XC - X0
Zc = [f'z_{f}' for f in F]
def inB(Z, Bm): return ((Z >= Bm[:, 0] - 1e-12) & (Z <= Bm[:, 1] + 1e-12)).all(1)
# rescore every library based method with training only feasibility
lad = lad[lad.method != 'fresh'].copy()
lad['vb'] = inB(lad[Zc].values, B14); lad['js'] = lad.vf & lad.vp & lad.vb; lad['fvf'] = lad.vf & lad.vb
base = lad[lad.method == 'stale'].copy(); Xt = df.set_index('rid').loc[base.rid, F].values
def block_ci(d, blk, nb=2000):
    s = pd.DataFrame({'d': d, 'b': blk}).groupby('b').d.agg(['sum', 'size']); r = np.random.default_rng(SEED); k = len(s); out = []
    for _ in range(nb):
        i = r.integers(0, k, k); out.append(s['sum'].values[i].sum() / s['size'].values[i].sum())
    return np.mean(d), np.quantile(out, .025), np.quantile(out, .975), k
base['block'] = base.MILE.astype(str) + '_' + (base.FEET // 200).astype(str)
R = {}

# ---------------- B  library free corrections ----------------
# per feature margins: WARP from the library (median overshoot beyond the project line, WARP origins),
# XLEVEL from XLEVEL only origins in the same way, GAGE and alignment scaled from WARP by MAD (no library examples)
xo = (np.abs(X0[:, X]) > 0.80) & (np.abs(X0[:, W]) <= 0.30)
M_X = float(np.median(np.abs(D0[xo, X]) - (np.abs(X0[xo, X]) - 0.80)))
M = M_W * madv / madv[W]; M[X] = M_X
def free(Xt, feats, marg):
    Z = Xt.copy()
    for j in feats:
        hi = Z[:, j] > PROJ_HI[j]; lo = Z[:, j] < PROJ_LO[j]
        Z[hi, j] = PROJ_HI[j] - marg[j]; Z[lo, j] = PROJ_LO[j] + marg[j]
    return Z
def score(Z, Xs, Bm=B14):
    vf, vp, vb, _ = validate(model, Xs, Z, Bm); vb = inB(Z, Bm)
    return vf, vp, vb, vf & vp & vb, (np.abs(Z - Xs) / madv).sum(1)
if PARTS in ('B', 'ALL'):
    meth = {'free_warp_margin': free(Xt, [W], M), 'free_all_margin': free(Xt, range(5), M)}
    rows = []
    for nm, Z in meth.items():
        vf, vp, vb, js, cost = score(Z, Xt)
        b2 = base.assign(method=nm, vf=vf, vp=vp, vb=vb, js=js, cost=cost)
        for j, f in enumerate(F): b2[f'z_{f}'] = Z[:, j]
        rows.append(b2)
    cmp_ = pd.concat([lad[lad.method.isin(['stale', 'inst_margin', 'dg_excess'])]] + rows, ignore_index=True)
    T = cmp_.groupby(['period', 'method']).agg(N=('js', 'size'), joint=('js', 'mean'), flip=('vf', 'mean'), project=('vp', 'mean'),
                                             feasible=('vb', 'mean'), cost_med=('cost', 'median')).reset_index()
    TG = cmp_[cmp_.period != 'Dev 2015'].groupby(['period', 'group', 'method']).js.mean().unstack('method').reset_index()
    # paired contrasts
    pc = []
    for per in ('Eval 2015', 'Test 2016'):
        for a, b, grp in [('free_warp_margin', 'inst_margin', 'WARP only'), ('free_warp_margin', 'inst_margin', None),
                          ('free_all_margin', 'inst_margin', None), ('free_all_margin', 'dg_excess', 'XLEVEL only')]:
            sa = cmp_[(cmp_.method == a) & (cmp_.period == per)].set_index('rid'); sb = cmp_[(cmp_.method == b) & (cmp_.period == per)].set_index('rid')
            if grp: sa = sa[sa.group == grp]; sb = sb.loc[sa.index]
            d = sa.js.astype(float).values - sb.loc[sa.index].js.astype(float).values
            mu, lo, hi, k = block_ci(d, base.set_index('rid').loc[sa.index, 'block'].values)
            agree = (sa.js.values == sb.loc[sa.index].js.values).mean()
            pc.append(dict(period=per, contrast=f'{a} vs {b}', subset=grp or 'all', n=len(d), diff=mu, lo=lo, hi=hi, blocks=k, case_agreement=agree))
    # how different are the actions themselves on WARP only cases
    s1 = cmp_[(cmp_.method == 'free_warp_margin') & (cmp_.group == 'WARP only')].set_index('rid')
    s2 = cmp_[(cmp_.method == 'inst_margin') & (cmp_.group == 'WARP only')].set_index('rid').loc[s1.index]
    dz = np.abs(s1[Zc].values - s2[Zc].values) / madv
    act = dict(n=len(s1), median_action_diff_mad=float(np.median(dz.sum(1))), share_warp_component_equal=float((dz[:, W] < 1e-6).mean()),
               share_nonwarp_diff_gt_0p1=float((dz[:, [0, 1, 2, 4]].sum(1) > 0.1).mean()))
    # margin sweep for the WARP only library free rule
    sw = []
    for m in [0.0, 0.005, 0.01, M_W, 0.02, 0.03, 0.05]:
        mm = M.copy(); mm[W] = m; Z = free(Xt, [W], mm); vf, vp, vb, js, _ = score(Z, Xt)
        wo = base.group.values == 'WARP only'
        for per in ('Eval 2015', 'Test 2016'):
            p = base.period.values == per; sw.append(dict(margin=m, period=per, joint_warp_only=js[p & wo].mean(), joint_all=js[p].mean()))
    R['B_table'] = T; R['B_by_group'] = TG; R['B_contrasts'] = pd.DataFrame(pc); R['B_action_diff'] = pd.DataFrame([act]); R['B_margin_sweep'] = pd.DataFrame(sw)
    R['B_margins'] = pd.DataFrame({'feature': F, 'margin_in': M, 'source': ['WARP scaled by MAD', 'WARP scaled by MAD', 'WARP scaled by MAD', 'library WARP origins', 'library XLEVEL only origins']})
    pickle.dump(cmp_, open('out/followup_B_cases.pkl', 'wb'))

# ---------------- C  diagnosis value ----------------
if PARTS in ('C', 'ALL'):
    lv = [best['Instance'], best['Scalar'], best['Diagonal'], best['OT']]
    anyok = lad[lad.method.isin(lv)].groupby('rid').js.max()
    instok = lad[lad.method == 'inst_margin'].set_index('rid').js
    c = base[base.period != 'Dev 2015'].set_index('rid').copy()
    c['fail_ladder'] = ~anyok.loc[c.index].values; c['fail_inst'] = ~instok.loc[c.index].values
    E = excess(df.set_index('rid').loc[c.index, F].values)
    feats_base = pd.DataFrame({'xlevel_only': (c.group == 'XLEVEL only').astype(float).values, **{f'exc_{f}': E[:, j] for j, f in enumerate(F)},
                               'year2016': (c.date.dt.year == 2016).astype(float).values}, index=c.index)
    feats_aug = feats_base.assign(log_ref_dist=np.log(c.nn_dist.values + 1e-3), mismatch=c.mismatch.astype(float).values)
    keep = c.group.isin(['WARP only', 'XLEVEL only']).values
    rows = []
    for outc in ('fail_ladder', 'fail_inst'):
        y = c[outc].values.astype(int)
        for name, Xf in [('base: group and feature excess', feats_base), ('augmented: plus reference distance and mismatch', feats_aug),
                         ('distance and mismatch only', feats_aug[['log_ref_dist', 'mismatch', 'year2016']])]:
            Xa = Xf.values[keep]; ya = y[keep]; g = c.block.values[keep]
            pred = np.zeros(len(ya))
            for tr, te in GroupKFold(5).split(Xa, ya, g):
                mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(Xa[tr], ya[tr]); pred[te] = mdl.predict_proba(Xa[te])[:, 1]
            t15 = (c.date.dt.year.values[keep] == 2015); mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xa[t15], ya[t15])
            p16 = mdl.predict_proba(Xa[~t15])[:, 1]
            rows.append(dict(outcome=outc, model=name, n=int(keep.sum()), failure_rate=ya.mean(), cv_auc=roc_auc_score(ya, pred), cv_logloss=log_loss(ya, pred),
                             cv_brier=brier_score_loss(ya, pred), temporal_auc_2016=roc_auc_score(ya[~t15], p16), temporal_logloss_2016=log_loss(ya[~t15], p16)))
        # cluster robust Wald test for the two added terms
        Xa = sm.add_constant(feats_aug.values[keep]); cols = ['const'] + list(feats_aug.columns)
        fit = sm.GLM(y[keep], Xa, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(c.block.values[keep])[0]})
        idx = [cols.index('log_ref_dist'), cols.index('mismatch')]; Rm = np.zeros((2, len(cols))); Rm[0, idx[0]] = 1; Rm[1, idx[1]] = 1
        from scipy.stats import chi2
        bb = np.asarray(fit.params)[idx]; VV = np.asarray(fit.cov_params())[np.ix_(idx, idx)]; wstat = float(bb @ np.linalg.solve(VV, bb))
        rows.append(dict(outcome=outc, model='Wald test, distance and mismatch jointly zero', n=int(keep.sum()), wald_stat=wstat, wald_p=float(chi2.sf(wstat, 2)),
                         or_ref_dist=float(np.exp(fit.params[idx[0]])), or_mismatch=float(np.exp(fit.params[idx[1]]))))
    R['C_diagnosis'] = pd.DataFrame(rows)
    # within group: is failure explained by XLEVEL excess alone for XLEVEL only cases
    xs = c[c.group == 'XLEVEL only']; ex = excess(df.set_index('rid').loc[xs.index, F].values)[:, X]
    R['C_xlevel_only_by_excess'] = pd.DataFrame({'excess_bin': pd.qcut(ex, 4).astype(str), 'fail': xs.fail_ladder.values, 'dist': xs.nn_dist.values}).groupby('excess_bin').agg(
        n=('fail', 'size'), fail=('fail', 'mean'), ref_dist_med=('dist', 'median')).reset_index()

# ---------------- A  training only bounds, fresh generation rerun ----------------
if PARTS in ('A', 'ALL'):
    fr = pd.concat([pickle.load(open(f, 'rb')) for f in sorted(glob.glob('out/fresh_tr_[0-9].pkl'))], ignore_index=True)
    fo = pd.concat([pickle.load(open(f, 'rb')) for f in sorted(glob.glob('out/fresh_[0-9].pkl'))], ignore_index=True)
    rows = []; fresh_rows = []
    upd = pickle.load(open('out/model_updated_2013_2015.pkl', 'rb'))
    for tag, frx, Bm in [('all year bounds (original)', fo, None), ('training only bounds', fr, B14)]:
        f0 = frx[(frx.model == 'f0') & (frx.seed == SEED)].set_index('rid').loc[base.rid]
        Z = np.array([z if o else x for z, o, x in zip(f0.z, f0.ok, Xt)])
        vf, vp, vb, _ = validate(model, Xt, Z, B14 if Bm is not None else bounds(df)); vf &= f0.ok.values
        if Bm is None: vb = inB(Z, bounds(df))
        js = vf & vp & vb; wd = np.abs(Xt[:, W]) > 0.30
        for per in ('Eval 2015', 'Test 2016'):
            p = base.period.values == per
            rows.append(dict(bounds=tag, period=per, fresh_joint=js[p].mean(), fresh_flip=vf[p].mean(), fresh_success_rate=f0.ok.values[p].mean(),
                             sec_per_case=f0.sec.values[p].mean(), queries=f0.queries.values[p].mean()))
        for yr in (2015, 2016):
            p = (base.date.dt.year.values == yr) & wd
            rows.append(dict(bounds=tag, period=f'{yr} WARP driven, frozen model', fresh_flip=vf[p].mean(), n=int(p.sum()), n_success=int(vf[p].sum())))
        fresh_rows.append(base.assign(method=f'fresh [{tag}]', vf=vf, vp=vp, vb=vb, js=js))
        # updated model WARP driven 2016 (locked 84.2%)
        fu = frx[frx.model == 'upd'].set_index('rid'); X16 = df.set_index('rid').loc[fu.index, F].values
        Zu = np.array([z if o else x for z, o, x in zip(fu.z, fu.ok, X16)]); vfu, _, _, _ = validate(upd, X16, Zu, B15 if Bm is not None else bounds(df)); vfu &= fu.ok.values
        wdu = np.abs(X16[:, W]) > 0.30
        rows.append(dict(bounds=tag, period='2016 WARP driven, updated model (locked 84.2%)', fresh_flip=vfu[wdu].mean(), n=int(wdu.sum()), n_success=int(vfu[wdu].sum())))
    R['A_fresh'] = pd.DataFrame(rows)
    # recovery and gate free policy under training only bounds
    fresh_tr = fresh_rows[1].set_index('rid')
    rec = []
    for per in ('Eval 2015', 'Test 2016'):
        sub = lad[lad.period == per].groupby('method').fvf.mean(); ff = (fresh_tr[fresh_tr.period == per].vf & fresh_tr[fresh_tr.period == per].vb).mean()
        for m in ['stale', 'inst_margin', 'inst_ratio', 'sc_excess', 'dg_excess', 'ot_1d', 'ot_mv_ret']:
            rec.append(dict(period=per, method=m, feasible_flip=sub[m], fresh=ff, recovery=(sub[m] - sub['stale']) / (ff - sub['stale'])))
    R['A_recovery'] = pd.DataFrame(rec)
    lv = [best['Instance'], best['Scalar'], best['Diagonal'], best['OT']]
    piv = lad[lad.method.isin(lv)].pivot_table(index='rid', columns='method', values='js', aggfunc='first')
    pol = []
    for per in ('Eval 2015', 'Test 2016'):
        ids = base[base.period == per].rid.values; P = piv.loc[ids]
        acc = P.any(axis=1).values; fj = fresh_tr.loc[ids].js.values
        pol.append(dict(period=per, strategy='Policy no gate, training bounds', success=(acc | fj).mean(), regen_frac=(~acc).mean(), fresh_alone=fj.mean(),
                        stale=lad[(lad.method == 'stale') & (lad.period == per)].js.mean()))
    R['A_policy'] = pd.DataFrame(pol)
    R['A_ladder_rescored'] = lad.groupby(['period', 'method']).js.mean().unstack(0).reset_index()

old = {}
try: old = pickle.load(open('out/followup.pkl', 'rb'))
except Exception: pass
old.update(R); pickle.dump(old, open('out/followup.pkl', 'wb'))
for k, v in R.items(): print('\n=====', k); print(v.round(4).to_string())
