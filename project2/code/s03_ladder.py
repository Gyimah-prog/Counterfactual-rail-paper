"""Counterfactual Project 2: adaptation ladder on real inspections.
For every target inspection t in 2015 and 2016, factors and maps are calibrated on the k completed
inspections before t (past only). Every method shares the same eligible cases, model, direction rule,
FRA clipping and validation contract."""
import numpy as np, pandas as pd, time, pickle, json, sys, ot
from sklearn.metrics import pairwise_distances
from cf2lib import *

K_WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 3      # rolling window (completed inspections)
CAP_HI = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0  # factor cap
TAG = sys.argv[3] if len(sys.argv) > 3 else 'main'
BETA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
import os
SKIP_MV = os.environ.get('SKIP_MV', '0') == '1'           # sensitivity runs skip the multivariate map
CAP_LO, MINCNT, EPS_V, EPS_H, EPS_E = 0.5, 20, 1e-4, 0.005, 0.005
KAP_LO, KAP_HI = 0.25, 8.0

df, model, lib = load(); madv = mad(df); B = bounds(df)
X0 = lib[[f'x0_{f}' for f in F]].values; XC = lib[[f'cf_{f}' for f in F]].values; D0 = XC - X0
dates = sorted(df.date.unique()); ref = df[df.year == 2014]; R = ref[F].values
med0 = np.median(R, 0)
# margin for the instance margin form: typical WARP overshoot beyond the project line in the 2014 library
wd = np.abs(X0[:, W]) > 0.30
M_MARGIN = float(np.median(np.abs(D0[wd, W]) - (np.abs(X0[wd, W]) - 0.30)))
# leave one out NN distance among library origins (support threshold)
dd = pairwise_distances(X0 / madv); np.fill_diagonal(dd, np.inf); SUP99 = float(np.quantile(dd.min(1), 0.99))
P1_DELTA = D0.mean(0)

def stats_family(Z, kind):
    """Per feature magnitude statistic for a population Z (rows x 5)."""
    if kind == 'sigma':
        return Z.var(0), np.full(5, len(Z))
    if kind == 'tail':
        dev = np.abs(Z - med0); dev0 = np.abs(R - med0); q0 = np.quantile(dev0, BETA, axis=0)
        out, cnt = np.zeros(5), np.zeros(5)
        for j in range(5):
            s = dev[:, j][dev[:, j] > q0[j]]; cnt[j] = len(s); out[j] = s.mean() if len(s) else np.nan
        return out, cnt
    if kind in ('excess', 'mproxy'):
        if kind == 'excess': e = excess(Z)
        else:
            e = np.zeros_like(Z); e[:, W] = np.maximum(np.abs(Z[:, W]) - 0.283, 0)
        out, cnt = np.zeros(5), np.zeros(5)
        for j in range(5):
            s = e[:, j][e[:, j] > 0]; cnt[j] = len(s); out[j] = s.mean() if len(s) else np.nan
        return out, cnt

REF_STATS = {k: stats_family(R, k) for k in ('sigma', 'tail', 'excess', 'mproxy')}

def factors(Cwin_list, kind):
    """Diagonal factor vector from the calibration window. Sparse tails expand to earlier windows,
    then shrink to 1 when still below the minimum count. Clipped to [CAP_LO, CAP_HI]."""
    r, rc = REF_STATS[kind]
    eps = EPS_V if kind == 'sigma' else (EPS_H if kind == 'tail' else EPS_E)
    a = np.ones(5); used = np.zeros(5, int); shrunk = np.zeros(5, bool)
    for j in range(5):
        ok = False
        # start from the registered K_WIN most recent inspections, expand to earlier ones only when sparse
        for n_use in range(min(K_WIN, len(Cwin_list)), len(Cwin_list) + 1):
            Z = np.vstack(Cwin_list[:n_use])
            v, c = stats_family(Z, kind)
            if c[j] >= MINCNT and rc[j] >= MINCNT and np.isfinite(v[j]):
                val = (v[j] + eps) / (r[j] + eps)
                a[j] = np.sqrt(val) if kind == 'sigma' else val; used[j] = n_use; ok = True; break
        if not ok: shrunk[j] = True
    raw = a.copy(); a = np.clip(a, CAP_LO, CAP_HI)
    return a, dict(raw=raw.tolist(), clipped=(raw != a).tolist(), shrunk=shrunk.tolist(), windows=used.tolist())

# ---------- optimal transport ----------
def qmap(src, tgt, n=2001):
    lv = np.linspace(0, 1, n); qs = np.quantile(src, lv); qt = np.quantile(tgt, lv)
    u, inv = np.unique(qs, return_inverse=True); qtu = np.bincount(inv, qt) / np.bincount(inv)
    def T(w):
        w = np.asarray(w, float); out = np.interp(w, u, qtu)
        out = np.where(w < u[0], w + (qtu[0] - u[0]), out); out = np.where(w > u[-1], w + (qtu[-1] - u[-1]), out)
        return out
    return T
rng = np.random.default_rng(SEED)
src_def = R[G(R)]; src_nd = R[~G(R)][rng.choice((~G(R)).sum(), 1500, replace=False)]
S_ATOMS = np.vstack([src_def, src_nd]); p0 = G(R).mean()
A_W = np.r_[np.full(len(src_def), p0 / len(src_def)), np.full(1500, (1 - p0) / 1500)]
SA = S_ATOMS / madv
dS = pairwise_distances(SA); np.fill_diagonal(dS, np.inf); S_SUP99 = float(np.quantile(dS.min(1), 0.99))

def fit_mv(Ct):
    g = G(Ct); tg = Ct[g]; nd = Ct[~g]; r2 = np.random.default_rng(SEED)
    nd = nd[r2.choice(len(nd), min(1500, len(nd)), replace=False)]
    Y = np.vstack([tg, nd]); pt = g.mean()
    b = np.r_[np.full(len(tg), pt / max(1, len(tg))), np.full(len(nd), (1 - pt) / len(nd))]
    M = pairwise_distances(SA, Y / madv, metric='sqeuclidean'); reg = 0.05 * np.median(M)
    P = ot.sinkhorn(A_W, b, M / np.median(M), reg / np.median(M), method='sinkhorn_log', numItermax=3000)
    TS = (P @ Y) / P.sum(1, keepdims=True)
    disp = TS - S_ATOMS; marg_err = float(np.abs(P.sum(0) - b).sum())
    def T(Z):
        Zs = np.atleast_2d(Z) / madv; d = pairwise_distances(Zs, SA); idx = np.argsort(d, 1)[:, :10]
        dk = np.take_along_axis(d, idx, 1); h = np.maximum(np.median(dk, 1, keepdims=True), 1e-6)
        w = np.exp(-dk ** 2 / (2 * h ** 2)); w /= w.sum(1, keepdims=True)
        return np.atleast_2d(Z) + np.einsum('ik,ikj->ij', w, disp[idx]), dk[:, 0]
    return T, marg_err

def run_month(t):
    prev = [d for d in dates if d < t][::-1]
    Cw = [df[df.date == d][F].values for d in prev]              # most recent first
    s = df[(df.date == t) & (df.defect == 1)]
    Xt = s[F].values; elig = model.predict(Xt) == 1
    s = s[elig]; Xt = Xt[elig]; n = len(Xt)
    if n == 0: return [], {}
    acts, nn, nnd, eT, unsup, finfo, tm = compute_actions(Xt, Cw)
    return validate_all(acts, Xt, s, t, nn, nnd, eT, unsup, model), dict(finfo, times=tm, n=n)

def compute_actions(Xt, Cw):
    Ct = np.vstack(Cw[:K_WIN]); n = len(Xt)
    tm = {}; t0 = time.time()
    dist = pairwise_distances(Xt / madv, X0 / madv); nn = dist.argmin(1); nnd = dist[np.arange(n), nn]
    tm['retrieval'] = time.time() - t0
    Dnn = D0[nn]; acts = {}
    acts['stale'] = Dnn
    acts['ctrl_p1'] = np.tile(P1_DELTA, (n, 1))
    acts['ctrl_none'] = np.zeros_like(Dnn)
    Zthr = Xt.copy()
    Zthr = np.where(Zthr > PROJ_HI, PROJ_HI - 0.001, np.where(Zthr < PROJ_LO, PROJ_LO + 0.001, Zthr))
    acts['ctrl_threshold'] = Zthr - Xt
    # instance benchmark
    eT = np.maximum(np.abs(Xt[:, W]) - 0.30, 0); eO = np.maximum(np.abs(X0[nn, W]) - 0.30, 0)
    kap = np.where(eT > 0, (eT + EPS_E) / (eO + EPS_E), 1.0); kap = np.clip(kap, KAP_LO, KAP_HI)
    acts['inst_ratio'] = Dnn * kap[:, None]
    dW = np.maximum(np.abs(Dnn[:, W]), 1e-4)
    kapm = np.where(eT > 0, (eT + M_MARGIN) / dW, 1.0); kapm = np.clip(kapm, KAP_LO, KAP_HI)
    acts['inst_margin'] = Dnn * kapm[:, None]
    finfo = {}
    t0 = time.time()
    for kind in ('sigma', 'tail', 'excess', 'mproxy'):
        a, info = factors(Cw, kind); finfo[kind] = dict(a=a.tolist(), **info)
        acts[f'sc_{kind}'] = Dnn * a[W]
        if kind != 'mproxy': acts[f'dg_{kind}'] = Dnn * a[None, :]
    tm['factors'] = time.time() - t0
    # OT level
    t0 = time.time(); T1 = qmap(R[:, W], Ct[:, W])
    d1 = Dnn.copy(); d1[:, W] = T1(XC[nn, W]) - T1(X0[nn, W]); acts['ot_1d'] = d1
    tm['ot_1d'] = time.time() - t0; t0 = time.time()
    Tm = [qmap(R[:, j], Ct[:, j]) for j in range(5)]
    dm = np.column_stack([Tm[j](XC[nn, j]) - Tm[j](X0[nn, j]) for j in range(5)]); acts['ot_marg'] = dm
    tm['ot_marg'] = time.time() - t0; t0 = time.time()
    if SKIP_MV:
        return acts, nn, nnd, eT, np.zeros(n, bool), finfo, tm
    Tmv, merr = fit_mv(Ct); tm['ot_mv_fit'] = time.time() - t0
    TX0, sup0 = Tmv(X0); TXC, supc = Tmv(XC); DOT = TXC - TX0
    acts['ot_mv'] = DOT[nn]
    dret = pairwise_distances(Xt / madv, TX0 / madv); nnr = dret.argmin(1); acts['ot_mv_ret'] = DOT[nnr]
    unsup = (supc[nn] > S_SUP99) | (sup0[nn] > S_SUP99)
    finfo['ot_mv'] = dict(marg_err=merr, frac_unsupported_endpoints=float(unsup.mean()))
    return acts, nn, nnd, eT, unsup, finfo, tm

def validate_all(acts, Xt, s, t, nn, nnd, eT, unsup, vmodel):
    Dnn = D0[nn]; n = len(Xt)
    rows = []; grp = group(Xt)
    ref_sup = np.abs(Dnn) / madv > 0.1          # active support of matched action, eta = 0.1
    viol = (Xt < PROJ_LO) | (Xt > PROJ_HI); mismatch = (viol & ~ref_sup).any(1)
    for m, D in acts.items():
        t0 = time.time(); Z = apply_action(Xt, D); vf, vp, vb, p1 = validate(vmodel, Xt, Z, B)
        tv = time.time() - t0
        clipped = ((Xt + warp_dir(Xt, D)) != Z).any(1)
        over = (np.sign(Z[:, W]) != np.sign(Xt[:, W])) & (Z[:, W] != 0) & (Xt[:, W] != 0)
        cost = (np.abs(Z - Xt) / madv).sum(1)
        rows.append(pd.DataFrame(dict(method=m, rid=s.rid.values, date=t, MILE=s.MILE.values.astype(int),
            FEET=s.FEET.values.astype(int), group=grp, eW=eT, nn_dist=nnd, nn_idx=nn, mismatch=mismatch,
            unsup_ot=unsup, vf=vf, vp=vp, vb=vb, js=vf & vp & vb, p1=p1, cost=cost, clipped=clipped,
            overshoot=over, dW=(Z - Xt)[:, W], t_validate=tv / n,
            **{f'z_{f}': Z[:, j] for j, f in enumerate(F)})))
    return rows

if __name__ == '__main__':
    targets = [d for d in dates if pd.Timestamp(d).year >= 2015]
    allrows, finfo = [], {}
    for t in targets:
        r, fi = run_month(t); allrows += r; finfo[str(pd.Timestamp(t).date())] = fi
        print(pd.Timestamp(t).date(), fi.get('n'), {k: round(v, 3) for k, v in fi['times'].items()}, flush=True)
    res = pd.concat(allrows, ignore_index=True)
    pickle.dump(res, open(f'out/ladder_{TAG}.pkl', 'wb'))
    json.dump(dict(finfo=finfo, M_MARGIN=M_MARGIN, SUP99=SUP99, S_SUP99=S_SUP99, K_WIN=K_WIN, CAP_HI=CAP_HI, BETA=BETA),
              open(f'out/ladder_{TAG}_info.json', 'w'), default=str, indent=1)
    res['yr'] = pd.to_datetime(res.date).dt.year
    print(res.groupby(['yr', 'method'])[['vf', 'js']].mean().unstack(0).round(3).to_string())
