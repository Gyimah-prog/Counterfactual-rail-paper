"""Controlled stress tests. Known statistical perturbations of the 2014 geometry, labels recomputed,
f0 fixed, calibration on the transformed population (transductive, labeled as such)."""
import numpy as np, pandas as pd, pickle
import s03_ladder as L
from cf2lib import *

R = L.R; med0 = L.med0; sd0 = R.std(0); rng = np.random.default_rng(SEED)
def rot(Z, deg):
    Z = Z.copy(); a = np.deg2rad(deg)
    u = (Z[:, W] - med0[W]) / sd0[W]; v = (Z[:, X] - med0[X]) / sd0[X]
    Z[:, W] = med0[W] + sd0[W] * (np.cos(a) * u + np.sin(a) * v)
    Z[:, X] = med0[X] + sd0[X] * (-np.sin(a) * u + np.cos(a) * v); return Z
def tail(Z, q=0.90, f=1.8):
    Z = Z.copy(); d = Z[:, W] - med0[W]; qq = np.quantile(np.abs(d), q); big = np.abs(d) > qq
    Z[big, W] = med0[W] + np.sign(d[big]) * (qq + f * (np.abs(d[big]) - qq)); return Z
def reweight(Z, k=4):
    g = group(Z); w = np.where(g == 'XLEVEL only', k, 1.0); w /= w.sum()
    return Z[rng.choice(len(Z), len(Z), replace=True, p=w)]
SC = {
 'S0 no change': lambda Z: Z.copy(),
 'S1 common scale x1.4': lambda Z: med0 + 1.4 * (Z - med0),
 'S2 heterogeneous scale': lambda Z: med0 + np.array([1.0, 0.9, 0.9, 1.6, 1.15]) * (Z - med0),
 'S3 WARP tail inflation': lambda Z: tail(Z),
 'S4 WARP location shift +0.06': lambda Z: Z + np.array([0, 0, 0, 0.06, 0]),
 'S5 WARP XLEVEL rotation 25 deg': lambda Z: rot(Z, 25),
 'S6 XLEVEL only reweight x4': lambda Z: reweight(Z),
}
rows = []
for name, f in SC.items():
    Zp = f(R); g = G(Zp); e = g & (L.model.predict(Zp) == 1); Xt = Zp[e]
    if len(Xt) > 1500: Xt = Xt[rng.choice(len(Xt), 1500, replace=False)]
    acts, nn, nnd, eT, unsup, finfo, tm = L.compute_actions(Xt, [Zp])
    for m, D in acts.items():
        Z = apply_action(Xt, D); vf, vp, vb, _ = validate(L.model, Xt, Z, L.B)
        rows.append(dict(scenario=name, method=m, n=len(Xt), n_elig_total=int(e.sum()),
                         xlev_share=float((group(Xt) == 'XLEVEL only').mean()),
                         vf=vf.mean(), vfb=(vf & vb).mean(), js=(vf & vp & vb).mean()))
    print(name, len(Xt), flush=True)
res = pd.DataFrame(rows); pickle.dump(res, open('out/stress.pkl', 'wb'))
print(res.pivot(index='method', columns='scenario', values='js').round(3).to_string())
