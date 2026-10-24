"""RQ1 monitoring. Monthly stale transfer validity next to model metrics and drift statistics for all
28 inspections. Alarm thresholds are registered before looking at 2015 and 2016 results."""
import numpy as np, pandas as pd, pickle, json
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, brier_score_loss, pairwise_distances
from cf2lib import *
df, model, lib = load(); madv = mad(df); B = bounds(df)
X0 = lib[[f'x0_{f}' for f in F]].values; D0 = lib[[f'd_{f}' for f in F]].values
ref = df[df.year == 2014][F].values
sdW0 = ref[:, W].std(); tp0 = (np.abs(ref[:, W]) > 0.30).mean()
ALARM = dict(validity=0.80, auc=0.99, ece=0.02, std_ratio=1.20, tail_ratio=2.0, ks_warp=0.05)
def ece(y, p, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1); e = 0
    for k in range(nb):
        m = b == k
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e
rows = []
for d in sorted(df.date.unique()):
    s = df[df.date == d]; Xa = s[F].values; y = s.defect.values; p = model.predict_proba(Xa)[:, 1]
    sd = s[s.defect == 1]; Xt = sd[F].values; el = model.predict(Xt) == 1; Xt = Xt[el]
    dist = pairwise_distances(Xt / madv, X0 / madv); nn = dist.argmin(1)
    Z = apply_action(Xt, D0[nn]); vf, vp, vb, _ = validate(model, Xt, Z, B)
    g = group(sd[F].values)
    rows.append(dict(date=pd.Timestamp(d), n_rows=len(s), n_def=int(y.sum()), n_elig=int(el.sum()),
        validity=(vf & vb).mean(), js=(vf & vp & vb).mean(),
        auc=roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan, brier=brier_score_loss(y, p), ece=ece(y, p),
        std_ratio=Xa[:, W].std() / sdW0, tail_ratio=(np.abs(Xa[:, W]) > 0.30).mean() / tp0,
        ks_warp=ks_2samp(Xa[:, W], ref[:, W]).statistic, ks_xlevel=ks_2samp(Xa[:, X], ref[:, X]).statistic,
        ref_dist_median=float(np.median(dist.min(1))),
        excess_W=float(np.mean(np.abs(Xa[:, W])[np.abs(Xa[:, W]) > 0.3] - 0.3)) if (np.abs(Xa[:, W]) > 0.3).any() else np.nan,
        share_warp_only=(g == 'WARP only').mean(), share_xlevel_only=(g == 'XLEVEL only').mean(),
        share_joint=(g == 'WARP and XLEVEL').mean(), share_other=(g == 'Other').mean()))
mon = pd.DataFrame(rows)
post = mon[mon.date >= '2015-01-01']
first = {}
for k, thr in ALARM.items():
    hit = post[post[k] < thr] if k in ('validity', 'auc') else post[post[k] > thr]
    first[k] = str(hit.date.min().date()) if len(hit) else 'never'
pickle.dump(mon, open('out/monitor.pkl', 'wb')); json.dump(dict(alarm_thresholds=ALARM, first_alarm=first), open('out/monitor_alarms.json', 'w'), indent=1)
print(mon.round(3).to_string()); print(first)
