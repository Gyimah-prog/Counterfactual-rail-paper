"""Fresh generation with the paper's two stage optimizer, split into independent single threaded workers.
Jobs: f0 on all eligible 2015 and 2016 cases, the updated 2013-2015 model on eligible 2016 cases,
and a repeated seed run (seed 7) on 200 random eligible 2016 cases for consistency."""
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, pandas as pd, time, pickle
from cf2lib import *
PART, NP = int(sys.argv[1]), int(sys.argv[2])
df, model, lib = load(); madv = mad(df); B = bounds(df)
TRAIN_BOUNDS = os.environ.get('TRAIN_BOUNDS', '0') == '1'   # training only bounds, no future data
def bnd(d): return np.c_[np.maximum(FRA_LO, d[F].min().values), np.minimum(FRA_HI, d[F].max().values)]
BM = {'f0': bnd(df[df.year.isin([2013, 2014])]), 'upd': bnd(df[df.year.isin([2013, 2014, 2015])])} if TRAIN_BOUNDS else {'f0': B, 'upd': B}
OUTP = 'fresh_tr' if TRAIN_BOUNDS else 'fresh'
upd = pickle.load(open('out/model_updated_2013_2015.pkl', 'rb'))
for m in (model, upd): m.set_params(n_jobs=1)
MODELS = {'f0': model, 'upd': upd}
jobs = []
for yr in (2015, 2016):
    s = df[(df.year == yr) & (df.defect == 1)]; e = model.predict(s[F].values) == 1
    jobs += [('f0', r, x, SEED) for r, x in zip(s.rid.values[e], s[F].values[e])]
s = df[(df.year == 2016) & (df.defect == 1)]; e = upd.predict(s[F].values) == 1
jobs += [('upd', r, x, SEED) for r, x in zip(s.rid.values[e], s[F].values[e])]
s = df[(df.year == 2016) & (df.defect == 1)]; e = model.predict(s[F].values) == 1
rng = np.random.default_rng(1); pick = rng.choice(np.where(e)[0], 200, replace=False)
jobs += [('f0', r, x, 7) for r, x in zip(s.rid.values[pick], s[F].values[pick])]
jobs = jobs[PART::NP]; print('jobs', len(jobs), flush=True)
res = []; t = time.time()
for i, (mk, rid, x, seed) in enumerate(jobs):
    cm = Counter(MODELS[mk]); t1 = time.time(); z = find_cf(cm, x, BM[mk], madv, seed=seed)
    res.append(dict(model=mk, rid=rid, seed=seed, ok=z is not None, z=z, sec=time.time() - t1, queries=cm.n))
    if (i + 1) % 250 == 0:
        print(i + 1, '%.0fs' % (time.time() - t), flush=True)
        pickle.dump(pd.DataFrame(res), open(f'out/{OUTP}_{PART}.pkl', 'wb'))
pickle.dump(pd.DataFrame(res), open(f'out/{OUTP}_{PART}.pkl', 'wb')); print('DONE %.0fs' % (time.time() - t), flush=True)
