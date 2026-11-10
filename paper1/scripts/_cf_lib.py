"""Shared CF generation library lifted from the notebook (cells 33, 34, 37).

Loads data, trains XGBoost on 2013-14, exposes find_cf().
"""
import os, re, pickle, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'scripts', '.cache')
os.makedirs(CACHE, exist_ok=True)

FEATURES = ['GAGE', 'ALI_L_62', 'ALI_R_62', 'WARP_62', 'XLEVEL']
SEED = 42

PROJECT_LIMITS = {
    'GAGE': (56.3, 57.2), 'ALI_L_62': (-0.20, 0.20),
    'ALI_R_62': (-0.20, 0.20), 'WARP_62': (-0.30, 0.30),
    'XLEVEL': (-0.80, 0.80),
}
FRA_LIMITS = {
    'GAGE': (56.0, 57.5), 'ALI_L_62': (-0.50, 0.50),
    'ALI_R_62': (-0.50, 0.50), 'WARP_62': (-1.00, 1.00),
    'XLEVEL': (-1.00, 1.00),
}
COST_W = np.array([5.0, 1.0, 1.0, 1.0, 1.0])

# ---------------------------------------------------------------------------
def load_data():
    pkl = os.path.join(CACHE, 'data.pkl')
    if os.path.exists(pkl):
        return pickle.load(open(pkl, 'rb'))

    xl = pd.read_excel(os.path.join(ROOT, 'data', 'RawD.xlsx'),
                       sheet_name=None, engine='openpyxl')
    rx = re.compile(r'^(\d{1,2})_(\d{4})$')
    frames = []
    for name, d in xl.items():
        m = rx.match(name)
        if not m: continue
        mo, yr = map(int, m.groups())
        d = d.copy()
        d['date']  = pd.Timestamp(year=yr, month=mo, day=1)
        d['year']  = d['date'].dt.year
        d['month'] = d['date'].dt.month
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True).sort_values('date').reset_index(drop=True)
    for c in ('MILE', 'FEET'):
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors='coerce').astype('Int64')

    df = raw.dropna(subset=FEATURES).copy()
    df['def_GAGE']   = (df['GAGE'] < 56.3) | (df['GAGE'] > 57.2)
    df['def_ALI_L']  = df['ALI_L_62'].abs() > 0.20
    df['def_ALI_R']  = df['ALI_R_62'].abs() > 0.20
    df['def_WARP']   = df['WARP_62'].abs()  > 0.30
    df['def_XLEVEL'] = df['XLEVEL'].abs()   > 0.80
    df['defect'] = df[[c for c in df.columns if c.startswith('def_')]].any(axis=1).astype(int)

    pickle.dump(df, open(pkl, 'wb'))
    return df

# ---------------------------------------------------------------------------
def train_model(df, train_years=(2013, 2014), tag='base'):
    pkl = os.path.join(CACHE, f'model_{tag}.pkl')
    if os.path.exists(pkl):
        return pickle.load(open(pkl, 'rb'))

    train = df[df['year'].isin(train_years)]
    X = train[FEATURES].values
    y = train['defect'].values
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric='logloss', random_state=SEED, n_jobs=-1,
    )
    model.fit(X, y)
    pickle.dump(model, open(pkl, 'wb'))
    return model

# ---------------------------------------------------------------------------
def compute_cf_bounds(df):
    return {
        f: (max(FRA_LIMITS[f][0], df[f].min()),
            min(FRA_LIMITS[f][1], df[f].max()))
        for f in FEATURES
    }

def compute_mad(df, train_years=(2013, 2014)):
    tr = df[df['year'].isin(train_years)]
    mad = tr[FEATURES].apply(lambda x: (x - x.median()).abs().median()).values
    return np.where(mad == 0, 1e-6, mad)

def mad_distance(x, x0, mad):
    return np.sum(np.abs(x - x0) / mad)

def find_warp_flip_point(model, x0, warp_idx, cf_bounds_arr):
    warp_val = x0[warp_idx]
    def pred_prob(val):
        x = x0.copy(); x[warp_idx] = val
        return model.predict_proba(x.reshape(1,-1))[0,1] - 0.5
    try:
        if warp_val < 0:
            lo = max(cf_bounds_arr[warp_idx,0], warp_val); hi = 0.0
            if pred_prob(lo) * pred_prob(hi) < 0:
                return brentq(pred_prob, lo, hi, xtol=1e-4)
            if pred_prob(0.0) * pred_prob(cf_bounds_arr[warp_idx,1]) < 0:
                return brentq(pred_prob, 0.0, cf_bounds_arr[warp_idx,1], xtol=1e-4)
        else:
            lo = 0.0; hi = min(cf_bounds_arr[warp_idx,1], warp_val)
            if pred_prob(lo) * pred_prob(hi) < 0:
                return brentq(pred_prob, lo, hi, xtol=1e-4)
            if pred_prob(cf_bounds_arr[warp_idx,0]) * pred_prob(0.0) < 0:
                return brentq(pred_prob, cf_bounds_arr[warp_idx,0], 0.0, xtol=1e-4)
    except ValueError:
        pass
    return None

def find_cf(model, x0, cf_bounds_arr, mad, l2_weight=0.5, seed=SEED):
    rng = np.random.default_rng(seed)
    x0  = np.array(x0, dtype=float)
    warp_idx = FEATURES.index('WARP_62')
    candidates = []

    def objective(x):
        x  = np.clip(x, cf_bounds_arr[:,0], cf_bounds_arr[:,1])
        p  = np.clip(model.predict_proba(x.reshape(1,-1))[0,1], 1e-9, 1-1e-9)
        return -np.log(1 - p) + l2_weight * mad_distance(x, x0, mad)

    flip_val = find_warp_flip_point(model, x0, warp_idx, cf_bounds_arr)
    if flip_val is not None:
        x_init = x0.copy()
        margin = 0.05 * abs(flip_val - x0[warp_idx])
        x_init[warp_idx] = flip_val - np.sign(x0[warp_idx]) * margin
        x_init = np.clip(x_init, cf_bounds_arr[:,0], cf_bounds_arr[:,1])
        if int(model.predict(x_init.reshape(1,-1))[0]) == 0:
            candidates.append((x_init.copy(), objective(x_init)))
        for noise in (0.0, 0.005, 0.01):
            xs = x_init + rng.normal(0, noise, len(x0)) * (cf_bounds_arr[:,1] - cf_bounds_arr[:,0])
            xs = np.clip(xs, cf_bounds_arr[:,0], cf_bounds_arr[:,1])
            res = minimize(objective, xs, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol':1e-5, 'fatol':1e-5})
            xr = np.clip(res.x, cf_bounds_arr[:,0], cf_bounds_arr[:,1])
            if int(model.predict(xr.reshape(1,-1))[0]) == 0:
                candidates.append((xr.copy(), res.fun))

    if not candidates:
        ranges = cf_bounds_arr[:,1] - cf_bounds_arr[:,0]
        for _ in range(5):
            xr = x0 + rng.normal(0, 0.02, len(x0)) * ranges
            xr = np.clip(xr, cf_bounds_arr[:,0], cf_bounds_arr[:,1])
            res = minimize(objective, xr, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol':1e-5, 'fatol':1e-5})
            xr2 = np.clip(res.x, cf_bounds_arr[:,0], cf_bounds_arr[:,1])
            if int(model.predict(xr2.reshape(1,-1))[0]) == 0:
                candidates.append((xr2.copy(), res.fun))

    if not candidates:
        return None
    best_x = min(candidates, key=lambda c: c[1])[0]
    return {
        'cf': best_x,
        'dist': mad_distance(best_x, x0, mad),
        'p_cf': model.predict_proba(best_x.reshape(1,-1))[0,1],
    }

# ---------------------------------------------------------------------------
def generate_cfs_for_cohort(model, df, year, tag, max_samples=None):
    """Generate CFs for all defective samples in a given year. Caches output."""
    pkl = os.path.join(CACHE, f'cf_{tag}.pkl')
    if os.path.exists(pkl):
        return pickle.load(open(pkl, 'rb'))

    cf_bounds_arr = np.array([list(compute_cf_bounds(df)[f]) for f in FEATURES])
    mad = compute_mad(df)

    cohort = df[(df['year'] == year) & (df['defect'] == 1)].copy()
    if max_samples is not None:
        cohort = cohort.sample(n=min(max_samples, len(cohort)),
                               random_state=SEED).copy()

    print(f'  [{tag}] {len(cohort)} defective samples in {year}')
    t0 = time.time()
    rows = []
    for i, (idx, row) in enumerate(cohort.iterrows()):
        x0 = row[FEATURES].values.astype(float)
        res = find_cf(model, x0, cf_bounds_arr, mad)
        if res is None: continue
        rec = {'orig_idx': idx, 'year': year, 'date': row['date'],
               'MILE': row.get('MILE'), 'FEET': row.get('FEET')}
        for k, f in enumerate(FEATURES):
            rec[f'x0_{f}'] = x0[k]
            rec[f'cf_{f}'] = res['cf'][k]
            rec[f'd_{f}']  = res['cf'][k] - x0[k]
        rec['cf_dist']      = res['dist']
        rec['cf_warp_abs']  = abs(res['cf'][FEATURES.index('WARP_62')] - x0[FEATURES.index('WARP_62')])
        rec['p_cf']         = res['p_cf']
        rows.append(rec)
        if (i+1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(cohort) - i - 1)
            print(f'    {i+1}/{len(cohort)} | {elapsed:.0f}s elapsed | ~{eta:.0f}s remaining',
                  flush=True)

    cf_df = pd.DataFrame(rows)
    print(f'  [{tag}] Done in {time.time()-t0:.0f}s — {len(cf_df)} CFs')
    pickle.dump(cf_df, open(pkl, 'wb'))
    return cf_df
