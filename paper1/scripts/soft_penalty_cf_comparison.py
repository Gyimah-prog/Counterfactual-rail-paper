"""Soft-penalty (Wachter-style) counterfactual experiment.

Adds a SECOND CF method for comparison with the existing hard-constrained
method. Nothing in the original notebook / hard-constrained code is touched.

All outputs go to outputs_soft_penalty_cf/ so the original results stay clean.

Objective:
    L(x') = lambda_pred * (p_defect(x') - 0)^2
            + lambda_dist * d_MAD(x', x0)
            + gamma_fra   * R_FRA(x')
            + gamma_emp   * R_EMP(x')

The optimizer searches a wider empirical-plus-10% box; the penalties
discourage but do not forbid going past FRA or empirical bounds, so we can
*measure* violation rates rather than masking them by hard clipping.

Usage:
    python scripts/soft_penalty_cf_comparison.py

Requires the existing scripts/.cache/ pickles produced by compute_all.py.
"""
from __future__ import annotations
import os, sys, time, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'scripts' / '.cache'
OUT   = ROOT / 'outputs_soft_penalty_cf'
OUT.mkdir(exist_ok=True)

FEATURES = ['GAGE', 'ALI_L_62', 'ALI_R_62', 'WARP_62', 'XLEVEL']
SEED     = 42
WARP_IDX = FEATURES.index('WARP_62')

PROJECT_LIMITS = {
    'GAGE':     (56.3,  57.2),
    'ALI_L_62': (-0.20,  0.20),
    'ALI_R_62': (-0.20,  0.20),
    'WARP_62':  (-0.30,  0.30),
    'XLEVEL':   (-0.80,  0.80),
}
FRA_LIMITS = {
    'GAGE':     (56.0,  57.5),
    'ALI_L_62': (-0.50,  0.50),
    'ALI_R_62': (-0.50,  0.50),
    'WARP_62':  (-1.00,  1.00),
    'XLEVEL':   (-1.00,  1.00),
}

# ──────────────────────────────────────────────────────────────────────────
# Required functions per spec
# ──────────────────────────────────────────────────────────────────────────
def mad_distance(x_prime, x0, MAD):
    """L1 distance scaled by per-feature MAD (Wachter-style proximity term)."""
    return float(np.sum(np.abs(x_prime - x0) / MAD))


def bound_penalty(x_prime, lower_arr, upper_arr):
    """Sum of squared bound violations across features. Zero if inside box."""
    low_violation  = np.maximum(0.0, lower_arr - x_prime)
    high_violation = np.maximum(0.0, x_prime - upper_arr)
    return float(np.sum(low_violation ** 2) + np.sum(high_violation ** 2))


def max_bound_violation(x_prime, lower_arr, upper_arr):
    """Largest single-feature violation amount (0 if compliant)."""
    low  = np.maximum(0.0, lower_arr - x_prime)
    high = np.maximum(0.0, x_prime - upper_arr)
    return float(max(low.max(), high.max()))


def violates_bounds(x_prime, lower_arr, upper_arr, tol=1e-8):
    """True if ANY feature is outside [lower, upper] by more than tol."""
    return bool(np.any(x_prime < lower_arr - tol) or
                np.any(x_prime > upper_arr + tol))


def cf_objective_soft(x_prime, x0, model, MAD,
                      fra_lower_arr, fra_upper_arr,
                      emp_min_arr,   emp_max_arr,
                      lambda_pred=1.0, lambda_dist=0.5,
                      gamma_fra=1000.0, gamma_empirical=100.0):
    """Soft-penalty Wachter-style CF objective. Lower = better.

    Components:
      - prediction term : encourages p(defect) -> 0
      - distance term   : encourages staying near x0 (MAD-scaled L1)
      - FRA penalty     : squared excursion past FRA limits
      - empirical penalty: squared excursion past observed feature range
    """
    p_defect = float(model.predict_proba(x_prime.reshape(1, -1))[0, 1])
    pred_term = lambda_pred * (p_defect - 0.0) ** 2
    dist_term = lambda_dist * mad_distance(x_prime, x0, MAD)
    fra_term  = gamma_fra * bound_penalty(x_prime, fra_lower_arr, fra_upper_arr)
    emp_term  = gamma_empirical * bound_penalty(x_prime, emp_min_arr, emp_max_arr)
    return pred_term + dist_term + fra_term + emp_term


def generate_cf_soft(x0, model, MAD,
                     soft_search_bounds_arr,
                     fra_lower_arr, fra_upper_arr,
                     emp_min_arr,   emp_max_arr,
                     lambda_pred=1.0, lambda_dist=0.5,
                     gamma_fra=1000.0, gamma_empirical=100.0,
                     method='Powell', maxiter=500):
    """Optimize the soft-penalty objective for one sample.

    Returns a dict with the CF, optimizer diagnostics, and violation metrics.
    Does NOT clip x_prime inside the objective — penalties do the work.
    """
    x0 = np.asarray(x0, dtype=float)
    args = (x0, model, MAD, fra_lower_arr, fra_upper_arr,
            emp_min_arr, emp_max_arr,
            lambda_pred, lambda_dist, gamma_fra, gamma_empirical)

    if method == 'Powell':
        # Powell accepts bounds since SciPy 1.5; pass the wider soft search box
        bounds = list(zip(soft_search_bounds_arr[:, 0],
                          soft_search_bounds_arr[:, 1]))
        res = minimize(cf_objective_soft, x0, args=args,
                       method='Powell',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'xtol': 1e-5, 'ftol': 1e-5})
    elif method == 'Nelder-Mead':
        # Nelder-Mead is unconstrained; penalties keep it inside
        res = minimize(cf_objective_soft, x0, args=args,
                       method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-5})
    else:
        raise ValueError(f'Unsupported method: {method}')

    x_cf = np.asarray(res.x, dtype=float)
    p0   = float(model.predict_proba(x0.reshape(1, -1))[0, 1])
    p_cf = float(model.predict_proba(x_cf.reshape(1, -1))[0, 1])
    pred0  = int(model.predict(x0.reshape(1, -1))[0])
    pred_cf = int(model.predict(x_cf.reshape(1, -1))[0])

    fra_violated = violates_bounds(x_cf, fra_lower_arr, fra_upper_arr)
    emp_violated = violates_bounds(x_cf, emp_min_arr,   emp_max_arr)
    max_fra_v   = max_bound_violation(x_cf, fra_lower_arr, fra_upper_arr)
    max_emp_v   = max_bound_violation(x_cf, emp_min_arr,   emp_max_arr)

    out = {
        # original
        **{f'x0_{f}': x0[i] for i, f in enumerate(FEATURES)},
        # counterfactual
        **{f'cf_{f}': x_cf[i] for i, f in enumerate(FEATURES)},
        # signed deltas
        **{f'd_{f}': x_cf[i] - x0[i] for i, f in enumerate(FEATURES)},
        # model probabilities & classes
        'p0': p0, 'p_cf': p_cf,
        'pred0': pred0, 'pred_cf': pred_cf,
        'flipped': (pred0 == 1) and (pred_cf == 0),
        # distance metrics
        'mad_l1':       mad_distance(x_cf, x0, MAD),
        'abs_delta_WARP': abs(x_cf[WARP_IDX] - x0[WARP_IDX]),
        'cf_warp_abs':   abs(x_cf[WARP_IDX] - x0[WARP_IDX]),  # alias used downstream
        # feasibility
        'FRA_violation': fra_violated,
        'empirical_violation': emp_violated,
        'max_FRA_violation_amount': max_fra_v,
        'max_empirical_violation_amount': max_emp_v,
        # optimizer diagnostics
        'optimizer_success': bool(res.success) if hasattr(res, 'success') else None,
        'optimizer_fun': float(res.fun) if hasattr(res, 'fun') else None,
        'optimizer_nit': int(getattr(res, 'nit', 0) or 0),
        # parameters
        'lambda_pred': lambda_pred,
        'lambda_dist': lambda_dist,
        'gamma_fra':   gamma_fra,
        'gamma_empirical': gamma_empirical,
        'optimizer_method': method,
    }
    return out


def generate_soft_cfs_for_defectives(df_defective, model, FEATURES, MAD,
                                      soft_search_bounds_arr,
                                      fra_lower_arr, fra_upper_arr,
                                      emp_min_arr,   emp_max_arr,
                                      lambda_pred=1.0, lambda_dist=0.5,
                                      gamma_fra=1000.0, gamma_empirical=100.0,
                                      method='Powell', maxiter=500,
                                      verbose=True, tag=''):
    """Loop over defective rows, generate soft-penalty CFs, return DataFrame."""
    rows = []
    n = len(df_defective)
    t0 = time.time()
    for i, (idx, row) in enumerate(df_defective.iterrows()):
        x0 = row[FEATURES].values.astype(float)
        rec = generate_cf_soft(
            x0, model, MAD,
            soft_search_bounds_arr,
            fra_lower_arr, fra_upper_arr,
            emp_min_arr, emp_max_arr,
            lambda_pred=lambda_pred, lambda_dist=lambda_dist,
            gamma_fra=gamma_fra, gamma_empirical=gamma_empirical,
            method=method, maxiter=maxiter,
        )
        rec['orig_idx'] = idx
        rec['MILE'] = row.get('MILE')
        rec['FEET'] = row.get('FEET')
        rec['date'] = row.get('date')
        rec['year'] = row.get('year')
        rows.append(rec)
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f'    [{tag}] {i+1}/{n} | {elapsed:.0f}s elapsed | '
                  f'~{eta:.0f}s remaining', flush=True)
    if verbose:
        print(f'    [{tag}] done in {time.time()-t0:.0f}s ({len(rows)} samples)')
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Load cached data + hard-constrained CFs
# ──────────────────────────────────────────────────────────────────────────
print('=' * 70)
print('Loading cached data')
print('=' * 70)
df            = pickle.load(open(CACHE / 'data.pkl',         'rb'))
model         = pickle.load(open(CACHE / 'model_base.pkl',   'rb'))
panel         = pickle.load(open(CACHE / 'persistence.pkl', 'rb'))
hard_cfs_2014 = pickle.load(open(CACHE / 'cf_2014_base.pkl', 'rb'))
print(f'  df rows         : {len(df):,}')
print(f'  hard CFs (2014) : {len(hard_cfs_2014):,}')

# MAD on training data (matches notebook convention)
train_df = df[df['year'].isin([2013, 2014])]
MAD = train_df[FEATURES].apply(
    lambda x: (x - x.median()).abs().median()
).values
MAD = np.where(MAD == 0, 1e-6, MAD)
print(f'  MAD per feature : {dict(zip(FEATURES, [f"{m:.4f}" for m in MAD]))}')

# FRA / empirical / soft-search arrays
fra_lower = np.array([FRA_LIMITS[f][0] for f in FEATURES])
fra_upper = np.array([FRA_LIMITS[f][1] for f in FEATURES])
emp_min   = df[FEATURES].min().values
emp_max   = df[FEATURES].max().values
emp_range = emp_max - emp_min

# Soft search bounds = ±10 % beyond observed range, intersected with FRA ±epsilon
soft_lower = emp_min - 0.10 * emp_range
soft_upper = emp_max + 0.10 * emp_range
# Keep the search range reasonable: never let it exceed FRA by more than 20 %
fra_pad = (fra_upper - fra_lower) * 0.20
soft_lower = np.maximum(soft_lower, fra_lower - fra_pad)
soft_upper = np.minimum(soft_upper, fra_upper + fra_pad)
soft_bounds_arr = np.column_stack([soft_lower, soft_upper])
print('  Soft search bounds:')
for f, (lo, hi) in zip(FEATURES, soft_bounds_arr):
    print(f'    {f:<10s} [{lo:8.3f}, {hi:8.3f}]')


# ──────────────────────────────────────────────────────────────────────────
# 1) Main comparison — soft-penalty CFs on 2014 defective cohort
#    (same population the hard method was run on)
# ──────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('Phase 1 — soft-penalty CFs for 2014 defective cohort')
print('=' * 70)

soft_cache = CACHE / 'cf_2014_soft_main.pkl'
if soft_cache.exists():
    print(f'  reusing cached {soft_cache.name}')
    soft_cfs_2014 = pickle.load(open(soft_cache, 'rb'))
else:
    df_def_2014 = df[(df['year'] == 2014) & (df['defect'] == 1)].copy()
    print(f'  cohort size: {len(df_def_2014)} defective samples')
    soft_cfs_2014 = generate_soft_cfs_for_defectives(
        df_def_2014, model, FEATURES, MAD,
        soft_bounds_arr, fra_lower, fra_upper, emp_min, emp_max,
        lambda_pred=1.0, lambda_dist=0.5,
        gamma_fra=1000.0, gamma_empirical=100.0,
        method='Powell', maxiter=500,
        tag='Powell γ_fra=1000 γ_emp=100',
    )
    pickle.dump(soft_cfs_2014, open(soft_cache, 'wb'))

soft_cfs_2014.to_csv(OUT / 'soft_cf_results.csv', index=False)
print(f'  wrote {OUT/"soft_cf_results.csv"}  ({len(soft_cfs_2014)} rows)')


# ──────────────────────────────────────────────────────────────────────────
# Helper — build a comparison row of metrics for a CF DataFrame
# ──────────────────────────────────────────────────────────────────────────
def cohort_persistence_aucs(cf_df, panel_df, score_col):
    """AUC of score_col against persist_{1,3,6}m on the matched cohort."""
    pp = panel_df.copy()
    pp['key'] = list(zip(pp['MILE'], pp['FEET'], pp['date']))
    cf = cf_df.copy()
    cf['key'] = list(zip(cf['MILE'], cf['FEET'], cf['date']))
    if score_col not in cf.columns:
        # gracefully bail if column missing
        return {f'AUC {h}': np.nan for h in ('1m','3m','6m')}
    merged = pp.merge(cf[['key', score_col]], on='key', how='inner')
    aucs = {}
    for h_col, h_lbl in [('persist_1m','1m'), ('persist_3m','3m'), ('persist_6m','6m')]:
        if h_col not in merged.columns:
            aucs[f'AUC {h_lbl}'] = np.nan; continue
        sub = merged[merged[h_col].notna()]
        y = sub[h_col].astype(int).values
        if len(y) < 20 or len(np.unique(y)) < 2:
            aucs[f'AUC {h_lbl}'] = np.nan
        else:
            aucs[f'AUC {h_lbl}'] = roc_auc_score(y, sub[score_col].values)
    return aucs

def summarize_method(cf_df, name, n_tested):
    """One row of the comparison table."""
    n_gen = len(cf_df)
    if n_gen == 0:
        return {'method': name, 'n_tested': n_tested, 'n_generated': 0}
    # FRA / empirical violations: hard method has no such columns → compute on cf_* values
    if 'FRA_violation' in cf_df.columns:
        fra_v = cf_df['FRA_violation'].mean() * 100
        emp_v = cf_df['empirical_violation'].mean() * 100
    else:
        # Rebuild for hard CFs from their cf_* columns
        cf_vals = cf_df[[f'cf_{f}' for f in FEATURES]].values
        fra_violations = np.array([
            violates_bounds(x, fra_lower, fra_upper) for x in cf_vals])
        emp_violations = np.array([
            violates_bounds(x, emp_min, emp_max) for x in cf_vals])
        fra_v = fra_violations.mean() * 100
        emp_v = emp_violations.mean() * 100

    flip_col = 'flipped' if 'flipped' in cf_df.columns else None
    if flip_col is None:
        # hard CFs: derive by re-checking with model
        cf_vals = cf_df[[f'cf_{f}' for f in FEATURES]].values
        pred_cf = model.predict(cf_vals)
        flip_rate = (pred_cf == 0).mean() * 100
    else:
        flip_rate = cf_df['flipped'].mean() * 100

    # distance metrics
    if 'cf_dist' in cf_df.columns and 'mad_l1' not in cf_df.columns:
        dist_col = 'cf_dist'
    elif 'mad_l1' in cf_df.columns:
        dist_col = 'mad_l1'
    else:
        dist_col = None
    if dist_col is None:
        mean_dist = med_dist = np.nan
    else:
        mean_dist = cf_df[dist_col].mean()
        med_dist  = cf_df[dist_col].median()

    warp_col = 'abs_delta_WARP' if 'abs_delta_WARP' in cf_df.columns else 'cf_warp_abs'
    mean_dwarp = cf_df[warp_col].mean()
    med_dwarp  = cf_df[warp_col].median()

    aucs_dwarp = cohort_persistence_aucs(cf_df, panel, 'cf_warp_abs')
    aucs_dist  = cohort_persistence_aucs(cf_df, panel, dist_col) if dist_col else \
                 {f'AUC {h}': np.nan for h in ('1m','3m','6m')}

    return {
        'method': name,
        'n_tested': n_tested,
        'n_generated': n_gen,
        'flip_rate_pct': flip_rate,
        'FRA_violation_rate_pct': fra_v,
        'empirical_violation_rate_pct': emp_v,
        'mean_mad_l1_distance': mean_dist,
        'median_mad_l1_distance': med_dist,
        'mean_abs_delta_WARP':  mean_dwarp,
        'median_abs_delta_WARP':med_dwarp,
        'AUC_dWARP_1m':  aucs_dwarp['AUC 1m'],
        'AUC_dWARP_3m':  aucs_dwarp['AUC 3m'],
        'AUC_dWARP_6m':  aucs_dwarp['AUC 6m'],
        'AUC_madL1_1m':  aucs_dist['AUC 1m'],
        'AUC_madL1_3m':  aucs_dist['AUC 3m'],
        'AUC_madL1_6m':  aucs_dist['AUC 6m'],
    }

# ──────────────────────────────────────────────────────────────────────────
# 2) Build hard vs soft comparison table
# ──────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('Phase 2 — comparison: hard vs soft')
print('=' * 70)

n_tested = int((df['year'] == 2014).sum() & (df['defect'] == 1).sum())
n_def_2014 = int(((df['year'] == 2014) & (df['defect'] == 1)).sum())

rows = [
    summarize_method(hard_cfs_2014, 'hard_constrained', n_def_2014),
    summarize_method(soft_cfs_2014, 'soft_penalty_powell_gfra1000', n_def_2014),
]
comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / 'soft_vs_hard_comparison.csv', index=False)
print(comparison.to_string(index=False))
print(f'\n  wrote {OUT/"soft_vs_hard_comparison.csv"}')


# ──────────────────────────────────────────────────────────────────────────
# 3) Gamma sensitivity sweep
# ──────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('Phase 3 — gamma sensitivity sweep')
print('=' * 70)

gamma_fra_values = [10, 100, 1000, 10000]
gamma_emp_values = [10, 100, 1000]

# subsample for sweep (full 649 × 12 combos would be too slow)
SWEEP_N = 200
rng = np.random.default_rng(SEED)
df_def_2014 = df[(df['year'] == 2014) & (df['defect'] == 1)].copy()
sweep_sample = df_def_2014.sample(n=min(SWEEP_N, len(df_def_2014)),
                                  random_state=SEED).copy()
print(f'  sweep cohort: {len(sweep_sample)} sub-sampled defectives')

sweep_cache = CACHE / 'cf_2014_soft_gamma_sweep.pkl'
if sweep_cache.exists():
    print(f'  reusing cached {sweep_cache.name}')
    sweep_results = pickle.load(open(sweep_cache, 'rb'))
else:
    sweep_results = {}
    for gf in gamma_fra_values:
        for ge in gamma_emp_values:
            tag = f'γ_fra={gf} γ_emp={ge}'
            print(f'\n  → {tag}')
            cf_df = generate_soft_cfs_for_defectives(
                sweep_sample, model, FEATURES, MAD,
                soft_bounds_arr, fra_lower, fra_upper, emp_min, emp_max,
                lambda_pred=1.0, lambda_dist=0.5,
                gamma_fra=float(gf), gamma_empirical=float(ge),
                method='Powell', maxiter=500,
                tag=tag,
            )
            sweep_results[(gf, ge)] = cf_df
    pickle.dump(sweep_results, open(sweep_cache, 'wb'))

sweep_rows = []
for (gf, ge), cf_df in sweep_results.items():
    aucs = cohort_persistence_aucs(cf_df, panel, 'cf_warp_abs')
    sweep_rows.append({
        'gamma_fra': gf,
        'gamma_empirical': ge,
        'flip_rate_pct': cf_df['flipped'].mean() * 100,
        'FRA_violation_rate_pct': cf_df['FRA_violation'].mean() * 100,
        'empirical_violation_rate_pct': cf_df['empirical_violation'].mean() * 100,
        'mean_abs_delta_WARP':   cf_df['abs_delta_WARP'].mean(),
        'median_abs_delta_WARP': cf_df['abs_delta_WARP'].median(),
        'AUC_dWARP_1m': aucs['AUC 1m'],
        'AUC_dWARP_3m': aucs['AUC 3m'],
        'AUC_dWARP_6m': aucs['AUC 6m'],
    })
sweep_df = pd.DataFrame(sweep_rows).sort_values(['gamma_fra', 'gamma_empirical'])
sweep_df.to_csv(OUT / 'soft_cf_gamma_sensitivity.csv', index=False)
print('\nGamma sweep summary:')
print(sweep_df.to_string(index=False))
print(f'\n  wrote {OUT/"soft_cf_gamma_sensitivity.csv"}')

# AUC summary table (flatten)
auc_rows = []
auc_rows.append({
    'method': 'hard_constrained', 'gamma_fra': None, 'gamma_empirical': None,
    'AUC_dWARP_1m': comparison.loc[0,'AUC_dWARP_1m'],
    'AUC_dWARP_3m': comparison.loc[0,'AUC_dWARP_3m'],
    'AUC_dWARP_6m': comparison.loc[0,'AUC_dWARP_6m'],
})
auc_rows.append({
    'method': 'soft_penalty_default',
    'gamma_fra': 1000, 'gamma_empirical': 100,
    'AUC_dWARP_1m': comparison.loc[1,'AUC_dWARP_1m'],
    'AUC_dWARP_3m': comparison.loc[1,'AUC_dWARP_3m'],
    'AUC_dWARP_6m': comparison.loc[1,'AUC_dWARP_6m'],
})
for _, r in sweep_df.iterrows():
    auc_rows.append({
        'method': f'soft_penalty_powell_g{int(r["gamma_fra"])}_e{int(r["gamma_empirical"])}',
        'gamma_fra': r['gamma_fra'], 'gamma_empirical': r['gamma_empirical'],
        'AUC_dWARP_1m': r['AUC_dWARP_1m'],
        'AUC_dWARP_3m': r['AUC_dWARP_3m'],
        'AUC_dWARP_6m': r['AUC_dWARP_6m'],
    })
auc_summary = pd.DataFrame(auc_rows)
auc_summary.to_csv(OUT / 'soft_cf_auc_summary.csv', index=False)
print(f'  wrote {OUT/"soft_cf_auc_summary.csv"}')

# ──────────────────────────────────────────────────────────────────────────
# 4) Plots
# ──────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('Phase 4 — plots')
print('=' * 70)
plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# (1) Bar chart: hard vs soft on flip / FRA / AUC metrics
metrics = ['flip_rate_pct', 'FRA_violation_rate_pct',
           'AUC_dWARP_1m', 'AUC_dWARP_3m', 'AUC_dWARP_6m']
labels  = ['Flip rate %', 'FRA violation %',
           '1-m AUC',     '3-m AUC',     '6-m AUC']
hard_vals = [comparison.loc[0, m] for m in metrics]
soft_vals = [comparison.loc[1, m] for m in metrics]

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(metrics)); w = 0.36
ax.bar(x - w/2, hard_vals, width=w, color='#1f77b4', edgecolor='white',
       label='Hard-constrained CF  (clip to FRA ∩ empirical bounds)')
ax.bar(x + w/2, soft_vals, width=w, color='#ff7f0e', edgecolor='white',
       label='Soft-penalty CF  (Wachter-style, γ_fra=1000, γ_emp=100)')
for xi, v in zip(x - w/2, hard_vals):
    if not np.isnan(v):
        ax.text(xi, v + max(hard_vals + soft_vals) * 0.012,
                f'{v:.2f}' if v < 5 else f'{v:.1f}',
                ha='center', fontsize=9, color='#1f4068')
for xi, v in zip(x + w/2, soft_vals):
    if not np.isnan(v):
        ax.text(xi, v + max(hard_vals + soft_vals) * 0.012,
                f'{v:.2f}' if v < 5 else f'{v:.1f}',
                ha='center', fontsize=9, color='#8a4500')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Metric value')
ax.set_title('Counterfactual method comparison  —  hard-constrained vs soft-penalty\n'
             '(2014 defective cohort, n = 649)', fontsize=12)
ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc', fontsize=10)
ax.grid(axis='y', alpha=0.4, linewidth=0.5)
plt.tight_layout()
fig.savefig(OUT / 'soft_penalty_flip_violation_auc.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  wrote {OUT/"soft_penalty_flip_violation_auc.png"}')

# (2) Line plot: gamma_fra vs FRA violation + AUC
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
emp_colors = {10: '#4c72b0', 100: '#dd8452', 1000: '#55a868'}
for ge in gamma_emp_values:
    sub = sweep_df[sweep_df['gamma_empirical'] == ge].sort_values('gamma_fra')
    c = emp_colors[ge]
    axes[0].plot(sub['gamma_fra'], sub['FRA_violation_rate_pct'],
                 marker='o', linewidth=2, markersize=8, color=c,
                 label=f'Soft-penalty CF, γ_emp = {ge}')
    axes[1].plot(sub['gamma_fra'], sub['AUC_dWARP_1m'],
                 marker='o', linewidth=2, markersize=8, color=c,
                 label=f'Soft-penalty, γ_emp = {ge} (1-month)')
    axes[1].plot(sub['gamma_fra'], sub['AUC_dWARP_3m'],
                 marker='s', linewidth=2, markersize=6, color=c,
                 linestyle='--', label=f'Soft-penalty, γ_emp = {ge} (3-month)')
# Reference: hard-constrained AUC at 1m and 3m on the same cohort
hard_auc_1m = comparison.loc[0, 'AUC_dWARP_1m']
hard_auc_3m = comparison.loc[0, 'AUC_dWARP_3m']
axes[1].axhline(hard_auc_1m, color='#c44e52', linestyle='-', linewidth=1.8,
                label=f'Hard-constrained CF (1-month) = {hard_auc_1m:.3f}')
axes[1].axhline(hard_auc_3m, color='#c44e52', linestyle=':', linewidth=1.8,
                label=f'Hard-constrained CF (3-month) = {hard_auc_3m:.3f}')

axes[0].set_xscale('log'); axes[1].set_xscale('log')
axes[0].set_xlabel('γ_fra  (penalty weight on FRA violation, log scale)')
axes[0].set_ylabel('FRA violation rate (%)')
axes[0].set_title('(A) FRA violation rate of soft-penalty CFs vs γ_fra')
axes[0].grid(alpha=0.4, linewidth=0.5)
axes[0].legend(framealpha=0.9, edgecolor='#cccccc', fontsize=9, title='Empirical penalty weight')
axes[1].set_xlabel('γ_fra  (penalty weight on FRA violation, log scale)')
axes[1].set_ylabel('Persistence-prediction AUC of |ΔWARP|')
axes[1].set_title('(B) Persistence AUC of soft-penalty CFs vs γ_fra\n'
                  '(red lines = hard-constrained CF baseline)')
axes[1].grid(alpha=0.4, linewidth=0.5)
axes[1].legend(framealpha=0.9, edgecolor='#cccccc', fontsize=8, ncol=1, loc='center right')
fig.suptitle('Gamma sensitivity of the soft-penalty method  —  benchmarked against hard-constrained CF',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(OUT / 'soft_penalty_gamma_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  wrote {OUT/"soft_penalty_gamma_sensitivity.png"}')

# (3) Histogram: |ΔWARP| under hard vs soft
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(hard_cfs_2014['cf_warp_abs'].values, bins=40, alpha=0.6,
        color='#1f77b4', edgecolor='white',
        label=f'Hard-constrained CF  (mean = {hard_cfs_2014["cf_warp_abs"].mean():.4f} in)')
ax.hist(soft_cfs_2014['abs_delta_WARP'].values, bins=40, alpha=0.6,
        color='#ff7f0e', edgecolor='white',
        label=f'Soft-penalty CF  (mean = {soft_cfs_2014["abs_delta_WARP"].mean():.4f} in)')
ax.set_xlabel('|ΔWARP|  —  magnitude of recommended WARP correction (in)')
ax.set_ylabel('Number of samples')
ax.set_title('Distribution of recommended |ΔWARP| under each CF method\n'
             '(2014 defective cohort, n = 649 — soft-penalty produces near-zero perturbations)',
             fontsize=11)
ax.legend(framealpha=0.95, edgecolor='#cccccc', loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.4, linewidth=0.5)
plt.tight_layout()
fig.savefig(OUT / 'soft_vs_hard_delta_warp_hist.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  wrote {OUT/"soft_vs_hard_delta_warp_hist.png"}')

# (4) Scatter: dist-to-limit vs soft |ΔWARP|, colored by persistence
def per_row_dtl(row):
    ex = []
    for f, (lo, hi) in PROJECT_LIMITS.items():
        x = row[f'x0_{f}']
        if f == 'GAGE':
            ex.append(max(0.0, lo - x, x - hi))
        else:
            ex.append(max(0.0, abs(x) - hi))
    return max(ex)

soft_cfs_2014['dist_to_limit'] = soft_cfs_2014.apply(per_row_dtl, axis=1)
# attach persistence_1m via merge
pp = panel.copy()
pp['key'] = list(zip(pp['MILE'], pp['FEET'], pp['date']))
sc = soft_cfs_2014.copy()
sc['key'] = list(zip(sc['MILE'], sc['FEET'], sc['date']))
merged = sc.merge(pp[['key','persist_1m']], on='key', how='left')

fig, ax = plt.subplots(figsize=(9, 6))
persists = merged['persist_1m'].fillna(-1).astype(int)
for label, mask, col in [
    ('Persists at 1m  (defective again)',  persists == 1, '#c44e52'),
    ('Does NOT persist',                    persists == 0, '#4c72b0'),
    ('Unknown (no future inspection)',     persists == -1, '#999999'),
]:
    if mask.sum() == 0: continue
    ax.scatter(merged.loc[mask, 'dist_to_limit'],
               merged.loc[mask, 'abs_delta_WARP'],
               s=22, alpha=0.55, color=col,
               edgecolor='none', label=f'{label}  (n={int(mask.sum())})')
ax.set_xlabel('Distance to project limit (in)  —  physical severity of defect')
ax.set_ylabel('Soft-penalty CF |ΔWARP| (in)  —  algorithm\'s severity score')
ax.set_title('Soft-penalty CF severity vs physical severity, 2014 defective cohort\n'
             '(if soft CF severity tracked physical severity, persistent points (red) would cluster top-right)',
             fontsize=11)
ax.legend(framealpha=0.95, edgecolor='#cccccc', title='1-month persistence outcome', fontsize=10)
ax.grid(alpha=0.4, linewidth=0.5)
plt.tight_layout()
fig.savefig(OUT / 'soft_penalty_dtl_vs_warp_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  wrote {OUT/"soft_penalty_dtl_vs_warp_scatter.png"}')

# ──────────────────────────────────────────────────────────────────────────
# 5) Interpretation summary
# ──────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('Phase 5 — interpretation')
print('=' * 70)

hard_row = comparison.iloc[0]
soft_row = comparison.iloc[1]

print(f'\nFlip rate            : hard {hard_row["flip_rate_pct"]:.2f}%  '
      f'vs soft {soft_row["flip_rate_pct"]:.2f}%')
print(f'FRA violation rate   : hard {hard_row["FRA_violation_rate_pct"]:.2f}%  '
      f'vs soft {soft_row["FRA_violation_rate_pct"]:.2f}%')
print(f'Empirical violation  : hard {hard_row["empirical_violation_rate_pct"]:.2f}%  '
      f'vs soft {soft_row["empirical_violation_rate_pct"]:.2f}%')
print(f'Mean MAD L1 distance : hard {hard_row["mean_mad_l1_distance"]:.3f}  '
      f'vs soft {soft_row["mean_mad_l1_distance"]:.3f}')
print(f'Mean |ΔWARP| (in)    : hard {hard_row["mean_abs_delta_WARP"]:.4f}  '
      f'vs soft {soft_row["mean_abs_delta_WARP"]:.4f}')
for h in ('1m', '3m', '6m'):
    print(f'AUC |ΔWARP| {h}        : hard {hard_row[f"AUC_dWARP_{h}"]:.3f}  '
          f'vs soft {soft_row[f"AUC_dWARP_{h}"]:.3f}')

# qualitative summary
print('\n— Interpretation —')

auc_gain_1m = soft_row['AUC_dWARP_1m'] - hard_row['AUC_dWARP_1m']
if not np.isnan(auc_gain_1m):
    if auc_gain_1m > 0.02:
        print(f'• Soft-penalty IMPROVES 1-m persistence AUC by '
              f'{auc_gain_1m*100:+.1f} pp.')
    elif auc_gain_1m < -0.02:
        print(f'• Soft-penalty DEGRADES 1-m persistence AUC by '
              f'{abs(auc_gain_1m)*100:.1f} pp.')
    else:
        print(f'• Soft-penalty AUC matches hard-constrained '
              f'({auc_gain_1m*100:+.1f} pp).')

if soft_row['FRA_violation_rate_pct'] > 0.5:
    print(f'• Soft-penalty produces {soft_row["FRA_violation_rate_pct"]:.1f}% '
          f'FRA-violating CFs at default γ_fra=1000.  Post-filtering required.')
else:
    print(f'• Soft-penalty FRA violation rate is low '
          f'({soft_row["FRA_violation_rate_pct"]:.2f}%); penalty effective.')

print(f'• γ_fra sensitivity: FRA violation '
      f'{sweep_df[sweep_df["gamma_empirical"]==100]["FRA_violation_rate_pct"].tolist()}  '
      f'at γ_fra = {sorted(sweep_df["gamma_fra"].unique())}.')

stable = (sweep_df.groupby('gamma_fra')['AUC_dWARP_1m'].std().mean() < 0.02)
print(f'• AUC stable across γ_fra: {bool(stable)}')

print('\n• Recommendation: ')
if soft_row['FRA_violation_rate_pct'] < 1.0 and auc_gain_1m > 0:
    print('  Soft-penalty is a viable alternative; can be the headline method.')
elif auc_gain_1m > 0:
    print('  Soft-penalty has predictive edge but needs post-filtering to be '
          'safety-credible. Keep hard-constrained as headline; report soft as '
          'an alternative.')
else:
    print('  Hard-constrained should remain the paper\'s headline method. '
          'Soft-penalty does not deliver a meaningful AUC gain and risks FRA '
          'non-compliance.')

print('\nAll outputs in:', OUT)
print('Done.')
