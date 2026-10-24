"""Unified recomputation of every headline number under the notebook-
hyperparameter pipeline (base_nb / recal_nb).

Outputs go to scripts/.cache/headline_numbers_nb.json so the paper-edit
script can read them with no fuss.
"""
import os, sys, pickle, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from _cf_lib import (FEATURES, CACHE, PROJECT_LIMITS, FRA_LIMITS, compute_mad)

df    = pickle.load(open(os.path.join(CACHE, 'data.pkl'),         'rb'))
panel = pickle.load(open(os.path.join(CACHE, 'persistence.pkl'), 'rb'))
mad   = compute_mad(df, train_years=(2013, 2014))
base  = pickle.load(open(os.path.join(CACHE, 'model_base_nb.pkl'),  'rb'))
recal = pickle.load(open(os.path.join(CACHE, 'model_recal_nb.pkl'), 'rb'))

cf = {}
for y in (2014, 2015, 2016):
    cf[('base',  y)] = pickle.load(open(os.path.join(CACHE, f'cf_{y}_base_nb.pkl'),  'rb'))
    cf[('recal', y)] = pickle.load(open(os.path.join(CACHE, f'cf_{y}_recal_nb.pkl'), 'rb'))

FRA_LO = np.array([FRA_LIMITS[f][0] for f in FEATURES])
FRA_HI = np.array([FRA_LIMITS[f][1] for f in FEATURES])
warp_idx = FEATURES.index('WARP_62')

def dtl_row(row):
    ex = []
    for f, (lo, hi) in PROJECT_LIMITS.items():
        x = row[f]
        if f == 'GAGE': ex.append(max(0.0, lo-x, x-hi))
        else:           ex.append(max(0.0, abs(x)-hi))
    return max(ex)

def add_predictors(cf_df, year):
    pp = panel[panel['year'] == year].copy()
    pp['key'] = list(zip(pp['MILE'], pp['FEET'], pp['date']))
    cfc = cf_df.copy()
    cfc['key'] = list(zip(cfc['MILE'], cfc['FEET'], cfc['date']))
    m = pp.merge(cfc[['key', 'cf_warp_abs']], on='key', how='inner')
    m['dist_to_limit'] = m.apply(dtl_row, axis=1)
    grp = m.groupby(['date', 'MILE'])
    for f in FEATURES:
        m[f'tqi_{f}'] = grp[f].transform('std')
    m['local_tqi'] = m[[f'tqi_{f}' for f in FEATURES]].mean(axis=1)
    return m

def auc_of(merged, hcol, predictor):
    sub = merged[merged[hcol].notna()]
    if len(sub) < 20:
        return float('nan'), int(len(sub))
    y = sub[hcol].astype(int).values
    x = sub[predictor].fillna(sub[predictor].median()).values
    return float(roc_auc_score(y, x)), int(len(sub))

# ============================================================
# 1) per-cohort AUC table (base + recal × 3 cohorts × 3 horizons × 3 predictors)
# ============================================================
auc_table = {}
for tag in ('base', 'recal'):
    for year in (2014, 2015, 2016):
        merged = add_predictors(cf[(tag, year)], year)
        for h, hlbl in (('persist_1m', '1m'),
                        ('persist_3m', '3m'),
                        ('persist_6m', '6m')):
            for pred in ('cf_warp_abs', 'dist_to_limit', 'local_tqi'):
                auc, n = auc_of(merged, h, pred)
                auc_table[(tag, year, hlbl, pred)] = (auc, n)

print('=== per-cohort predictor AUCs (notebook-hyperparam pipeline) ===')
for tag in ('base', 'recal'):
    print(f'\n--- model = {tag} ---')
    print(f'{"cohort":>6s} {"horizon":>8s} {"CF":>8s} {"Dist":>8s} {"TQI":>8s}   {"n":>5s}')
    for year in (2014, 2015, 2016):
        for hlbl in ('1m', '3m', '6m'):
            cf_auc,n  = auc_table[(tag, year, hlbl, 'cf_warp_abs')]
            dt_auc,_  = auc_table[(tag, year, hlbl, 'dist_to_limit')]
            tqi_auc,_ = auc_table[(tag, year, hlbl, 'local_tqi')]
            print(f'{year:>6d} {hlbl:>8s} {cf_auc:>8.3f} {dt_auc:>8.3f} {tqi_auc:>8.3f}   {n:>5d}')

# ============================================================
# 2) Classifier AUC per year, under both models
# ============================================================
clf_auc = {}
for tag, model in (('base', base), ('recal', recal)):
    for year in (2014, 2015, 2016):
        sub = df[df['year'] == year]
        p = model.predict_proba(sub[FEATURES].values)[:, 1]
        clf_auc[(tag, year)] = float(roc_auc_score(sub['defect'].values, p))

print('\n=== classifier held-out AUC ===')
print(f'{"year":>4s} {"base":>10s} {"recal":>10s}')
for year in (2014, 2015, 2016):
    print(f'{year:>4d} {clf_auc[("base",year)]:>10.4f} {clf_auc[("recal",year)]:>10.4f}')

# ============================================================
# 3) Spearman ρ between cf_warp_abs and dist_to_limit per cohort
# ============================================================
spearman = {}
for tag in ('base', 'recal'):
    for year in (2014, 2015, 2016):
        merged = add_predictors(cf[(tag, year)], year)
        rho, _ = spearmanr(merged['cf_warp_abs'], merged['dist_to_limit'])
        spearman[(tag, year)] = float(rho)

print('\n=== Spearman ρ (CF |ΔWARP| vs dist-to-limit) ===')
print(f'{"cohort":>6s} {"base":>10s} {"recal":>10s}')
for year in (2014, 2015, 2016):
    print(f'{year:>6d} {spearman[("base",year)]:>10.3f} {spearman[("recal",year)]:>10.3f}')

# ============================================================
# 4) Transfer matrix — flip rates (sign-corrected, FRA-clipped)
# ============================================================
def p1_flip(model_, cf_ref, df_test):
    md = cf_ref[[f'd_{f}' for f in FEATURES]].mean().values
    X0 = df_test[FEATURES].values.astype(float)
    deltas = np.tile(md, (len(X0), 1))
    pos = X0[:, warp_idx] > 0
    deltas[ pos, warp_idx] = -abs(md[warp_idx])
    deltas[~pos, warp_idx] = +abs(md[warp_idx])
    Xs = np.clip(X0 + deltas, FRA_LO, FRA_HI)
    p0, p1 = model_.predict(X0), model_.predict(Xs)
    return float(((p0==1)&(p1==0)).sum() / max(1, (p0==1).sum()) * 100)

def p2_flip(model_, cf_ref, df_test):
    Xref = cf_ref[[f'x0_{f}' for f in FEATURES]].values
    Dref = cf_ref[[f'd_{f}' for f in FEATURES]].values
    X0 = df_test[FEATURES].values.astype(float)
    Xn = X0/mad; Rn = Xref/mad
    dist = np.sqrt(((Xn[:,None,:] - Rn[None,:,:])**2).sum(axis=2))
    nn = dist.argmin(axis=1)
    deltas = Dref[nn].copy()
    pos = X0[:, warp_idx] > 0
    deltas[ pos, warp_idx] = -np.abs(deltas[ pos, warp_idx])
    deltas[~pos, warp_idx] = +np.abs(deltas[~pos, warp_idx])
    Xs = np.clip(X0 + deltas, FRA_LO, FRA_HI)
    p0, p1 = model_.predict(X0), model_.predict(Xs)
    return float(((p0==1)&(p1==0)).sum() / max(1, (p0==1).sum()) * 100)

P1 = {}; P2 = {}
print('\n=== Transfer matrix (base_nb model) ===')
print(f'  ref \\ test    {"2014":>8s} {"2015":>8s} {"2016":>8s}')
for ref_year in (2014, 2015, 2016):
    print(f'  ref {ref_year}', end='')
    for test_year in (2014, 2015, 2016):
        df_test = df[(df['year']==test_year) & (df['defect']==1)]
        p1 = p1_flip(base, cf[('base', ref_year)], df_test)
        p2 = p2_flip(base, cf[('base', ref_year)], df_test)
        P1[('base', ref_year, test_year)] = p1
        P2[('base', ref_year, test_year)] = p2
        print(f'  P1={p1:4.1f} P2={p2:4.1f}', end='')
    print()

# ============================================================
# 5) Recal flip rate comparison on 2016
# ============================================================
cohort_2016 = df[(df['year']==2016) & (df['defect']==1)]

# Stale baseline: base + 2014 ref
stale_p1 = p1_flip(base, cf[('base', 2014)], cohort_2016)
stale_p2 = p2_flip(base, cf[('base', 2014)], cohort_2016)

# Annual recal: recal + 2015 ref (proper workflow)
recal_p1 = p1_flip(recal, cf[('recal', 2015)], cohort_2016)
recal_p2 = p2_flip(recal, cf[('recal', 2015)], cohort_2016)

# Ablation: recal model + 2014 ref (model only)
abl_p1 = p1_flip(recal, cf[('recal', 2014)], cohort_2016)
abl_p2 = p2_flip(recal, cf[('recal', 2014)], cohort_2016)

print('\n=== 2016 cohort flip rates ===')
print(f'  Stale (base + 2014 ref)        : P1 {stale_p1:5.2f}%   P2 {stale_p2:5.2f}%')
print(f'  Ablation (recal + 2014 ref)    : P1 {abl_p1:5.2f}%   P2 {abl_p2:5.2f}%')
print(f'  Annual recal (recal + 2015 ref): P1 {recal_p1:5.2f}%   P2 {recal_p2:5.2f}%')

# ============================================================
# 6) Severity tier persistence (for validation_B-style figure)
# ============================================================
def tier_stats(merged):
    bins = [0, 0.03, 0.08, 0.15, np.inf]
    labels = ['Marginal', 'Moderate', 'Severe', 'Critical']
    merged = merged.copy()
    merged['tier'] = pd.cut(merged['cf_warp_abs'], bins=bins, labels=labels)
    out = {}
    for label in labels:
        sub = merged[merged['tier'] == label]
        n = len(sub)
        if n == 0:
            out[label] = {'n':0}
            continue
        p3 = sub['persist_3m'].mean() * 100 if 'persist_3m' in sub else None
        p6 = sub['persist_6m'].mean() * 100 if 'persist_6m' in sub else None
        out[label] = {'n': int(n), 'p3': float(p3), 'p6': float(p6)}
    return out

print('\n=== Severity-tier persistence rates (2014, base_nb) ===')
tiers_2014 = tier_stats(add_predictors(cf[('base', 2014)], 2014))
for label, s in tiers_2014.items():
    if s['n'] == 0:
        print(f'  {label:10s}  n=0')
    else:
        print(f'  {label:10s}  n={s["n"]:4d}  3m={s["p3"]:5.1f}%  6m={s["p6"]:5.1f}%')

# ============================================================
# 7) Dump everything to json for downstream consumption
# ============================================================
out = {
    'auc_per_cohort': {f'{k[0]}_{k[1]}_{k[2]}_{k[3]}': v[0]
                        for k, v in auc_table.items()},
    'auc_per_cohort_n': {f'{k[0]}_{k[1]}_{k[2]}_{k[3]}': v[1]
                          for k, v in auc_table.items()},
    'classifier_auc': {f'{k[0]}_{k[1]}': v for k, v in clf_auc.items()},
    'spearman_cf_vs_dtl': {f'{k[0]}_{k[1]}': v for k, v in spearman.items()},
    'transfer_P1': {f'{k[0]}_{k[1]}_{k[2]}': v for k, v in P1.items()},
    'transfer_P2': {f'{k[0]}_{k[1]}_{k[2]}': v for k, v in P2.items()},
    'flip_2016': {
        'stale_p1': stale_p1, 'stale_p2': stale_p2,
        'ablation_p1': abl_p1, 'ablation_p2': abl_p2,
        'annual_recal_p1': recal_p1, 'annual_recal_p2': recal_p2,
    },
    'tiers_2014_base_nb': {k: v for k, v in tiers_2014.items()},
}
json_path = os.path.join(CACHE, 'headline_numbers_nb.json')
with open(json_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nWrote: {json_path}')
