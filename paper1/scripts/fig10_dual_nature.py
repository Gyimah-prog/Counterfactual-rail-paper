"""Fig #10 — The dual nature of recalibration.

Two panels showing the same recalibration moves CF quality in OPPOSITE
directions for two different operational properties:

  Panel A (prescription quality):
    Flip rate on 2016 cohort under (base + 2014 ref CFs) vs
    (recal + 2014 ref CFs) — using each model's own reference.
    P1 and P2 protocols shown.
    → Flip rate DECREASES under recal (recal model is more discriminating).

  Panel B (ranking quality):
    AUC of CF |ΔWARP| for predicting 2016 persistence,
    base model vs recal model.
    → AUC INCREASES under recal (better-calibrated severity).
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from _cf_lib import (FEATURES, ROOT, CACHE, FRA_LIMITS, PROJECT_LIMITS, compute_mad)

OUT = os.path.join(ROOT, 'figures', 'paper_ready')

sns.set_theme(style='whitegrid', context='paper', font_scale=1.15)
sns.set_palette('colorblind')
plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica'],
    'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
})

df    = pickle.load(open(os.path.join(CACHE, 'data.pkl'), 'rb'))
base  = pickle.load(open(os.path.join(CACHE, 'model_base.pkl'), 'rb'))
recal = pickle.load(open(os.path.join(CACHE, 'model_recal.pkl'), 'rb'))
panel = pickle.load(open(os.path.join(CACHE, 'persistence.pkl'), 'rb'))
cf_2014_base  = pickle.load(open(os.path.join(CACHE, 'cf_2014_base.pkl'), 'rb'))
cf_2014_recal = pickle.load(open(os.path.join(CACHE, 'cf_2014_recal.pkl'), 'rb'))
cf_2016_base  = pickle.load(open(os.path.join(CACHE, 'cf_2016_base.pkl'), 'rb'))
cf_2016_recal = pickle.load(open(os.path.join(CACHE, 'cf_2016_recal.pkl'), 'rb'))
mad           = compute_mad(df, train_years=(2013, 2014))

FRA_LO = np.array([FRA_LIMITS[f][0] for f in FEATURES])
FRA_HI = np.array([FRA_LIMITS[f][1] for f in FEATURES])
warp_idx = FEATURES.index('WARP_62')

# ─── Panel A — flip rates under P1 and P2 ─────────────────────────────────
def p1_flip(model_, cf_ref, df_test):
    md = cf_ref[[f'd_{f}' for f in FEATURES]].mean().values
    X0 = df_test[FEATURES].values.astype(float)
    deltas = np.tile(md, (len(X0), 1))
    pos = X0[:, warp_idx] > 0
    deltas[ pos, warp_idx] = -abs(md[warp_idx])
    deltas[~pos, warp_idx] = +abs(md[warp_idx])
    Xs = np.clip(X0 + deltas, FRA_LO, FRA_HI)
    pred0 = model_.predict(X0); pred1 = model_.predict(Xs)
    flipped = (pred0 == 1) & (pred1 == 0)
    return flipped.sum() / max(1, (pred0==1).sum()) * 100

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
    pred0 = model_.predict(X0); pred1 = model_.predict(Xs)
    flipped = (pred0 == 1) & (pred1 == 0)
    return flipped.sum() / max(1, (pred0==1).sum()) * 100

cohort_2016 = df[(df['year']==2016) & (df['defect']==1)]
p1_base  = p1_flip(base,  cf_2014_base,  cohort_2016)
p1_recal = p1_flip(recal, cf_2014_recal, cohort_2016)
p2_base  = p2_flip(base,  cf_2014_base,  cohort_2016)
p2_recal = p2_flip(recal, cf_2014_recal, cohort_2016)
print(f'Flip rates on 2016 cohort:')
print(f'  P1 base  = {p1_base:5.1f}%   |   P1 recal = {p1_recal:5.1f}%')
print(f'  P2 base  = {p2_base:5.1f}%   |   P2 recal = {p2_recal:5.1f}%')

# ─── Panel B — CF |ΔWARP| AUC for 2016 persistence ────────────────────────
def cohort_cf_auc(cf_df, year=2016):
    pp = panel[panel['year'] == year].copy()
    pp['key'] = list(zip(pp['MILE'], pp['FEET'], pp['date']))
    cfc = cf_df.copy()
    cfc['key'] = list(zip(cfc['MILE'], cfc['FEET'], cfc['date']))
    merged = pp.merge(cfc[['key', 'cf_warp_abs']], on='key', how='inner')
    sub = merged[merged['persist_1m'].notna()]
    y = sub['persist_1m'].astype(int).values
    return roc_auc_score(y, sub['cf_warp_abs'].values), len(sub)

auc_base,  n_b = cohort_cf_auc(cf_2016_base)
auc_recal, n_r = cohort_cf_auc(cf_2016_recal)
print(f'CF AUC on 2016 cohort:')
print(f'  base  = {auc_base:.3f} (n={n_b})')
print(f'  recal = {auc_recal:.3f} (n={n_r})')

# ─── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ----- Panel A: Flip rate -----
ax = axes[0]
groups = ['Protocol 1\n(mean Δ)', 'Protocol 2\n(nearest-neighbour Δ)']
base_vals  = [p1_base,  p2_base]
recal_vals = [p1_recal, p2_recal]
x = np.arange(len(groups))
w = 0.36
b1 = ax.bar(x - w/2, base_vals,  width=w, color='#1f77b4', edgecolor='white', label='Base model + 2014 ref CFs')
b2 = ax.bar(x + w/2, recal_vals, width=w, color='#ff7f0e', edgecolor='white', label='Recal model + 2014 ref CFs')
for xi, v in zip(x - w/2, base_vals):
    ax.text(xi, v + 1.5, f'{v:.1f}%', ha='center', fontsize=10, color='#1f4068')
for xi, v in zip(x + w/2, recal_vals):
    ax.text(xi, v + 1.5, f'{v:.1f}%', ha='center', fontsize=10, color='#8a4500')
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10.5)
ax.set_ylabel('Flip rate on 2016 cohort (%)')
ax.set_ylim(0, max(base_vals + recal_vals) * 1.3)
ax.set_title('(A) Recommendation success ↓\nDoes applying the CF flip the model?')
ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc', fontsize=9.5)
ax.grid(axis='y', alpha=0.4, linewidth=0.5)
# Add arrow showing direction
ax.annotate('', xy=(0.5, max(base_vals)*0.55), xytext=(0.5, max(base_vals)*0.85),
            arrowprops=dict(arrowstyle='->', color='#c44e52', lw=2.5),
            zorder=10)
ax.text(0.62, max(base_vals)*0.72, 'recalibration\nLOWERS flip rate',
        fontsize=9.5, color='#c44e52', style='italic')

# ----- Panel B: CF AUC -----
ax = axes[1]
labels = ['Base model\n(2013–14)', 'Recalibrated model\n(2014–15)']
vals   = [auc_base, auc_recal]
colors = ['#1f77b4', '#ff7f0e']
x = np.arange(len(labels))
ax.bar(x, vals, width=0.5, color=colors, edgecolor='white')
for xi, v in zip(x, vals):
    ax.text(xi, v + 0.012, f'{v:.3f}', ha='center', fontsize=11.5,
            fontweight='bold', color='#444441')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel('AUC for 1-month persistence prediction')
ax.set_ylim(0.45, 1.0)
ax.set_title('(B) Severity-as-predictor ↑\nDoes CF |ΔWARP| rank persistence?')
ax.axhline(0.5, linestyle=':', color='#aaa', linewidth=0.8)
# dist-to-limit reference
dtl_ref = 0.894
ax.axhline(dtl_ref, linestyle='--', color='#c44e52', linewidth=1.5)
ax.text(1.5, dtl_ref + 0.005, f'dist-to-limit baseline = {dtl_ref:.3f}',
        ha='right', fontsize=9, color='#c44e52', style='italic')
ax.grid(axis='y', alpha=0.4, linewidth=0.5)
# Add arrow showing direction
ax.annotate('', xy=(0.5, 0.78), xytext=(0.5, 0.65),
            arrowprops=dict(arrowstyle='->', color='#2a7d2a', lw=2.5),
            zorder=10)
ax.text(0.62, 0.72, 'recalibration\nRAISES CF AUC',
        fontsize=9.5, color='#2a7d2a', style='italic')

fig.suptitle("The two faces of model recalibration  —  evaluated on the 2016 cohort\n"
             "Same recalibration: makes the prescription harder to satisfy, but the severity score more predictive",
             fontsize=12, fontweight='bold', y=1.04)

plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'recalibration_dual_nature.{ext}'),
                dpi=300, bbox_inches='tight')
plt.close()
print('Wrote: figures/paper_ready/recalibration_dual_nature.{png,pdf}')
