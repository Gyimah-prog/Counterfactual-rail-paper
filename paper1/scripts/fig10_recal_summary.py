"""Fig #10 — Recalibration improves both flip rate AND CF AUC.

Replaces the earlier (incorrect) 'two faces' framing.  Under the notebook's
sign-corrected protocol, both metrics improve under recalibration; just by
very different magnitudes.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from _cf_lib import FEATURES, ROOT, CACHE, FRA_LIMITS, compute_mad

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
mad   = compute_mad(df, train_years=(2013, 2014))

cf_2014_base  = pickle.load(open(os.path.join(CACHE, 'cf_2014_base.pkl'),  'rb'))
cf_2014_recal = pickle.load(open(os.path.join(CACHE, 'cf_2014_recal.pkl'), 'rb'))
cf_2015_recal = pickle.load(open(os.path.join(CACHE, 'cf_2015_recal.pkl'), 'rb'))
cf_2016_base  = pickle.load(open(os.path.join(CACHE, 'cf_2016_base.pkl'),  'rb'))
cf_2016_recal = pickle.load(open(os.path.join(CACHE, 'cf_2016_recal.pkl'), 'rb'))

FRA_LO = np.array([FRA_LIMITS[f][0] for f in FEATURES])
FRA_HI = np.array([FRA_LIMITS[f][1] for f in FEATURES])
warp_idx = FEATURES.index('WARP_62')

def p1_flip(model_, cf_ref, df_test):
    md = cf_ref[[f'd_{f}' for f in FEATURES]].mean().values
    X0 = df_test[FEATURES].values.astype(float)
    deltas = np.tile(md, (len(X0), 1))
    pos = X0[:, warp_idx] > 0
    deltas[ pos, warp_idx] = -abs(md[warp_idx])
    deltas[~pos, warp_idx] = +abs(md[warp_idx])
    Xs = np.clip(X0 + deltas, FRA_LO, FRA_HI)
    p0, p1 = model_.predict(X0), model_.predict(Xs)
    return ((p0 == 1) & (p1 == 0)).sum() / max(1, (p0==1).sum()) * 100

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
    return ((p0 == 1) & (p1 == 0)).sum() / max(1, (p0==1).sum()) * 100

cohort_2016 = df[(df['year']==2016) & (df['defect']==1)]
# Each pipeline uses its own in-distribution reference cohort:
#   base  pipeline: 2014 CFs (2014 is in-distribution for the 2013-14 model)
#   recal pipeline: 2015 CFs (2015 is in-distribution for the 2014-15 model)
p1_base  = p1_flip(base,  cf_2014_base,  cohort_2016)
p1_recal = p1_flip(recal, cf_2015_recal, cohort_2016)
p2_base  = p2_flip(base,  cf_2014_base,  cohort_2016)
p2_recal = p2_flip(recal, cf_2015_recal, cohort_2016)

def auc_2016(cf_df):
    pp = panel[panel['year'] == 2016].copy()
    pp['key'] = list(zip(pp['MILE'], pp['FEET'], pp['date']))
    cfc = cf_df.copy()
    cfc['key'] = list(zip(cfc['MILE'], cfc['FEET'], cfc['date']))
    merged = pp.merge(cfc[['key', 'cf_warp_abs']], on='key', how='inner')
    sub = merged[merged['persist_1m'].notna()]
    return roc_auc_score(sub['persist_1m'].astype(int).values,
                          sub['cf_warp_abs'].values), len(sub)

auc_base,  _ = auc_2016(cf_2016_base)
auc_recal, _ = auc_2016(cf_2016_recal)

print(f'Flip rate 2016:  P1 base={p1_base:.1f}%  recal={p1_recal:.1f}%  (+{p1_recal-p1_base:.1f}pp)')
print(f'                 P2 base={p2_base:.1f}%  recal={p2_recal:.1f}%  (+{p2_recal-p2_base:.1f}pp)')
print(f'AUC 2016:        base={auc_base:.3f}  recal={auc_recal:.3f}  (+{auc_recal-auc_base:.3f})')

# ─── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A — Flip rate
ax = axes[0]
groups = ['Protocol 1\n(mean Δ)', 'Protocol 2\n(NN Δ)']
base_vals  = [p1_base,  p2_base]
recal_vals = [p1_recal, p2_recal]
x = np.arange(len(groups)); w = 0.36
ax.bar(x - w/2, base_vals,  width=w, color='#1f77b4', edgecolor='white',
       label='Base model + 2014 reference CFs  (stale baseline)')
ax.bar(x + w/2, recal_vals, width=w, color='#ff7f0e', edgecolor='white',
       label='Recal model + 2015 reference CFs  (annual-recal workflow)')
for xi, v in zip(x - w/2, base_vals):
    ax.text(xi, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10, color='#1f4068')
for xi, v in zip(x + w/2, recal_vals):
    ax.text(xi, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10, color='#8a4500')
for j, (b, r) in enumerate(zip(base_vals, recal_vals)):
    gain = r - b
    ax.annotate(f'+{gain:.1f} pp',
                xy=(x[j], max(b, r) + 1.3),
                ha='center', fontsize=9, color='#2a7d2a', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10.5)
ax.set_ylabel('Flip rate on 2016 cohort (%)')
ax.set_ylim(0, max(base_vals + recal_vals) * 1.4)
ax.set_title('(A) Prescription success — flip rate roughly doubles', fontsize=11.5, pad=8)
ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc', fontsize=9.5)
ax.grid(axis='y', alpha=0.4, linewidth=0.5)

# Panel B — CF AUC
ax = axes[1]
labels = ['Base model\n(2013–14)', 'Recal model\n(2014–15)']
vals   = [auc_base, auc_recal]
ax.bar([0, 1], vals, width=0.5, color=['#1f77b4', '#ff7f0e'], edgecolor='white')
for xi, v in zip([0, 1], vals):
    ax.text(xi, v + 0.012, f'{v:.3f}', ha='center', fontsize=12,
            fontweight='bold', color='#444441')
ax.annotate(f'+{auc_recal-auc_base:.3f}\n(+{(auc_recal-auc_base)*100:.0f} pp AUC)',
            xy=(0.5, max(vals) + 0.05), ha='center', fontsize=10.5,
            color='#2a7d2a', fontweight='bold')
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel('AUC for 1-month persistence prediction')
ax.set_ylim(0.45, 1.0)
ax.set_title('(B) Severity-as-predictor — large gain', fontsize=11.5, pad=8)
ax.axhline(0.5, linestyle=':', color='#aaa', linewidth=0.8)
ax.axhline(0.894, linestyle='--', color='#c44e52', linewidth=1.5)
ax.text(1.5, 0.894 + 0.005, 'dist-to-limit baseline = 0.894',
        ha='right', fontsize=9, color='#c44e52', style='italic')
ax.grid(axis='y', alpha=0.4, linewidth=0.5)

fig.suptitle('Annual recalibration of model + reference CFs improves both operational properties on the 2016 cohort\n'
             '(both pipelines use their own in-distribution reference cohort)',
             fontsize=12, fontweight='bold', y=1.04)

plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'recal_summary.{ext}'),
                dpi=300, bbox_inches='tight')
plt.close()
print('Wrote: figures/paper_ready/recal_summary.{png,pdf}')

# Clean up the misleading earlier figures
for stale in ['recalibration_dual_nature', 'recalibration']:
    for ext in ('png', 'pdf'):
        p = os.path.join(OUT, f'{stale}.{ext}')
        if os.path.exists(p):
            os.remove(p)
            print(f'Removed misleading: {stale}.{ext}')
