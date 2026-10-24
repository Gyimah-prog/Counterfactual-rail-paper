"""Fig #8 — CF severity vs Dist-to-limit vs Local TQI as persistence predictors.

For each (cohort year × prediction horizon), compute AUC of each single
predictor against future-defect persistence.  AUC is computed directly on
the raw predictor values (no logistic fit needed for 1-D predictors), so
there is no in-sample / out-of-sample confound.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from _cf_lib import FEATURES, ROOT, CACHE, PROJECT_LIMITS

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
panel = pickle.load(open(os.path.join(CACHE, 'persistence.pkl'), 'rb'))
cf_by_year = {
    2014: pickle.load(open(os.path.join(CACHE, 'cf_2014_base.pkl'), 'rb')),
    2015: pickle.load(open(os.path.join(CACHE, 'cf_2015_base.pkl'), 'rb')),
    2016: pickle.load(open(os.path.join(CACHE, 'cf_2016_base.pkl'), 'rb')),
}

# Per-row distance-to-limit (max exceedance across features)
def dtl(row):
    ex = []
    for f, (lo, hi) in PROJECT_LIMITS.items():
        x = row[f]
        if f == 'GAGE':
            ex.append(max(0.0, lo - x, x - hi))
        else:
            ex.append(max(0.0, abs(x) - hi))
    return max(ex)

def build_cohort(cf_cohort, year):
    """Merge CF severity, dist-to-limit, and local TQI with persistence labels."""
    pp = panel[panel['year'] == year].copy()
    pp['key'] = list(zip(pp['MILE'], pp['FEET'], pp['date']))
    cfc = cf_cohort.copy()
    cfc['key'] = list(zip(cfc['MILE'], cfc['FEET'], cfc['date']))
    pp = pp[pp['key'].isin(set(cfc['key']))].copy()
    pp['dist_to_limit'] = pp.apply(dtl, axis=1)
    grp = pp.groupby(['date', 'MILE'])
    for f in FEATURES:
        pp[f'tqi_{f}'] = grp[f].transform('std')
    pp['local_tqi'] = pp[[f'tqi_{f}' for f in FEATURES]].mean(axis=1)
    return pp.merge(cfc[['key', 'cf_warp_abs']], on='key', how='inner')

YEARS    = [2014, 2015, 2016]
HORIZONS = [('persist_1m', '1-month'),
            ('persist_3m', '3-month'),
            ('persist_6m', '6-month')]
PREDS    = [('cf_warp_abs',   'CF |ΔWARP|',    '#d62728', 'o'),
            ('dist_to_limit', 'Dist-to-limit', '#1f77b4', 's'),
            ('local_tqi',     'Local TQI',     '#7f7f7f', '^')]

# Compute AUCs
results = {}
sample_counts = {}
for y in YEARS:
    tbl = build_cohort(cf_by_year[y], y)
    for (h_col, h_name) in HORIZONS:
        valid = tbl[h_col].notna()
        sub = tbl[valid]
        for (p_col, p_name, _, _) in PREDS:
            y_true = sub[h_col].astype(int).values
            x_val  = sub[p_col].values
            if len(np.unique(y_true)) < 2 or len(y_true) < 20:
                results[(y, h_name, p_name)] = (np.nan, len(y_true))
            else:
                auc = roc_auc_score(y_true, x_val)
                results[(y, h_name, p_name)] = (auc, len(y_true))
        sample_counts[(y, h_name)] = len(sub)
        print(f'  cohort {y} | {h_name}: n={len(sub)}  '
              f'CF={results[(y,h_name,"CF |ΔWARP|")][0]:.3f}  '
              f'Dist={results[(y,h_name,"Dist-to-limit")][0]:.3f}  '
              f'TQI={results[(y,h_name,"Local TQI")][0]:.3f}')

# ─── Plot: 1×3 facets, one per horizon ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
x = np.arange(len(YEARS))

for ax, (h_col, h_name) in zip(axes, HORIZONS):
    for (p_col, p_name, color, marker) in PREDS:
        vals = [results[(y, h_name, p_name)][0] for y in YEARS]
        ax.plot(x, vals, marker=marker, markersize=11, linewidth=2.5,
                color=color, label=p_name)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.annotate(f'{v:.3f}', (xi, v),
                            textcoords='offset points', xytext=(7, 5),
                            fontsize=9, color=color)
    # n labels under x-axis
    for xi, y in zip(x, YEARS):
        n = sample_counts[(y, h_name)]
        ax.annotate(f'n={n}', xy=(xi, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -32), textcoords='offset points',
                    ha='center', fontsize=8.5, color='#888')
    ax.axhline(0.5, linestyle=':', color='#aaa', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in YEARS], fontsize=11)
    ax.set_xlabel('Cohort year')
    ax.set_ylim(0.25, 1.0)
    ax.set_title(f'{h_name} persistence', fontsize=11.5)
    ax.grid(axis='y', alpha=0.4, linewidth=0.5)

axes[0].set_ylabel('AUC')
axes[0].legend(loc='lower left', framealpha=0.95, edgecolor='#cccccc')

fig.suptitle('Counterfactual severity vs traditional predictors  —  '
             'persistence prediction by cohort and horizon\n'
             '(AUC computed directly on raw predictor values; '
             'no logistic fit, no in-sample bias)',
             fontsize=12, fontweight='bold', y=1.04)

plt.subplots_adjust(bottom=0.18)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'predictor_comparison.{ext}'),
                dpi=300, bbox_inches='tight')
plt.close()
print('\nWrote: figures/paper_ready/predictor_comparison.{png,pdf}')
