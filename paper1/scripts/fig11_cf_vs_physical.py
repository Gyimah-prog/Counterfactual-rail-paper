"""Fig #11 — Why CF AUC degrades / why recalibration restores it.

The 2014 base CFs track physical severity tightly. The 2016 base CFs
DON'T — the CF score scatters away from physical severity, which is why
its AUC for persistence (which IS predicted by physical severity)
collapses. The 2016 recal CFs restore the correlation.

Six panels: 3 cohorts × 2 models. Scatter of CF |ΔWARP| vs dist-to-limit
(physical severity), with Spearman ρ annotated.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from _cf_lib import FEATURES, ROOT, CACHE, PROJECT_LIMITS

OUT = os.path.join(ROOT, 'figures', 'paper_ready')

sns.set_theme(style='whitegrid', context='paper', font_scale=1.15)
plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica'],
    'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
})

df = pickle.load(open(os.path.join(CACHE,'data.pkl'),'rb'))

cf_base = {
    2014: pickle.load(open(os.path.join(CACHE, 'cf_2014_base.pkl'),  'rb')),
    2015: pickle.load(open(os.path.join(CACHE, 'cf_2015_base.pkl'),  'rb')),
    2016: pickle.load(open(os.path.join(CACHE, 'cf_2016_base.pkl'),  'rb')),
}
cf_recal = {
    2014: pickle.load(open(os.path.join(CACHE, 'cf_2014_recal.pkl'), 'rb')),
    2015: pickle.load(open(os.path.join(CACHE, 'cf_2015_recal.pkl'), 'rb')),
    2016: pickle.load(open(os.path.join(CACHE, 'cf_2016_recal.pkl'), 'rb')),
}

def dtl(row):
    ex = []
    for f, (lo, hi) in PROJECT_LIMITS.items():
        x = row[f]
        if f == 'GAGE':
            ex.append(max(0.0, lo - x, x - hi))
        else:
            ex.append(max(0.0, abs(x) - hi))
    return max(ex)

# Merge with original samples to get dist-to-limit per CF
def add_dtl(cf_df):
    df_def = df[df['defect'] == 1][['MILE','FEET','date'] + FEATURES].copy()
    df_def['key'] = list(zip(df_def['MILE'], df_def['FEET'], df_def['date']))
    cf_df = cf_df.copy()
    cf_df['key'] = list(zip(cf_df['MILE'], cf_df['FEET'], cf_df['date']))
    merged = cf_df.merge(df_def, on='key', how='inner', suffixes=('', '_orig'))
    merged['dist_to_limit'] = merged.apply(lambda r: dtl({f: r[f] for f in FEATURES}), axis=1)
    return merged

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
xmax = 0
for d in list(cf_base.values()) + list(cf_recal.values()):
    m = add_dtl(d)
    xmax = max(xmax, m['dist_to_limit'].quantile(0.99))
ymax = 0
for d in list(cf_base.values()) + list(cf_recal.values()):
    ymax = max(ymax, d['cf_warp_abs'].quantile(0.99))

year_colors = {2014: '#4c72b0', 2015: '#dd8452', 2016: '#c44e52'}

for j, year in enumerate([2014, 2015, 2016]):
    for i, (label, dct) in enumerate([('Base model\n(train 2013–14)', cf_base),
                                       ('Recal model\n(train 2014–15)', cf_recal)]):
        ax = axes[i, j]
        m = add_dtl(dct[year])
        x_v = m['dist_to_limit'].values
        y_v = m['cf_warp_abs'].values
        rho, p = spearmanr(x_v, y_v)
        ax.scatter(x_v, y_v, s=10, alpha=0.45, color=year_colors[year],
                   edgecolor='none')
        # robust trend line
        if len(x_v) > 5:
            # use a simple polynomial fit for visualization
            try:
                coeffs = np.polyfit(x_v, y_v, 1)
                xline = np.linspace(0, xmax, 100)
                ax.plot(xline, np.polyval(coeffs, xline), color='black',
                        linewidth=1.2, linestyle='--', alpha=0.7)
            except Exception:
                pass
        ax.text(0.04, 0.96,
                f'Spearman ρ = {rho:.2f}\nn = {len(x_v)}',
                transform=ax.transAxes, va='top', fontsize=10,
                color='#444441',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', linewidth=0.6))
        if i == 0:
            ax.set_title(f'Cohort {year}', fontsize=12)
        if j == 0:
            ax.set_ylabel(f'{label}\n\nCF |ΔWARP| (in)', fontsize=10.5)
        if i == 1:
            ax.set_xlabel('Dist-to-limit (in)  —  physical severity', fontsize=10.5)
        ax.set_xlim(0, xmax * 1.05)
        ax.set_ylim(0, ymax * 1.05)
        ax.grid(alpha=0.4, linewidth=0.5)
        sns.despine(ax=ax)

fig.suptitle('CF severity vs physical severity, by cohort and model\n'
             '(tight diagonal = CF tracks physics → high persistence AUC;  scattered = CF decoupled from physics → AUC drops)',
             fontsize=12, fontweight='bold', y=1.01)

plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'cf_vs_physical.{ext}'),
                dpi=300, bbox_inches='tight')
plt.close()
print('Wrote: figures/paper_ready/cf_vs_physical.{png,pdf}')

# Remove old (wrong) saturation diagnostic
for ext in ('png','pdf'):
    p = os.path.join(OUT, f'cf_saturation.{ext}')
    if os.path.exists(p):
        os.remove(p); print(f'Removed: cf_saturation.{ext}')
