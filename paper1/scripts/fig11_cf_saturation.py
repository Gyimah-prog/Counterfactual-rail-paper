"""Fig #11 — CF |ΔWARP| distribution per cohort (saturation diagnostic).

Shows that under the frozen base model, the CF |ΔWARP| distribution compresses
as cohort drifts forward (2014 spread → 2016 narrow), explaining why CF severity
loses rank-discrimination in 2016.  Under the recalibrated model, the spread
is restored.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from _cf_lib import FEATURES, ROOT, CACHE

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

# Stats for the diagnostic table
print('CF |ΔWARP| spread (std / IQR) per cohort:')
print(f'{"cohort":>8} {"base std":>10} {"recal std":>10} {"base IQR":>10} {"recal IQR":>10}')
for y in [2014, 2015, 2016]:
    bvals = cf_base[y]['cf_warp_abs'].values
    rvals = cf_recal[y]['cf_warp_abs'].values
    iqr_b = np.subtract(*np.percentile(bvals, [75, 25]))
    iqr_r = np.subtract(*np.percentile(rvals, [75, 25]))
    print(f'{y:>8} {bvals.std():10.4f} {rvals.std():10.4f} {iqr_b:10.4f} {iqr_r:10.4f}')

# ─── Plot: 2 rows (base / recal) × 3 cols (cohorts) of KDEs ───────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True)

xlim = (0, 0.20)
year_colors = {2014: '#4c72b0', 2015: '#dd8452', 2016: '#c44e52'}

for j, year in enumerate([2014, 2015, 2016]):
    for i, (label, dct) in enumerate([('Base model\n(train 2013–14)', cf_base),
                                       ('Recalibrated model\n(train 2014–15)', cf_recal)]):
        ax = axes[i, j]
        vals = dct[year]['cf_warp_abs'].values
        sns.histplot(vals, ax=ax, bins=40, kde=True,
                     color=year_colors[year], edgecolor='white', alpha=0.7)
        med = np.median(vals)
        iqr_lo, iqr_hi = np.percentile(vals, [25, 75])
        ax.axvline(med, color='black', linestyle='--', linewidth=1.2)
        ax.axvspan(iqr_lo, iqr_hi, alpha=0.12, color='black')
        ax.text(0.97, 0.92,
                f'n = {len(vals)}\nmedian = {med:.3f}\nIQR = {iqr_hi - iqr_lo:.3f}\nstd = {vals.std():.3f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color='#444441',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', linewidth=0.6))
        if i == 0:
            ax.set_title(f'Cohort {year}', fontsize=11.5)
        if j == 0:
            ax.set_ylabel(f'{label}\n\nCount', fontsize=10.5)
        else:
            ax.set_ylabel('')
        if i == 1:
            ax.set_xlabel('CF |ΔWARP| (in)')
        else:
            ax.set_xlabel('')
        ax.set_xlim(*xlim)
        ax.grid(axis='y', alpha=0.4, linewidth=0.5)
        sns.despine(ax=ax)

fig.suptitle('CF |ΔWARP| severity-score distribution by cohort and model\n'
             '(narrow distribution = less rank-discrimination = lower predictive AUC)',
             fontsize=12, fontweight='bold', y=1.01)

plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'cf_saturation.{ext}'),
                dpi=300, bbox_inches='tight')
plt.close()
print('Wrote: figures/paper_ready/cf_saturation.{png,pdf}')
