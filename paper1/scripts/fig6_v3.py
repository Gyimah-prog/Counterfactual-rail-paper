"""Revised fig #6 — spatial defect migration over time.

Shows that defects don't just *grow* with drift — they *migrate* to new
windows. This is direct evidence that the distribution being learned in
2014 (defects in windows 5, 21, 23, 24, 25) is not the distribution to be
explained in 2016 (heavy in 2, 10, 14, 16, 17, 22, 23).
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from _cf_lib import ROOT, CACHE

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
defs = df[df['defect']==1].copy()
defs['win'] = defs['FEET'].astype(int) // 200

pivot = defs.groupby(['win','year']).size().unstack(fill_value=0)
# Restrict to windows that ever have defects
pivot = pivot[(pivot > 0).any(axis=1)]
# Normalize: defects per inspection-month per window (rate, not count)
months_per_year = df.groupby('year')['date'].nunique()
rate = pivot.div(months_per_year, axis=1)

# Sort windows by 2014 defect count (descending) so reference regime is top
pivot = pivot.sort_values(2014, ascending=False)
rate  = rate.loc[pivot.index]

fig, ax = plt.subplots(figsize=(7.5, 0.35*len(pivot)+1.8))
im = ax.imshow(rate.values, cmap='YlOrRd', aspect='auto',
               vmin=0, vmax=rate.values.max())
ax.set_xticks(range(len(rate.columns)))
ax.set_xticklabels([str(c) for c in rate.columns], fontsize=10)
ax.set_yticks(range(len(rate.index)))
ax.set_yticklabels([f'win {w}' for w in rate.index], fontsize=9)
ax.set_xlabel('Year')
ax.set_ylabel('200-ft window ID  (sorted by 2014 defect count)')
ax.set_title('Defects migrate spatially as the distribution drifts\n'
             '(defects per inspection-month per window)',
             fontsize=11.5, pad=10)

cbar = plt.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label('Defects per month')

# Annotate counts so the migration is readable
for i, w in enumerate(rate.index):
    for j, y in enumerate(rate.columns):
        cnt = pivot.loc[w, y]
        if cnt == 0:
            ax.text(j, i, '·', ha='center', va='center', fontsize=10, color='#bbb')
        else:
            v = rate.iloc[i, j]
            col = 'white' if v > rate.values.max()*0.5 else 'black'
            ax.text(j, i, f'{int(cnt)}', ha='center', va='center',
                    fontsize=8, color=col)

plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'spatial_defect_migration.{ext}'),
                dpi=300, bbox_inches='tight')
plt.close()
# Drop the old name
old = os.path.join(OUT, 'spatial_flip_rate.png')
if os.path.exists(old):
    os.remove(old)
    os.remove(os.path.join(OUT, 'spatial_flip_rate.pdf'))
print('Wrote: spatial_defect_migration.{png,pdf}  (replaces spatial_flip_rate)')
