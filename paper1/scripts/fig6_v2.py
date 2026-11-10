"""Revised fig #6 — per-window flip rate decay; drop empty rows, sort by 2014 baseline."""
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

df       = pickle.load(open(os.path.join(CACHE, 'data.pkl'), 'rb'))
model    = pickle.load(open(os.path.join(CACHE, 'model_base.pkl'), 'rb'))
cf_2014  = pickle.load(open(os.path.join(CACHE, 'cf_2014_base.pkl'), 'rb'))

mean_delta_base = cf_2014[[f'd_{f}' for f in FEATURES]].mean().values

def window_flip_rate(year, min_n=5):
    target = df[(df['year'] == year) & (df['defect'] == 1)].copy()
    target['window_200ft'] = (target['FEET'].astype(int) // 200)
    X0 = target[FEATURES].values.astype(float)
    Xs = X0 + mean_delta_base
    target['flipped'] = (model.predict(Xs) == 0).astype(int)
    stats = (target.groupby('window_200ft')['flipped']
             .agg(['mean', 'count']).reset_index())
    stats = stats[stats['count'] >= min_n]
    stats['mean'] *= 100
    return stats

periods = [('Late 2014\n(in-distribution)', 2014),
           ('2015', 2015),
           ('2016', 2016)]

# Build wide matrix
all_stats = {label: window_flip_rate(y) for label, y in periods}

# Keep windows that appear in ALL three periods with enough samples
common = set(all_stats[periods[0][0]]['window_200ft'])
for label, _ in periods[1:]:
    common &= set(all_stats[label]['window_200ft'])
common = sorted(common)
print(f'  {len(common)} windows present in all 3 periods')

mat   = np.zeros((len(common), len(periods)))
counts = np.zeros_like(mat, dtype=int)
for j, (label, _) in enumerate(periods):
    s = all_stats[label].set_index('window_200ft')
    for i, w in enumerate(common):
        mat[i, j]    = s.loc[w, 'mean']
        counts[i, j] = s.loc[w, 'count']

# Sort windows by their late-2014 flip rate (highest at top) for readability
order = np.argsort(-mat[:, 0])
mat = mat[order]
counts = counts[order]
windows_sorted = [common[i] for i in order]

fig, ax = plt.subplots(figsize=(8.5, 0.35 * len(common) + 1.5))
im = ax.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(periods)))
ax.set_xticklabels([p[0] for p in periods], fontsize=10)
ax.set_yticks(range(len(windows_sorted)))
ax.set_yticklabels([f'win {w}' for w in windows_sorted], fontsize=9)
ax.set_ylabel('200-ft window ID (sorted by 2014 flip rate)')
ax.set_title('Per-window CF flip-rate decay across periods\n'
             '(2014 mean Δ applied; only windows with ≥5 defects in all 3 periods shown)',
             fontsize=11.5, pad=10)

cbar = plt.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label('Flip rate (%)')

for i in range(len(windows_sorted)):
    for j in range(len(periods)):
        v = mat[i, j]
        col = 'white' if 25 < v < 75 else 'black'
        ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                fontsize=8.5, color=col)
        # Annotate sample sizes in tiny gray on the right margin
        if j == len(periods) - 1:
            ax.text(j + 0.55, i, f'n≈{int(counts[i,j])}',
                    va='center', fontsize=7, color='#888')

plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'spatial_flip_rate.{ext}'), dpi=300, bbox_inches='tight')
plt.close()
print('Wrote: figures/paper_ready/spatial_flip_rate.{png,pdf}')
