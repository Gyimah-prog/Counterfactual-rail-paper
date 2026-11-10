"""Fig #7 — Full CF transferability matrix.

For each (reference year × test year) combination, compute flip rate under
both protocols P1 (mean delta) and P2 (nearest-neighbour matched delta).

Diagonal = in-distribution; off-diagonal = transfer.  This shows whether
using a more recent reference rescues the recommendation, or whether any
cross-year transfer is symmetrically harmful.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from _cf_lib import FEATURES, ROOT, CACHE, compute_mad

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

df    = pickle.load(open(os.path.join(CACHE, 'data.pkl'), 'rb'))
model = pickle.load(open(os.path.join(CACHE, 'model_base.pkl'), 'rb'))
mad   = compute_mad(df, train_years=(2013, 2014))

cf_by_year = {
    2014: pickle.load(open(os.path.join(CACHE, 'cf_2014_base.pkl'), 'rb')),
    2015: pickle.load(open(os.path.join(CACHE, 'cf_2015_base.pkl'), 'rb')),
    2016: pickle.load(open(os.path.join(CACHE, 'cf_2016_base.pkl'), 'rb')),
}

def p1_flip(cf_ref, df_test):
    md = cf_ref[[f'd_{f}' for f in FEATURES]].mean().values
    X0 = df_test[FEATURES].values.astype(float)
    Xs = X0 + md
    return (model.predict(Xs) == 0).mean() * 100

def p2_flip(cf_ref, df_test):
    X_ref_orig  = cf_ref[[f'x0_{f}' for f in FEATURES]].values
    X_ref_delta = cf_ref[[f'd_{f}' for f in FEATURES]].values
    X0_test = df_test[FEATURES].values.astype(float)
    # MAD-scaled L1 distance per (test, ref) pair
    diff = np.abs((X0_test[:, None, :] - X_ref_orig[None, :, :]) / mad)
    dist = diff.sum(axis=2)
    nn_idx = dist.argmin(axis=1)
    deltas = X_ref_delta[nn_idx]
    Xs = X0_test + deltas
    return (model.predict(Xs) == 0).mean() * 100

YEARS = [2014, 2015, 2016]
P1 = np.zeros((3, 3)); P2 = np.zeros((3, 3))

for i, ref_y in enumerate(YEARS):
    cf_ref = cf_by_year[ref_y]
    for j, test_y in enumerate(YEARS):
        df_test = df[(df['year'] == test_y) & (df['defect'] == 1)]
        P1[i, j] = p1_flip(cf_ref, df_test)
        P2[i, j] = p2_flip(cf_ref, df_test)
        print(f'  ref {ref_y} → test {test_y} :  '
              f'P1={P1[i,j]:5.1f}%   P2={P2[i,j]:5.1f}%   '
              f'(n_test={len(df_test)})')

# ─── Plot: two heatmaps side by side ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
for ax, mat, title in [(axes[0], P1, 'Protocol 1: mean delta'),
                        (axes[1], P2, 'Protocol 2: nearest-neighbour delta')]:
    im = ax.imshow(mat, cmap='RdYlGn', aspect='equal', vmin=0, vmax=100)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([str(y) for y in YEARS], fontsize=11)
    ax.set_yticklabels([str(y) for y in YEARS], fontsize=11)
    ax.set_xlabel('Test year (where CFs are applied)', fontsize=10.5)
    ax.set_ylabel('Reference year (where CFs are computed)', fontsize=10.5)
    ax.set_title(title, fontsize=11.5)
    for i in range(3):
        for j in range(3):
            v = mat[i, j]
            txt_color = 'white' if 25 < v < 75 else 'black'
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                    fontsize=12, fontweight='bold', color=txt_color)
    # Highlight diagonal (in-distribution)
    for k in range(3):
        ax.add_patch(plt.Rectangle((k-0.5, k-0.5), 1, 1, fill=False,
                                    edgecolor='black', linewidth=2))

cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02, shrink=0.85)
cbar.set_label('Flip rate (%)')

fig.suptitle('Counterfactual transferability across reference and test years\n'
             '(base model trained on 2013–14; diagonal = in-distribution)',
             fontsize=12.5, fontweight='bold', y=1.04)

for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'transfer_matrix.{ext}'),
                dpi=300, bbox_inches='tight')
plt.close()
print('\nWrote: figures/paper_ready/transfer_matrix.{png,pdf}')

# Also dump the raw matrices as a small CSV for the paper's table
out_csv = os.path.join(OUT, 'transfer_matrix.csv')
with open(out_csv, 'w') as f:
    f.write('protocol,ref_year,test_year,flip_rate_pct\n')
    for label, M in [('P1', P1), ('P2', P2)]:
        for i, ry in enumerate(YEARS):
            for j, ty in enumerate(YEARS):
                f.write(f'{label},{ry},{ty},{M[i,j]:.2f}\n')
print(f'Wrote: {out_csv}')
