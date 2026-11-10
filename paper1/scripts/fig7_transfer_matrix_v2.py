"""Fig #7 v2 — Transfer matrix using the notebook's EXACT protocols.

Key differences from v1:
  • P1 uses sign-correction on WARP direction (per notebook cell 45)
  • Both protocols clip the perturbed point to FRA_LIMITS (notebook does this)
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from _cf_lib import FEATURES, ROOT, CACHE, FRA_LIMITS, compute_mad

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

FRA_LO = np.array([FRA_LIMITS[f][0] for f in FEATURES])
FRA_HI = np.array([FRA_LIMITS[f][1] for f in FEATURES])
warp_idx = FEATURES.index('WARP_62')

def p1_signcorrected_flip(cf_ref, df_test):
    """Notebook's actual P1: mean magnitude with per-sample WARP direction."""
    md = cf_ref[[f'd_{f}' for f in FEATURES]].mean().values
    X0 = df_test[FEATURES].values.astype(float)
    # Per-sample sign-corrected delta
    deltas = np.tile(md, (len(X0), 1))
    pos_warp = X0[:, warp_idx] > 0
    deltas[ pos_warp, warp_idx] = -abs(md[warp_idx])
    deltas[~pos_warp, warp_idx] = +abs(md[warp_idx])
    Xs = np.clip(X0 + deltas, FRA_LO, FRA_HI)
    pred0 = model.predict(X0)
    pred1 = model.predict(Xs)
    flipped = (pred0 == 1) & (pred1 == 0)
    n_def = (pred0 == 1).sum()
    return flipped.sum() / max(1, n_def) * 100

def p2_signcorrected_flip(cf_ref, df_test):
    """Notebook's P2: NN-matched delta with WARP sign correction + FRA clip."""
    X_ref_orig  = cf_ref[[f'x0_{f}' for f in FEATURES]].values
    X_ref_delta = cf_ref[[f'd_{f}' for f in FEATURES]].values
    X0 = df_test[FEATURES].values.astype(float)
    # MAD-scaled Euclidean (notebook uses euclidean, not L1)
    Xn  = X0 / mad
    Rn  = X_ref_orig / mad
    diff = Xn[:, None, :] - Rn[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    nn = dist.argmin(axis=1)
    deltas = X_ref_delta[nn].copy()
    # sign-correct WARP per sample
    pos_warp = X0[:, warp_idx] > 0
    deltas[ pos_warp, warp_idx] = -np.abs(deltas[ pos_warp, warp_idx])
    deltas[~pos_warp, warp_idx] = +np.abs(deltas[~pos_warp, warp_idx])
    Xs = np.clip(X0 + deltas, FRA_LO, FRA_HI)
    pred0 = model.predict(X0)
    pred1 = model.predict(Xs)
    flipped = (pred0 == 1) & (pred1 == 0)
    n_def = (pred0 == 1).sum()
    return flipped.sum() / max(1, n_def) * 100

YEARS = [2014, 2015, 2016]
P1 = np.zeros((3, 3)); P2 = np.zeros((3, 3))

for i, ref_y in enumerate(YEARS):
    cf_ref = cf_by_year[ref_y]
    for j, test_y in enumerate(YEARS):
        df_test = df[(df['year'] == test_y) & (df['defect'] == 1)]
        P1[i, j] = p1_signcorrected_flip(cf_ref, df_test)
        P2[i, j] = p2_signcorrected_flip(cf_ref, df_test)
        print(f'  ref {ref_y} → test {test_y} :  '
              f'P1={P1[i,j]:5.1f}%   P2={P2[i,j]:5.1f}%   (n={len(df_test)})')

# ─── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
for ax, mat, title in [(axes[0], P1, 'Protocol 1: mean Δ + WARP-sign correction'),
                        (axes[1], P2, 'Protocol 2: NN-matched Δ + WARP-sign correction')]:
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
    for k in range(3):
        ax.add_patch(plt.Rectangle((k-0.5, k-0.5), 1, 1, fill=False,
                                    edgecolor='black', linewidth=2))

cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02, shrink=0.85)
cbar.set_label('Flip rate (%)')

fig.suptitle("Counterfactual transferability — notebook's exact protocols\n"
             '(base model trained on 2013–14; diagonal = in-distribution)',
             fontsize=12.5, fontweight='bold', y=1.04)

for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'transfer_matrix.{ext}'),
                dpi=300, bbox_inches='tight')
plt.close()
print('\nWrote: figures/paper_ready/transfer_matrix.{png,pdf}')

# Refresh CSV
out_csv = os.path.join(OUT, 'transfer_matrix.csv')
with open(out_csv, 'w') as f:
    f.write('protocol,ref_year,test_year,flip_rate_pct\n')
    for label, M in [('P1', P1), ('P2', P2)]:
        for i, ry in enumerate(YEARS):
            for j, ty in enumerate(YEARS):
                f.write(f'{label},{ry},{ty},{M[i,j]:.2f}\n')
print(f'Wrote: {out_csv}')
