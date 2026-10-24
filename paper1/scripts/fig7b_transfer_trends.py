"""Fig #7b — Transfer trends.

Same numbers as the 3×3 matrix (notebook's exact sign-corrected protocols),
but rendered as line plots so the trend across test years is visible at a
glance.  One line per reference year, one panel per protocol.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def p1(cf_ref, df_test):
    md = cf_ref[[f'd_{f}' for f in FEATURES]].mean().values
    X0 = df_test[FEATURES].values.astype(float)
    deltas = np.tile(md, (len(X0), 1))
    pos = X0[:, warp_idx] > 0
    deltas[ pos, warp_idx] = -abs(md[warp_idx])
    deltas[~pos, warp_idx] = +abs(md[warp_idx])
    Xs = np.clip(X0 + deltas, FRA_LO, FRA_HI)
    return ((model.predict(X0) == 1) & (model.predict(Xs) == 0)).sum() / max(1, (model.predict(X0)==1).sum()) * 100

def p2(cf_ref, df_test):
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
    return ((model.predict(X0) == 1) & (model.predict(Xs) == 0)).sum() / max(1, (model.predict(X0)==1).sum()) * 100

YEARS = [2014, 2015, 2016]
P1 = {ry: [p1(cf_by_year[ry], df[(df['year']==ty)&(df['defect']==1)]) for ty in YEARS] for ry in YEARS}
P2 = {ry: [p2(cf_by_year[ry], df[(df['year']==ty)&(df['defect']==1)]) for ty in YEARS] for ry in YEARS}

ref_colors  = {2014: '#1f77b4', 2015: '#ff7f0e', 2016: '#2ca02c'}
ref_markers = {2014: 'o',       2015: 's',       2016: 'D'}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
x = np.arange(len(YEARS))

for ax, results, title in [
    (axes[0], P1, 'Protocol 1: mean Δ + WARP sign-correction'),
    (axes[1], P2, 'Protocol 2: NN-matched Δ + WARP sign-correction'),
]:
    for ry in YEARS:
        vals = results[ry]
        ax.plot(x, vals, marker=ref_markers[ry], markersize=11, linewidth=2.5,
                color=ref_colors[ry], label=f'Reference year = {ry}')
        # annotate
        for xi, v in zip(x, vals):
            offset = (0, 10) if ry == 2014 else ((0, -16) if ry == 2016 else (12, 0))
            ax.annotate(f'{v:.1f}%', (xi, v), textcoords='offset points',
                        xytext=offset, fontsize=9, color=ref_colors[ry],
                        ha='center')
        # highlight in-distribution point
        in_dist = YEARS.index(ry)
        ax.scatter([in_dist], [vals[in_dist]], s=260, facecolors='none',
                   edgecolors=ref_colors[ry], linewidth=2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in YEARS], fontsize=11)
    ax.set_xlabel('Test year (where CFs are applied)')
    ax.set_title(title, fontsize=11.5)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right' if title.startswith('Protocol 2') else 'upper right',
              framealpha=0.95, edgecolor='#cccccc', title='Reference CFs')
    ax.grid(axis='y', alpha=0.4, linewidth=0.5)

axes[0].set_ylabel('Flip rate (%)')
fig.suptitle('Counterfactual transferability — by reference year\n'
             '(notebook\'s exact P1/P2; circled = in-distribution)',
             fontsize=12.5, fontweight='bold', y=1.02)

plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'transfer_trends.{ext}'), dpi=300, bbox_inches='tight')
plt.close()
print('Wrote: figures/paper_ready/transfer_trends.{png,pdf}')

print('\n=== Summary ===')
print(f'{"":12s} {"test 2014":>10s} {"test 2015":>10s} {"test 2016":>10s}')
for proto, R in [('P1', P1), ('P2', P2)]:
    for ry in YEARS:
        print(f'{proto} ref {ry}: {R[ry][0]:9.1f}% {R[ry][1]:9.1f}% {R[ry][2]:9.1f}%')
