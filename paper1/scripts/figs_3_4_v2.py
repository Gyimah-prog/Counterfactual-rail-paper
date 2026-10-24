"""Revised figs #3 (worked CF) and #4 (stale CF) — delta-based bars + WARP zoom."""
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

def save(fig, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{stem}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {stem}.{{png,pdf}}')

df       = pickle.load(open(os.path.join(CACHE, 'data.pkl'), 'rb'))
model    = pickle.load(open(os.path.join(CACHE, 'model_base.pkl'), 'rb'))
cf_2014  = pickle.load(open(os.path.join(CACHE, 'cf_2014_base.pkl'), 'rb'))

warp_idx = FEATURES.index('WARP_62')

def delta_bar_panel(ax, deltas, title, colors=('#2a7d2a', '#c44e52')):
    """Horizontal bar of feature deltas, green=helpful, red=harmful."""
    bar_colors = [colors[1] if abs(d) < 0.001 else (colors[0] if d != 0 else '#888') for d in deltas]
    # Use signed colour: blue for negative shift, orange for positive
    bar_colors = ['#1f77b4' if d < 0 else ('#ff7f0e' if d > 0 else '#bbb') for d in deltas]
    y = np.arange(len(FEATURES))
    ax.barh(y, deltas, color=bar_colors, edgecolor='white', height=0.55)
    ax.set_yticks(y); ax.set_yticklabels(FEATURES, fontsize=10)
    ax.axvline(0, color='#444', linewidth=0.8)
    ax.set_xlabel('Recommended change (in)')
    ax.set_title(title)
    for yi, d in enumerate(deltas):
        if abs(d) < 1e-5: continue
        offset = max(abs(d) for d in deltas) * 0.04
        xpos = d + (offset if d > 0 else -offset)
        ax.text(xpos, yi, f'{d:+.4f}', va='center',
                ha='left' if d > 0 else 'right',
                fontsize=9, color='#444441')
    rng = max(abs(min(deltas)), abs(max(deltas)), 1e-3)
    ax.set_xlim(-rng * 1.5, rng * 1.5)
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True)
    ax.tick_params(axis='y', length=0)

def warp_zoom_panel(ax, warp_orig, warp_after, label_orig, label_after,
                     ok_after, title):
    """Zoom of WARP_62 number line showing original and CF/perturbed."""
    ax.axvspan(-0.30, 0.30, alpha=0.13, color='#2a7d2a')
    ax.axvline(-0.30, color='#666', linestyle='--', linewidth=1)
    ax.axvline( 0.30, color='#666', linestyle='--', linewidth=1)
    ax.scatter([warp_orig], [1], s=220, color='#dd8452', zorder=5,
               edgecolor='black', label=f'{label_orig}\n({warp_orig:+.3f} in)')
    ok_color = '#4c72b0' if ok_after else '#999'
    ax.scatter([warp_after], [1], s=220, color=ok_color, zorder=5,
               edgecolor='black',
               label=f'{label_after}\n({warp_after:+.3f} in)')
    ax.annotate('', xy=(warp_after, 1), xytext=(warp_orig, 1),
                arrowprops=dict(arrowstyle='->', color='#444' if ok_after else '#c44e52', lw=2))
    mid = (warp_orig + warp_after) / 2
    label = (f'|Δ| = {abs(warp_after-warp_orig):.3f} in' if ok_after
             else f'|Δ| = {abs(warp_after-warp_orig):.3f} in, advice fell short')
    ax.text(mid, 1.18, label, ha='center', fontsize=10,
            color='#444' if ok_after else '#c44e52')
    ax.text(0, 0.65, 'Compliant region (±0.30 in)', ha='center',
            fontsize=8.5, color='#2a7d2a', style='italic')
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(0.5, 1.4)
    ax.set_xlabel('WARP_62 (in)')
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.92, edgecolor='#cccccc')
    sns.despine(ax=ax, left=True)
    ax.tick_params(axis='y', length=0)

# ============================================================================
# Fig #3 — Worked CF example
# ============================================================================
print('[#3] worked CF example (v2)')
# Pick a median-severity WARP-only CF for clarity
sub = cf_2014[(cf_2014['cf_warp_abs'] > 0.02) & (cf_2014['cf_warp_abs'] < 0.08)].copy()
sub['total_abs'] = (sub[[f'd_{f}' for f in FEATURES]].abs().sum(axis=1))
ex = sub.sort_values('total_abs').iloc[len(sub)//2]
x0 = np.array([ex[f'x0_{f}'] for f in FEATURES])
cf = np.array([ex[f'cf_{f}'] for f in FEATURES])
deltas = cf - x0
print(f'  picked: MILE {int(ex["MILE"])} FEET {int(ex["FEET"])}  WARP0={x0[warp_idx]:+.3f}  CFWARP={cf[warp_idx]:+.3f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
delta_bar_panel(axes[0], deltas,
                'Per-feature minimum change recommended by CF')
warp_zoom_panel(axes[1], x0[warp_idx], cf[warp_idx],
                'Original (defective)', 'Counterfactual',
                ok_after=True,
                title='WARP_62: original vs CF on the number line')

mile = int(ex['MILE']); feet = int(ex['FEET'])
fig.suptitle(f'Worked CF example  —  segment MILE {mile}, FEET {feet}, '
             f'{ex["date"].strftime("%b %Y")}',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, 'cf_worked_example')

# ============================================================================
# Fig #4 — Stale CF example
# ============================================================================
print('\n[#4] stale CF example (v2)')
mean_delta = cf_2014[[f'd_{f}' for f in FEATURES]].mean().values
print(f'  2014 mean delta: {dict(zip(FEATURES, [f"{d:+.4f}" for d in mean_delta]))}')

def_2016 = df[(df['year'] == 2016) & (df['defect'] == 1)].copy()

# Pick a 2016 sample where 2014 advice fails — prefer "wrong direction"
# i.e. 2014 mean delta moves WARP toward MORE defective
candidates = []
for _, row in def_2016.iterrows():
    x0_2016 = row[FEATURES].values.astype(float)
    x_shifted = x0_2016 + mean_delta
    p_shifted = float(model.predict_proba(x_shifted.reshape(1,-1))[0,1])
    if p_shifted > 0.7:
        # Strong failure case — keep it
        candidates.append((row, x_shifted, p_shifted, abs(x_shifted[warp_idx]) - abs(x0_2016[warp_idx])))
    if len(candidates) >= 30:
        break

# Among failures, pick one where direction was wrong (made it worse)
wrong_dir = [c for c in candidates if c[3] > 0.005]
ex16, x_shifted, p_shifted, _ = wrong_dir[0] if wrong_dir else candidates[0]
x0_2016 = ex16[FEATURES].values.astype(float)
deltas_applied = x_shifted - x0_2016  # same as mean_delta (sanity)
print(f'  picked: WARP0={x0_2016[warp_idx]:+.3f}  shifted={x_shifted[warp_idx]:+.3f}  '
      f'p_after={p_shifted:.2f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
delta_bar_panel(axes[0], deltas_applied,
                'Per-feature change from 2014 mean Δ (applied uniformly)')
warp_zoom_panel(axes[1], x0_2016[warp_idx], x_shifted[warp_idx],
                '2016 original (defective)',
                f'After 2014 advice (p={p_shifted:.2f})',
                ok_after=False,
                title='WARP_62: 2014 advice fails on a 2016 sample')

mile = int(ex16['MILE']); feet = int(ex16['FEET'])
fig.suptitle(f'Failed CF transfer  —  segment MILE {mile}, FEET {feet}, '
             f'{ex16["date"].strftime("%b %Y")}',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, 'cf_stale_example')
