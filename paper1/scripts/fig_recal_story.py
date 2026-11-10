"""Three figures crystallizing the recalibration / reference-cohort findings
under the notebook-hyperparameter pipeline.

A) recal_decomposition.png      — where the recalibration gain comes from
B) reference_cohort_effect.png  — P2 flip rate by reference choice across test years
C) ranking_vs_prescription.png  — the two-operational-properties dissociation
"""
import os, sys, pickle, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_auc_score
from _cf_lib import FEATURES, CACHE, ROOT, PROJECT_LIMITS

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

# load the headline-numbers json
hn = json.load(open(os.path.join(CACHE, 'headline_numbers_nb.json')))

# ============================================================
# Figure A — recalibration decomposition
# ============================================================
print('Building A — recal_decomposition.png')

stale_p2  = hn['flip_2016']['stale_p2']
abl_p2    = hn['flip_2016']['ablation_p2']
recal_p2  = hn['flip_2016']['annual_recal_p2']
# Reference-cohort-only: base + 2015 ref on 2016 (from transfer matrix)
ref_only_p2 = hn['transfer_P2']['base_2015_2016']
# In-distribution within base model: base + 2016 ref on 2016
oracle_p2 = hn['transfer_P2']['base_2016_2016']

labels = [
    'Stale\n(base + 2014 ref)',
    'Model update only\n(recal + 2014 ref)',
    'Reference update only\n(base + 2015 ref)',
    'Full annual recal\n(recal + 2015 ref)',
    'In-distribution ceiling\n(base + 2016 ref)',
]
values = [stale_p2, abl_p2, ref_only_p2, recal_p2, oracle_p2]
colors = ['#1f4068', '#a4b3d6', '#ff9d4d', '#ff7f0e', '#2a7d2a']

fig, ax = plt.subplots(figsize=(11.5, 5.6))
x = np.arange(len(labels))
bars = ax.bar(x, values, color=colors, edgecolor='white', width=0.6)
for xi, v in zip(x, values):
    ax.text(xi, v + 1.0, f'{v:.1f}%', ha='center', fontsize=11,
            fontweight='bold', color='#444441')

# Add Δ annotations between bars
deltas = []
for i in range(1, len(values)):
    delta = values[i] - values[0]
    deltas.append(delta)

ax.annotate(f'+{values[1] - values[0]:.1f} pp', xy=(0.5, 12),
            ha='center', fontsize=9.5, color='#666', style='italic')
ax.annotate(f'+{values[2] - values[0]:.1f} pp', xy=(1.5, 22),
            ha='center', fontsize=9.5, color='#666', style='italic')
ax.annotate(f'+{values[3] - values[0]:.1f} pp', xy=(2.5, 24),
            ha='center', fontsize=9.5, color='#666', style='italic')
ax.annotate(f'(structural\nceiling)', xy=(3.5, 30),
            ha='center', fontsize=9, color='#666', style='italic')

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('P2 flip rate on 2016 defective cohort (%)')
ax.set_ylim(0, max(values) * 1.25)
ax.set_title('Where does the recalibration gain come from?\n'
             'Reference-cohort update dominates; model retraining adds little; in-distribution ceiling caps the recovery',
             fontsize=11.5, pad=10)
ax.grid(axis='y', alpha=0.4, linewidth=0.5)

# Legend by intervention type
legend_items = [
    mpatches.Patch(color='#1f4068', label='No intervention (stale)'),
    mpatches.Patch(color='#a4b3d6', label='Model retrained only'),
    mpatches.Patch(color='#ff9d4d', label='Reference cohort updated only'),
    mpatches.Patch(color='#ff7f0e', label='Both updated (annual recal)'),
    mpatches.Patch(color='#2a7d2a', label='In-distribution oracle (best case under frozen model)'),
]
ax.legend(handles=legend_items, loc='upper left', framealpha=0.95,
          edgecolor='#cccccc', fontsize=9)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'recal_decomposition.{ext}'), dpi=300, bbox_inches='tight')
plt.close()
print('  wrote recal_decomposition.{png,pdf}')


# ============================================================
# Figure B — reference cohort effect across test years
# ============================================================
print('Building B — reference_cohort_effect.png')

years = [2014, 2015, 2016]
mat = np.zeros((3, 3))
for i, ref_y in enumerate(years):
    for j, test_y in enumerate(years):
        mat[i, j] = hn['transfer_P2'][f'base_{ref_y}_{test_y}']

ref_colors = {2014: '#1f77b4', 2015: '#ff7f0e', 2016: '#2ca02c'}
ref_markers = {2014: 'o', 2015: 's', 2016: 'D'}

fig, ax = plt.subplots(figsize=(9.5, 5.2))
x = np.arange(len(years))
for i, ref_y in enumerate(years):
    vals = mat[i, :]
    ax.plot(x, vals, marker=ref_markers[ref_y], markersize=11, linewidth=2.5,
            color=ref_colors[ref_y], label=f'CFs from {ref_y} reference cohort')
    for xi, v, ty in zip(x, vals, years):
        if ref_y == ty:
            # In-distribution: highlight with bigger circle
            ax.scatter([xi], [v], s=300, facecolors='none',
                       edgecolors=ref_colors[ref_y], linewidth=2.2, zorder=4)
        ax.annotate(f'{v:.1f}%', (xi, v), textcoords='offset points',
                    xytext=(8, 6), fontsize=9, color=ref_colors[ref_y])

ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years], fontsize=11)
ax.set_xlabel('Test year (where CFs are applied)')
ax.set_ylabel('P2 flip rate (%)')
ax.set_ylim(0, 110)
ax.set_title('Choosing the right CF reference cohort matters more than the model\n'
             '(base_nb model used throughout; circled = in-distribution)',
             fontsize=11.5, pad=10)
ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc',
          title='Reference cohort')
ax.grid(axis='y', alpha=0.4, linewidth=0.5)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'reference_cohort_effect.{ext}'), dpi=300, bbox_inches='tight')
plt.close()
print('  wrote reference_cohort_effect.{png,pdf}')


# ============================================================
# Figure C — ranking robustness vs prescription brittleness
# ============================================================
print('Building C — ranking_vs_prescription.png')

# Per-cohort:
#   - CF |ΔWARP| AUC for persistence (1-month)
#   - P2 flip rate using in-distribution reference (diagonal of transfer matrix)
auc_by_year = {y: hn['auc_per_cohort'][f'base_{y}_1m_cf_warp_abs'] for y in years}
flip_by_year = {y: hn['transfer_P2'][f'base_{y}_{y}'] for y in years}
# transfer (forward from 2014, the canonical paper finding)
flip_forward = {y: hn['transfer_P2'][f'base_2014_{y}'] for y in years}

fig, ax1 = plt.subplots(figsize=(10, 5.2))
x = np.arange(len(years))

# Ranking (CF AUC) on left axis
c_auc = '#1f77b4'
ax1.plot(x, [auc_by_year[y] for y in years],
         marker='o', linewidth=2.8, markersize=11, color=c_auc,
         label='CF |ΔWARP| AUC for persistence (ranking quality)')
for xi, y in zip(x, years):
    v = auc_by_year[y]
    ax1.annotate(f'{v:.3f}', (xi, v), textcoords='offset points',
                 xytext=(0, 12), ha='center', fontsize=10, color=c_auc)
ax1.set_ylim(0.0, 1.05)
ax1.set_ylabel('AUC for 1-month persistence prediction', color=c_auc, fontsize=11)
ax1.tick_params(axis='y', colors=c_auc)
ax1.axhline(0.5, color='#999', linestyle=':', linewidth=0.7)

# Prescription (P2 flip rate) on right axis
ax2 = ax1.twinx()
c_flip = '#d62728'
ax2.plot(x, [flip_forward[y] for y in years],
         marker='s', linewidth=2.8, markersize=11, color=c_flip,
         linestyle='--', label='P2 flip rate with 2014 reference (prescription quality)')
for xi, y in zip(x, years):
    v = flip_forward[y]
    ax2.annotate(f'{v:.1f}%', (xi, v), textcoords='offset points',
                 xytext=(0, -18), ha='center', fontsize=10, color=c_flip)
ax2.set_ylim(0, 105)
ax2.set_ylabel('P2 flip rate using 2014 reference cohort (%)', color=c_flip, fontsize=11)
ax2.tick_params(axis='y', colors=c_flip)
ax2.spines['top'].set_visible(False)
ax2.grid(False)

ax1.set_xticks(x); ax1.set_xticklabels([str(y) for y in years], fontsize=11)
ax1.set_xlabel('Cohort year (test data)', fontsize=11)
ax1.set_title('Ranking robust, prescription brittle\n'
              'CF severity score stays predictive (AUC ≈ 0.85), but transferred CF prescriptions collapse',
              fontsize=11.5, pad=10)

# Combined legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], color=c_auc, marker='o', linewidth=2.5, label='CF severity AUC  (ranking)'),
    Line2D([0],[0], color=c_flip, marker='s', linewidth=2.5, linestyle='--',
            label='P2 flip rate, 2014 ref  (prescription)'),
]
ax1.legend(handles=handles, loc='center left', framealpha=0.95,
           edgecolor='#cccccc', fontsize=10)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'ranking_vs_prescription.{ext}'), dpi=300, bbox_inches='tight')
plt.close()
print('  wrote ranking_vs_prescription.{png,pdf}')

print('\nDone.')
