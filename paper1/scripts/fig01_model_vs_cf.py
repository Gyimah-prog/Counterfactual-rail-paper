"""Fig #1 — Classifier AUC vs CF flip rate, both by year.

Shows the central tension of the paper: model performance stays high while
counterfactual recommendations decay.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from _cf_lib import load_data, train_model, FEATURES, ROOT

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

print('Loading data + training model...')
df = load_data()
model = train_model(df, train_years=(2013, 2014), tag='base')

# Per-year AUC on the FULL year (defects + non-defects)
years = [2014, 2015, 2016]
aucs  = []
for y in years:
    sub = df[df['year'] == y]
    p   = model.predict_proba(sub[FEATURES].values)[:, 1]
    aucs.append(roc_auc_score(sub['defect'].values, p))
print('AUCs:', dict(zip(years, [f'{a:.4f}' for a in aucs])))

# Flip rates (from notebook output; P2 nearest-neighbour protocol)
flip_rates = {'2014': 98.2, '2015': 28.5, '2016': 6.9}
flips = [flip_rates[str(y)] for y in years]

# ─── Plot ──────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8.5, 5))
x = np.arange(len(years))

# Classifier AUC (left axis)
c_auc = '#1f77b4'
ax1.plot(x, aucs, marker='o', markersize=10, color=c_auc, linewidth=2.5,
         label='Classifier AUC (held-out)', zorder=3)
for xi, v in zip(x, aucs):
    ax1.annotate(f'{v:.4f}', (xi, v), textcoords='offset points',
                 xytext=(0, 12), ha='center', fontsize=9.5, color=c_auc)
ax1.set_ylim(0.0, 1.05)
ax1.set_ylabel('Classifier AUC', color=c_auc, fontsize=11)
ax1.tick_params(axis='y', colors=c_auc)
ax1.axhline(0.5, linestyle=':', color='#999', linewidth=0.8, zorder=1)

# CF flip rate (right axis)
ax2 = ax1.twinx()
c_flip = '#d62728'
ax2.plot(x, flips, marker='s', markersize=10, color=c_flip, linewidth=2.5,
         linestyle='--', label='CF flip rate (P2-NN)', zorder=3)
for xi, v in zip(x, flips):
    ax2.annotate(f'{v:.1f}%', (xi, v), textcoords='offset points',
                 xytext=(0, -18), ha='center', fontsize=9.5, color=c_flip)
ax2.set_ylim(0, 105)
ax2.set_ylabel('CF flip rate (%)', color=c_flip, fontsize=11)
ax2.tick_params(axis='y', colors=c_flip)
ax2.spines['top'].set_visible(False)
ax2.grid(False)

ax1.set_xticks(x)
ax1.set_xticklabels([str(y) for y in years], fontsize=11)
ax1.set_xlabel('Year', fontsize=11)
ax1.set_title('Classifier remains accurate while counterfactual recommendations decay\n'
              '(2013–14 training regime; frozen XGBoost evaluated on each subsequent year)',
              fontsize=11.5, pad=10)

# Manual combined legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], color=c_auc, marker='o', markersize=8, linewidth=2.5,
           label='Classifier AUC (held-out)'),
    Line2D([0],[0], color=c_flip, marker='s', markersize=8, linewidth=2.5,
           linestyle='--', label='CF flip rate (P2 nearest-neighbour)'),
]
ax1.legend(handles=handles, loc='center left', framealpha=0.95,
           edgecolor='#cccccc', fontsize=10)

plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'model_vs_cf.{ext}'), dpi=300, bbox_inches='tight')
plt.close()
print('Wrote: figures/paper_ready/model_vs_cf.{png,pdf}')
