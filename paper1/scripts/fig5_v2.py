"""Revised fig #5 — recalibration with FAIR comparison.

Each pipeline uses its OWN reference CFs (computed under its own model):
  • Base pipeline:    base model + 2014 reference CFs → applied to 2016
  • Recal pipeline:   recal model + 2014 reference CFs → applied to 2016
  • Recal pipeline+:  recal model + 2015 reference CFs → applied to 2016 (most recent ref)

This isolates "did retraining the model help?" from "did using a more
recent reference year help?"
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
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

df    = pickle.load(open(os.path.join(CACHE, 'data.pkl'), 'rb'))
base  = pickle.load(open(os.path.join(CACHE, 'model_base.pkl'), 'rb'))
recal = pickle.load(open(os.path.join(CACHE, 'model_recal.pkl'), 'rb'))

cf_2014_base  = pickle.load(open(os.path.join(CACHE, 'cf_2014_base.pkl'), 'rb'))
cf_2014_recal = pickle.load(open(os.path.join(CACHE, 'cf_2014_recal.pkl'), 'rb'))

# Try optional 2015 reference CFs under recal
try:
    cf_2015_recal = pickle.load(open(os.path.join(CACHE, 'cf_2015_recal.pkl'), 'rb'))
    have_2015r = True
except FileNotFoundError:
    have_2015r = False

def mean_delta(cf_df):
    return cf_df[[f'd_{f}' for f in FEATURES]].mean().values

def flip_rate_2016(model, md):
    t = df[(df['year'] == 2016) & (df['defect'] == 1)].copy()
    X0 = t[FEATURES].values.astype(float)
    Xs = X0 + md
    return (model.predict(Xs) == 0).mean() * 100

md_2014_base  = mean_delta(cf_2014_base)
md_2014_recal = mean_delta(cf_2014_recal)

results = [
    ('Base model\n+ 2014 ref CFs',         flip_rate_2016(base,  md_2014_base),  '#dd8452'),
    ('Recalibrated model\n+ 2014 ref CFs', flip_rate_2016(recal, md_2014_recal), '#4c72b0'),
]
if have_2015r:
    md_2015_recal = mean_delta(cf_2015_recal)
    results.append(('Recalibrated model\n+ 2015 ref CFs',
                    flip_rate_2016(recal, md_2015_recal), '#2a7d2a'))

labels   = [r[0] for r in results]
values   = [r[1] for r in results]
colors   = [r[2] for r in results]

print('Flip rates on 2016 defective cohort:')
for lbl, v in zip(labels, values):
    print(f'  {v:5.1f}%   {lbl.replace(chr(10), " ")}')

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(labels))
bars = ax.bar(x, values, color=colors, edgecolor='white', width=0.55)
for xi, v in zip(x, values):
    ax.text(xi, v + 1, f'{v:.1f}%', ha='center', fontsize=10.5, color='#444441')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Flip rate on 2016 defective cohort (%)')
ax.set_ylim(0, 32)
ax.set_title('Does recalibration recover counterfactual transferability?\n'
             '(each pipeline uses its own model AND its own reference CFs)',
             fontsize=11.5, pad=10)

# In-distribution reference for comparison context
ax.text(len(labels) - 0.5, 30,
        'For context: in-distribution late-2014 flip rate = 98.2%',
        ha='right', fontsize=9, color='#666', style='italic')

ax.grid(axis='y', alpha=0.4, linewidth=0.5)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'recalibration.{ext}'), dpi=300, bbox_inches='tight')
plt.close()
print('Wrote: figures/paper_ready/recalibration.{png,pdf}')
