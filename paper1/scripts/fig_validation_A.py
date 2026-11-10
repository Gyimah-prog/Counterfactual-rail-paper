"""Figure 4 (validation_A) -- severity predictor comparison, 2014 reference cohort.

Left : Spearman correlation of each predictor with the number of defective
       observations among the next three available inspections.
Right: ROC AUC for defect occurrence at the next inspection and within the
       next three and six available inspections.

Predictors: counterfactual |dWARP|, CF distance, distance to limit,
            TQI (section: one value per inspection over the one mile section),
            TQI (200 ft: per inspection within 200 ft blocks).
Joint models (logistic regression on standardized predictors, no penalty):
            D2 = distance to limit + TQI (200 ft)
            E3 = D2 + counterfactual |dWARP|

Inputs (canonical nb cache): data.pkl, persistence.pkl, cf_2014_base_nb.pkl
Output: figures/paper_ready/validation_A.{png,pdf}
Also prints every value used in Table 6 and Table 7 of the paper.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from _cf_lib import FEATURES, ROOT, CACHE, PROJECT_LIMITS

OUT = os.path.join(ROOT, 'figures', 'paper_ready')
os.makedirs(OUT, exist_ok=True)

df    = pickle.load(open(os.path.join(CACHE, 'data.pkl'), 'rb'))
panel = pickle.load(open(os.path.join(CACHE, 'persistence.pkl'), 'rb'))
cf    = pickle.load(open(os.path.join(CACHE, 'cf_2014_base_nb.pkl'), 'rb'))

# ---- TQI at two scales, computed on all measurements --------------------
d = df[['date', 'MILE', 'FEET'] + FEATURES].copy()
d['FEET'] = d['FEET'].astype(int)
d['block200'] = d['FEET'] // 200
g_sec = d.groupby(['date', 'MILE'])
g_200 = d.groupby(['date', 'block200'])
d['tqi_section'] = sum(g_sec[f].transform('std') for f in FEATURES) / len(FEATURES)
d['tqi_200ft']   = sum(g_200[f].transform('std') for f in FEATURES) / len(FEATURES)

# ---- distance to limit ---------------------------------------------------
def dtl(x0):
    ex = []
    for j, f in enumerate(FEATURES):
        lo, hi = PROJECT_LIMITS[f]
        ex.append(max(0.0, lo - x0[j], x0[j] - hi) if f == 'GAGE'
                  else max(0.0, abs(x0[j]) - hi))
    return max(ex)

ev = cf.copy()
ev['FEET'] = ev['FEET'].astype(int); ev['MILE'] = ev['MILE'].astype(int)
ev['dist_to_limit'] = [dtl(r) for r in ev[[f'x0_{f}' for f in FEATURES]].values]
pp = panel[['MILE', 'FEET', 'date', 'persist_1m', 'persist_3m', 'persist_6m',
            'persist_count_3m']].copy()
pp['FEET'] = pp['FEET'].astype(int); pp['MILE'] = pp['MILE'].astype(int)
d['MILE'] = d['MILE'].astype(int)
ev = ev.merge(pp, on=['MILE', 'FEET', 'date'], how='left')
ev = ev.merge(d[['MILE', 'FEET', 'date', 'tqi_section', 'tqi_200ft']],
              on=['MILE', 'FEET', 'date'], how='left')
print(f'n = {len(ev)}')

OUTCOMES = ['persist_1m', 'persist_3m', 'persist_6m']

def auc_single(col):
    out = []
    for o in OUTCOMES:
        s = ev[[col, o]].dropna()
        out.append(roc_auc_score(s[o].astype(int), s[col]))
    return out

def auc_joint(cols):
    out = []
    for o in OUTCOMES:
        s = ev[cols + [o]].dropna()
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=20000, C=1e6))
        m.fit(s[cols], s[o].astype(int))
        out.append(roc_auc_score(s[o].astype(int), m.predict_proba(s[cols])[:, 1]))
    return out

auc = {
    'C2': auc_single('cf_warp_abs'),
    'B2': auc_single('tqi_200ft'),
    'C':  auc_single('cf_dist'),
    'A':  auc_single('dist_to_limit'),
    'B':  auc_single('tqi_section'),
    'D2': auc_joint(['dist_to_limit', 'tqi_200ft']),
    'E3': auc_joint(['dist_to_limit', 'tqi_200ft', 'cf_warp_abs']),
}
print('\nTable 7 -- AUC (next 1 / 3 / 6 inspections)')
for k, v in auc.items():
    print(f'  {k:3s} ' + ' / '.join(f'{x:.3f}' for x in v))

RHO_ROWS = [('|ΔWARP| (CF)', 'cf_warp_abs'), ('TQI (200 ft)', 'tqi_200ft'),
            ('CF distance', 'cf_dist'), ('Distance to limit', 'dist_to_limit'),
            ('TQI (section)', 'tqi_section')]
rho = {}
print('\nTable 6 -- Spearman with defect count in next three inspections')
for lab, col in RHO_ROWS:
    s = ev[[col, 'persist_count_3m']].dropna()
    rho[lab] = spearmanr(s[col], s['persist_count_3m'])
    print(f'  {lab:18s} rho = {rho[lab][0]:.3f}   p = {rho[lab][1]:.3g}')

# ---- plot ----------------------------------------------------------------
sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 150})
C_CF, C_DIST, C_COMBO, C_TQI = '#D4537E', '#EF9F27', '#185FA5', '#888780'

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
labels = [r[0] for r in RHO_ROWS]
vals   = [rho[k][0] for k in labels]
cols   = [C_CF, C_TQI, C_CF, C_DIST, C_TQI]
y = np.arange(len(labels))
ax.barh(y, vals[::-1], color=cols[::-1], edgecolor='white', height=0.55)
ax.set_yticks(y); ax.set_yticklabels(labels[::-1], fontsize=10)
ax.set_xlim(0, 0.72)
ax.set_xlabel('Spearman ρ with defect count in next three inspections', fontsize=10)
ax.set_title('(A) Predictor and persistence correlation', fontsize=10, pad=8,
             fontweight='bold')
ax.axvline(0, color='#B4B2A9', linewidth=0.8)
for i, k in enumerate(labels[::-1]):
    r, p = rho[k]
    star = '***' if p < 0.001 else ('*' if p < 0.05 else 'n.s.')
    ax.text(r + 0.012, i, f'ρ={r:.3f}  {star}', va='center', fontsize=9,
            color='#444441')
ax.grid(axis='x', linewidth=0.5, alpha=0.5)
sns.despine(ax=ax, left=True)

ax = axes[1]
x = np.arange(3)
SERIES = [('E3', 'Dist + TQI (200 ft) + CF |ΔWARP| (E3)', C_COMBO, 'D', '-'),
          ('C2', 'CF |ΔWARP| only (C2)',                  C_CF,    's', '-'),
          ('B2', 'TQI (200 ft) only (B2)',                C_TQI,   '^', '-'),
          ('A',  'Distance to limit only (A)',            C_DIST,  'o', '--'),
          ('B',  'TQI (section) only (B)',                C_TQI,   'v', ':')]
OFFSET = {'E3': 4, 'C2': -8, 'B2': -2, 'A': 0, 'B': 0}
for k, lab, col, m, ls in SERIES:
    ax.plot(x, auc[k], marker=m, linewidth=2.5, markersize=8, color=col,
            linestyle=ls, label=lab)
    ax.annotate(f'{auc[k][-1]:.3f}', (2, auc[k][-1]), textcoords='offset points',
                xytext=(8, OFFSET[k]), fontsize=8.5, color=col, va='center')
ax.set_xticks(x)
ax.set_xticklabels(['Next inspection', 'Next 3 inspections', 'Next 6 inspections'],
                   fontsize=10)
ax.set_xlim(-0.15, 2.35)
ax.set_ylim(0.45, 0.93)
ax.axhline(0.5, color='#B4B2A9', linewidth=0.8)
ax.text(1.0, 0.493, 'chance level', fontsize=7.5, color='#888780',
        va='top', ha='center')
ax.set_ylabel('ROC AUC', fontsize=10)
ax.set_title('(B) AUC for subsequent defect occurrence\nacross inspection horizons',
             fontsize=10, pad=8, fontweight='bold')
ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='#D3D1C7', loc='upper center',
          bbox_to_anchor=(0.5, -0.10), ncol=2)
ax.grid(axis='y', linewidth=0.5, alpha=0.5)
sns.despine(ax=ax)

plt.suptitle('CF-derived severity metric vs traditional predictors: '
             'defect persistence prediction', fontsize=11, y=1.02)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'validation_A.{ext}'), dpi=300,
                bbox_inches='tight')
plt.close()
print('\nWrote: figures/paper_ready/validation_A.{png,pdf}')
