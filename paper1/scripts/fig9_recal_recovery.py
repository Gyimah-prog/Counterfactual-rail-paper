"""Fig #9 — Does recalibration restore CF severity as a predictor?

For each cohort year, compute the AUC of CF |ΔWARP| against 1-month
persistence under:
  (a) base model (frozen on 2013-14)
  (b) recal model (refit on 2014-15)
  (c) [optional] recal2 model (refit on 2015-16; only for 2016 cohort)

Dist-to-limit AUC (which doesn't depend on the model) is plotted as a
fixed baseline.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from _cf_lib import FEATURES, ROOT, CACHE, PROJECT_LIMITS

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
panel = pickle.load(open(os.path.join(CACHE, 'persistence.pkl'), 'rb'))

cfs = {
    ('base',  2014): pickle.load(open(os.path.join(CACHE, 'cf_2014_base.pkl'),  'rb')),
    ('base',  2015): pickle.load(open(os.path.join(CACHE, 'cf_2015_base.pkl'),  'rb')),
    ('base',  2016): pickle.load(open(os.path.join(CACHE, 'cf_2016_base.pkl'),  'rb')),
    ('recal', 2014): pickle.load(open(os.path.join(CACHE, 'cf_2014_recal.pkl'), 'rb')),
    ('recal', 2015): pickle.load(open(os.path.join(CACHE, 'cf_2015_recal.pkl'), 'rb')),
    ('recal', 2016): pickle.load(open(os.path.join(CACHE, 'cf_2016_recal.pkl'), 'rb')),
}
try:
    cfs[('recal2', 2016)] = pickle.load(open(os.path.join(CACHE, 'cf_2016_recal2.pkl'), 'rb'))
    has_recal2 = True
except FileNotFoundError:
    has_recal2 = False
    print('  recal2 not yet cached — figure will be rebuilt later')

def dtl(row):
    ex = []
    for f, (lo, hi) in PROJECT_LIMITS.items():
        x = row[f]
        if f == 'GAGE':
            ex.append(max(0.0, lo - x, x - hi))
        else:
            ex.append(max(0.0, abs(x) - hi))
    return max(ex)

def cohort_auc(cf_df, year, horizon='persist_1m'):
    pp = panel[panel['year'] == year].copy()
    pp['key'] = list(zip(pp['MILE'], pp['FEET'], pp['date']))
    cf = cf_df.copy()
    cf['key'] = list(zip(cf['MILE'], cf['FEET'], cf['date']))
    merged = pp.merge(cf[['key', 'cf_warp_abs']], on='key', how='inner')
    merged['dist_to_limit'] = merged.apply(dtl, axis=1)
    grp = merged.groupby(['date', 'MILE'])
    for f in FEATURES:
        merged[f'tqi_{f}'] = grp[f].transform('std')
    merged['local_tqi'] = merged[[f'tqi_{f}' for f in FEATURES]].mean(axis=1)
    valid = merged[horizon].notna()
    sub = merged[valid]
    if len(sub) < 20 or sub[horizon].nunique() < 2:
        return np.nan, np.nan, np.nan, len(sub)
    y = sub[horizon].astype(int).values
    auc_cf   = roc_auc_score(y, sub['cf_warp_abs'].values)
    auc_dtl  = roc_auc_score(y, sub['dist_to_limit'].values)
    auc_tqi  = roc_auc_score(y, sub['local_tqi'].fillna(sub['local_tqi'].median()).values)
    return auc_cf, auc_dtl, auc_tqi, len(sub)

# ─── Build table ──────────────────────────────────────────────────────────
rows = []
for year in [2014, 2015, 2016]:
    for model_tag in ['base', 'recal']:
        cf = cfs.get((model_tag, year))
        if cf is None: continue
        auc_cf, auc_dtl, auc_tqi, n = cohort_auc(cf, year)
        rows.append({'year': year, 'model': model_tag,
                     'auc_cf': auc_cf, 'auc_dtl': auc_dtl, 'auc_tqi': auc_tqi, 'n': n})
if has_recal2:
    cf = cfs[('recal2', 2016)]
    auc_cf, auc_dtl, auc_tqi, n = cohort_auc(cf, 2016)
    rows.append({'year': 2016, 'model': 'recal2',
                 'auc_cf': auc_cf, 'auc_dtl': auc_dtl, 'auc_tqi': auc_tqi, 'n': n})

tbl = pd.DataFrame(rows)
print(tbl.to_string(index=False))

# ─── Plot ──────────────────────────────────────────────────────────────────
# Group bars per cohort year showing CF AUC under each model
YEARS = [2014, 2015, 2016]
model_labels = {'base':  'Base model\n(train 2013–14)',
                'recal': 'Recal model\n(train 2014–15)',
                'recal2':'Recal2 model\n(train 2015–16)'}
model_colors = {'base': '#1f77b4', 'recal': '#ff7f0e', 'recal2': '#2ca02c'}

fig, ax = plt.subplots(figsize=(10, 5.2))
bar_w = 0.22
x_base = np.arange(len(YEARS))

# Plot bars for each model
for i, mtag in enumerate(['base', 'recal', 'recal2']):
    if mtag == 'recal2' and not has_recal2: continue
    vals = []; xs = []
    for j, y in enumerate(YEARS):
        sub = tbl[(tbl['model'] == mtag) & (tbl['year'] == y)]
        if len(sub) == 0:
            continue
        vals.append(float(sub['auc_cf'].iloc[0]))
        xs.append(x_base[j] + (i - 1) * bar_w)
    ax.bar(xs, vals, width=bar_w, label=model_labels[mtag],
           color=model_colors[mtag], edgecolor='white')
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 0.012, f'{v:.3f}', ha='center', fontsize=9,
                color='#444441')

# Dist-to-limit and TQI reference lines per cohort (model-independent)
dtl_by_year = {y: float(tbl[(tbl['year'] == y) & (tbl['model'] == 'base')]['auc_dtl'].iloc[0])
               for y in YEARS}
tqi_by_year = {y: float(tbl[(tbl['year'] == y) & (tbl['model'] == 'base')]['auc_tqi'].iloc[0])
               for y in YEARS}
for j, y in enumerate(YEARS):
    ax.hlines(dtl_by_year[y], x_base[j] - 0.45, x_base[j] + 0.45,
              colors='#c44e52', linestyles='--', linewidth=1.8, zorder=4)
    ax.text(x_base[j] + 0.46, dtl_by_year[y], f'dist-to-limit: {dtl_by_year[y]:.3f}',
            va='center', fontsize=8.5, color='#c44e52')
    ax.hlines(tqi_by_year[y], x_base[j] - 0.45, x_base[j] + 0.45,
              colors='#7f7f7f', linestyles=':', linewidth=1.5, zorder=4)
    ax.text(x_base[j] + 0.46, tqi_by_year[y], f'TQI: {tqi_by_year[y]:.3f}',
            va='center', fontsize=8.5, color='#7f7f7f')

ax.axhline(0.5, linestyle=':', color='#aaa', linewidth=0.8)
ax.set_xticks(x_base); ax.set_xticklabels([str(y) for y in YEARS], fontsize=11)
ax.set_ylim(0.25, 1.0)
ax.set_xlabel('Cohort year')
ax.set_ylabel('AUC for 1-month persistence prediction')
ax.set_title('Does recalibrating the model restore CF |ΔWARP| as a predictor?\n'
             '(red dashes = dist-to-limit; gray dots = TQI;  both model-independent)',
             fontsize=11.5, pad=10)
ax.legend(loc='lower left', framealpha=0.95, edgecolor='#cccccc')
ax.grid(axis='y', alpha=0.4, linewidth=0.5)

plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'recal_recovery.{ext}'),
                dpi=300, bbox_inches='tight')
plt.close()
print('Wrote: figures/paper_ready/recal_recovery.{png,pdf}')
