"""Build figs #2, #3, #4, #5, #6 from cached compute results."""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from _cf_lib import (load_data, train_model, FEATURES, ROOT, CACHE,
                     PROJECT_LIMITS, compute_cf_bounds)

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

# Load all caches
df       = pickle.load(open(os.path.join(CACHE, 'data.pkl'), 'rb'))
model    = pickle.load(open(os.path.join(CACHE, 'model_base.pkl'), 'rb'))
recal    = pickle.load(open(os.path.join(CACHE, 'model_recal.pkl'), 'rb'))
cf_2014  = pickle.load(open(os.path.join(CACHE, 'cf_2014_base.pkl'), 'rb'))
cf_2015  = pickle.load(open(os.path.join(CACHE, 'cf_2015_base.pkl'), 'rb'))
cf_2016  = pickle.load(open(os.path.join(CACHE, 'cf_2016_base.pkl'), 'rb'))
cf_2016r = pickle.load(open(os.path.join(CACHE, 'cf_2016_recal.pkl'), 'rb'))
panel    = pickle.load(open(os.path.join(CACHE, 'persistence.pkl'), 'rb'))

print('Cached counts:', {'2014': len(cf_2014), '2015': len(cf_2015),
                          '2016': len(cf_2016), '2016_recal': len(cf_2016r)})

# ===========================================================================
# Fig #3 — Worked example: one 2014 CF in physical units
# ===========================================================================
print('\n[#3] worked CF example')
# Pick a median-severity sample with a clean WARP-only intervention
cf2014_warp_only = cf_2014[(cf_2014['cf_warp_abs'] > 0.02) &
                            (cf_2014['cf_warp_abs'] < 0.08)].copy()
cf2014_warp_only['total_abs'] = (cf2014_warp_only[[f'd_{f}' for f in FEATURES]]
                                  .abs().sum(axis=1))
example = cf2014_warp_only.sort_values('total_abs').iloc[len(cf2014_warp_only)//2]

x0 = np.array([example[f'x0_{f}'] for f in FEATURES])
cf = np.array([example[f'cf_{f}'] for f in FEATURES])
d  = cf - x0

# Diagnostic: which features moved more than 1% of their bound range
cf_bounds = compute_cf_bounds(df)
ranges = np.array([cf_bounds[f][1] - cf_bounds[f][0] for f in FEATURES])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))

# Left: bar chart of original vs CF for each feature
ax = axes[0]
xpos = np.arange(len(FEATURES))
w = 0.38
ax.bar(xpos - w/2, x0, width=w, color='#dd8452', label='Original (defective)', edgecolor='white')
ax.bar(xpos + w/2, cf, width=w, color='#4c72b0', label='Counterfactual',       edgecolor='white')
ax.set_xticks(xpos); ax.set_xticklabels(FEATURES, fontsize=10)
ax.set_ylabel('Feature value (in)')
ax.set_title('Geometry feature values: original vs CF')
ax.legend(loc='upper left', framealpha=0.9, edgecolor='#cccccc')
# annotate change
for i, (a, b) in enumerate(zip(x0, cf)):
    dv = b - a
    if abs(dv) > 0.001:
        col = '#2a7d2a' if dv < 0 else '#c44e52'
        ax.annotate(f'Δ {dv:+.3f}', (xpos[i], max(a, b)),
                    textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=8.5, color=col, fontweight='bold')

# Right: WARP-focused zoom showing the project limit
ax = axes[1]
warp_idx = FEATURES.index('WARP_62')
warp0, warpcf = x0[warp_idx], cf[warp_idx]
ax.axvspan(-0.30, 0.30, alpha=0.12, color='#2a7d2a', label='Compliant region')
ax.axvline(-0.30, color='#666', linestyle='--', linewidth=1)
ax.axvline( 0.30, color='#666', linestyle='--', linewidth=1, label='Project limit ±0.30 in')
ax.scatter([warp0],  [1], s=200, color='#dd8452', zorder=5, edgecolor='black',
            label=f'Original ({warp0:+.3f} in)')
ax.scatter([warpcf], [1], s=200, color='#4c72b0', zorder=5, edgecolor='black',
            label=f'CF ({warpcf:+.3f} in)')
ax.annotate('', xy=(warpcf, 1), xytext=(warp0, 1),
            arrowprops=dict(arrowstyle='->', color='#444', lw=2))
ax.text((warp0 + warpcf)/2, 1.12, f'minimum WARP intervention\n|Δ| = {abs(warpcf-warp0):.3f} in',
        ha='center', fontsize=10, color='#444441')
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(0.5, 1.5)
ax.set_xlabel('WARP_62 (in)')
ax.set_yticks([])
ax.set_title('Minimum WARP perturbation to flip model decision')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='#cccccc')

mile = int(example.get('MILE', -1)) if pd.notna(example.get('MILE')) else 'unknown'
feet = int(example.get('FEET', -1)) if pd.notna(example.get('FEET')) else 'unknown'
fig.suptitle(f'Worked counterfactual example  —  segment MILE {mile}, FEET {feet}, {example["date"].strftime("%b %Y")}',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, 'cf_worked_example')

# ===========================================================================
# Fig #4 — Stale CF example: 2014 mean delta applied to a 2016 sample
# ===========================================================================
print('\n[#4] stale CF example')
mean_delta = cf_2014[[f'd_{f}' for f in FEATURES]].mean().values

# Pick a 2016 defective sample whose CF (under base model) failed to flip
# i.e. one where applying mean delta keeps it defective
def_2016 = df[(df['year'] == 2016) & (df['defect'] == 1)].copy()
# Try the first defective sample where p_cf_post > 0.5
candidates = []
for _, row in def_2016.head(100).iterrows():
    x0_2016 = row[FEATURES].values.astype(float)
    x_shifted = x0_2016 + mean_delta
    p_shifted = float(model.predict_proba(x_shifted.reshape(1,-1))[0,1])
    if p_shifted > 0.5:   # still defective after applying 2014 CF delta
        candidates.append((row, x_shifted, p_shifted))
    if len(candidates) >= 5:
        break

example2016, x_shifted, p_shifted = candidates[0]
x0_2016 = example2016[FEATURES].values.astype(float)

# What WOULD have been a successful CF in 2016 for this sample?
# (Use the optimizer? Too slow inline. Just show project-limit boundary.)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))

# Left: bar chart of orig + 2014-suggested-CF for each feature
ax = axes[0]
xpos = np.arange(len(FEATURES))
w = 0.38
ax.bar(xpos - w/2, x0_2016, width=w, color='#dd8452',
       label='Original 2016 (defective)', edgecolor='white')
ax.bar(xpos + w/2, x_shifted, width=w, color='#999',
       label='After 2014 CF delta (still defective!)', edgecolor='white')
ax.set_xticks(xpos); ax.set_xticklabels(FEATURES, fontsize=10)
ax.set_ylabel('Feature value (in)')
ax.set_title('Applying the 2014 mean CF delta to a 2016 sample')
ax.legend(loc='upper left', framealpha=0.9, edgecolor='#cccccc')

# Right: WARP zoom showing the 2014 advice misses the 2016 boundary
ax = axes[1]
warp0  = x0_2016[warp_idx]
warpfix = x_shifted[warp_idx]

ax.axvspan(-0.30, 0.30, alpha=0.12, color='#2a7d2a', label='Compliant region')
ax.axvline(-0.30, color='#666', linestyle='--', linewidth=1, label='Project limit ±0.30 in')
ax.axvline( 0.30, color='#666', linestyle='--', linewidth=1)
ax.scatter([warp0],  [1], s=200, color='#dd8452', zorder=5, edgecolor='black',
            label=f'2016 original ({warp0:+.3f} in)')
ax.scatter([warpfix], [1], s=200, color='#999', zorder=5, edgecolor='black',
            label=f'After 2014 advice ({warpfix:+.3f} in)\nstill p={p_shifted:.2f}')
ax.annotate('', xy=(warpfix, 1), xytext=(warp0, 1),
            arrowprops=dict(arrowstyle='->', color='#c44e52', lw=2))
ax.text((warp0 + warpfix)/2, 1.12,
        f'2014 advice fell short:\nstill outside compliant region',
        ha='center', fontsize=10, color='#c44e52')
ax.set_xlim(-0.7, 0.7)
ax.set_ylim(0.5, 1.5)
ax.set_xlabel('WARP_62 (in)')
ax.set_yticks([])
ax.set_title('Stale CF: 2014-derived intervention does not flip 2016 sample')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='#cccccc')

mile = int(example2016.get('MILE', -1)) if pd.notna(example2016.get('MILE')) else 'unknown'
feet = int(example2016.get('FEET', -1)) if pd.notna(example2016.get('FEET')) else 'unknown'
fig.suptitle(f'Failed counterfactual transfer  —  segment MILE {mile}, FEET {feet}, {example2016["date"].strftime("%b %Y")}',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, 'cf_stale_example')

# ===========================================================================
# Fig #2 — Cross-year AUC: does the CF advantage decay across cohorts?
# ===========================================================================
print('\n[#2] cross-year AUC comparison')

def build_cohort_table(cf_cohort, panel_df, year):
    """Merge CFs with persistence labels and predictor columns."""
    persist_year = panel_df[panel_df['year'] == year].copy()
    persist_year['key'] = list(zip(persist_year['MILE'], persist_year['FEET'],
                                    persist_year['date']))
    cf_cohort = cf_cohort.copy()
    cf_cohort['key'] = list(zip(cf_cohort['MILE'], cf_cohort['FEET'],
                                 cf_cohort['date']))
    keys_cf = set(cf_cohort['key'])
    pp = persist_year[persist_year['key'].isin(keys_cf)].copy()
    # dist-to-limit per row
    def dtl(row):
        ex = []
        for f, (lo, hi) in PROJECT_LIMITS.items():
            x = row[f]
            if f == 'GAGE':
                ex.append(max(0.0, lo - x, x - hi))
            else:
                ex.append(max(0.0, abs(x) - hi))
        return max(ex)
    pp['dist_to_limit'] = pp.apply(dtl, axis=1)
    # local TQI (std within (date, MILE))
    grp = pp.groupby(['date', 'MILE'])
    tqi_cols = []
    for f in FEATURES:
        pp[f'tqi_{f}'] = grp[f].transform('std')
        tqi_cols.append(f'tqi_{f}')
    pp['local_tqi'] = pp[tqi_cols].mean(axis=1)
    # Merge CF severity by key
    cf_lite = cf_cohort[['key', 'cf_warp_abs', 'cf_dist']]
    merged = pp.merge(cf_lite, on='key', how='inner')
    return merged

def fit_auc(X, y):
    valid = X.notna().all(axis=1) & y.notna()
    Xv = X.loc[valid]; yv = y.loc[valid].astype(int)
    if len(np.unique(yv)) < 2 or len(yv) < 20:
        return np.nan, len(yv)
    clf = LogisticRegression(max_iter=5000, solver='liblinear')
    clf.fit(Xv, yv)
    return roc_auc_score(yv, clf.predict_proba(Xv)[:, 1]), len(yv)

cohort_aucs = {}
for year, cf_y in [(2014, cf_2014), (2015, cf_2015), (2016, cf_2016)]:
    tbl = build_cohort_table(cf_y, panel, year)
    print(f'  cohort {year}: {len(tbl)} merged rows')
    results = {}
    for pred_name, X_cols in [
        ('CF |ΔWARP|',     ['cf_warp_abs']),
        ('Dist-to-limit',  ['dist_to_limit']),
        ('Local TQI',      ['local_tqi']),
    ]:
        auc, n = fit_auc(tbl[X_cols], tbl['persist_1m'])
        results[pred_name] = (auc, n)
        print(f'    {pred_name:15s} AUC={auc:.3f}  (n={n})')
    cohort_aucs[year] = results

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
years = sorted(cohort_aucs)
predictors = ['CF |ΔWARP|', 'Dist-to-limit', 'Local TQI']
colors = {'CF |ΔWARP|': '#d62728', 'Dist-to-limit': '#1f77b4', 'Local TQI': '#7f7f7f'}
markers = {'CF |ΔWARP|': 'o', 'Dist-to-limit': 's', 'Local TQI': '^'}

x = np.arange(len(years))
for p in predictors:
    vals = [cohort_aucs[y][p][0] for y in years]
    ax.plot(x, vals, marker=markers[p], markersize=10, linewidth=2.5,
            color=colors[p], label=p)
    for xi, v in zip(x, vals):
        if not np.isnan(v):
            ax.annotate(f'{v:.3f}', (xi, v), textcoords='offset points',
                        xytext=(7, 6), fontsize=9, color=colors[p])

ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years], fontsize=11)
ax.set_ylim(0.45, 1.02)
ax.set_xlabel('Cohort year')
ax.set_ylabel('AUC for 1-month persistence prediction')
ax.set_title('Does the CF advantage hold under drift?\n(predictors fit and evaluated within each cohort)',
             fontsize=11.5, pad=10)
ax.axhline(0.5, linestyle=':', color='#aaa', linewidth=0.8)
ax.legend(loc='lower left', framealpha=0.95, edgecolor='#cccccc')
plt.tight_layout()
save(fig, 'cross_year_auc')

# ===========================================================================
# Fig #5 — Recalibration: flip rate on 2016 cohort, base vs recalibrated model
# ===========================================================================
print('\n[#5] recalibration')

def flip_rate_under(model_, cf_df_, df_panel, year_target, mean_delta):
    """Apply mean_delta to all defective samples in year_target; return flip rate."""
    target = df_panel[(df_panel['year'] == year_target) & (df_panel['defect'] == 1)]
    X0 = target[FEATURES].values.astype(float)
    Xshift = X0 + mean_delta
    preds = model_.predict(Xshift)
    return (preds == 0).mean() * 100

# Base model: 2014 mean delta → 2016
mean_delta_base = cf_2014[[f'd_{f}' for f in FEATURES]].mean().values
fr_base_2016 = flip_rate_under(model, cf_2014, df, 2016, mean_delta_base)

# Recal model: use CFs computed under recalibrated model on 2015 cohort
# (we need 2015 reference CFs under the recal model)
# Approximation: compute mean delta from cf_2016r and apply same to a 2017 cohort
# We don't have 2017, so the cleanest comparison is:
#   - base model: 2014 mean delta → 2016 (gives the standard 6.9% number)
#   - base model: 2015 mean delta → 2016
#   - recal model: 2014 mean delta → 2016 (using recal as model)
#   - recal model: 2015 mean delta → 2016
mean_delta_2015 = cf_2015[[f'd_{f}' for f in FEATURES]].mean().values

fr_base_2014ref  = flip_rate_under(model,  cf_2014, df, 2016, mean_delta_base)
fr_base_2015ref  = flip_rate_under(model,  cf_2015, df, 2016, mean_delta_2015)
fr_recal_2014ref = flip_rate_under(recal,  cf_2014, df, 2016, mean_delta_base)
fr_recal_2015ref = flip_rate_under(recal,  cf_2015, df, 2016, mean_delta_2015)

print(f'  Base model,  2014 ref CF → 2016: {fr_base_2014ref:.1f}%')
print(f'  Base model,  2015 ref CF → 2016: {fr_base_2015ref:.1f}%')
print(f'  Recal model, 2014 ref CF → 2016: {fr_recal_2014ref:.1f}%')
print(f'  Recal model, 2015 ref CF → 2016: {fr_recal_2015ref:.1f}%')

fig, ax = plt.subplots(figsize=(8.5, 5))
labels = ['Base model\n(train 2013–14)', 'Recalibrated model\n(train 2014–15)']
yref_2014  = [fr_base_2014ref,  fr_recal_2014ref]
yref_2015  = [fr_base_2015ref,  fr_recal_2015ref]

xpos = np.arange(len(labels))
w = 0.35
ax.bar(xpos - w/2, yref_2014, width=w, color='#dd8452',
       label='Reference CFs from 2014', edgecolor='white')
ax.bar(xpos + w/2, yref_2015, width=w, color='#4c72b0',
       label='Reference CFs from 2015', edgecolor='white')

for i, v in enumerate(yref_2014):
    ax.text(xpos[i] - w/2, v + 1, f'{v:.1f}%', ha='center', fontsize=9.5, color='#444441')
for i, v in enumerate(yref_2015):
    ax.text(xpos[i] + w/2, v + 1, f'{v:.1f}%', ha='center', fontsize=9.5, color='#444441')

ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Flip rate on 2016 defective cohort (%)')
ax.set_ylim(0, max(yref_2014 + yref_2015) * 1.25)
ax.set_title('Does recalibration recover counterfactual transferability?\n'
             '(mean Δ applied to every 2016 defective sample)',
             fontsize=11.5, pad=10)
ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc')
ax.grid(axis='y', alpha=0.4, linewidth=0.5)
plt.tight_layout()
save(fig, 'recalibration')

# ===========================================================================
# Fig #6 — Spatial localisation: per-window CF flip rate by year
# ===========================================================================
print('\n[#6] spatial localisation')

# Compute per-(window, year) flip rate. Use 200-ft windows like the TQI heatmap.
def add_window_flips(cf_df_year, year, mean_delta):
    target = df[(df['year'] == year) & (df['defect'] == 1)].copy()
    target['window_200ft'] = (target['FEET'].astype(int) // 200)
    X0 = target[FEATURES].values.astype(float)
    Xs = X0 + mean_delta
    target['flipped'] = (model.predict(Xs) == 0).astype(int)
    return target.groupby('window_200ft')['flipped'].agg(['mean', 'count']).reset_index()

# Use 2014 mean delta (the reference protocol)
spatial = {}
for y, label in [(2014, 'Late 2014 (in-distribution)'),
                 (2015, '2015'),
                 (2016, '2016')]:
    spatial[label] = add_window_flips(cf_2014, y, mean_delta_base)

# Build a matrix: windows × years, flip rate
all_windows = sorted(set().union(*(set(t['window_200ft']) for t in spatial.values())))
periods = list(spatial.keys())
M = np.full((len(all_windows), len(periods)), np.nan)
N = np.full((len(all_windows), len(periods)), np.nan)
for j, lbl in enumerate(periods):
    t = spatial[lbl].set_index('window_200ft')
    for i, w in enumerate(all_windows):
        if w in t.index:
            M[i, j] = t.loc[w, 'mean'] * 100
            N[i, j] = t.loc[w, 'count']

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(M, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(periods))); ax.set_xticklabels(periods, fontsize=10)
ax.set_yticks(range(len(all_windows))); ax.set_yticklabels(all_windows, fontsize=8)
ax.set_ylabel('200-ft window ID (along corridor)')
ax.set_title('Per-window CF flip rate decay across periods\n(2014 mean Δ applied uniformly; darker red = stale)',
             fontsize=11.5, pad=10)

cbar = plt.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label('Flip rate (%)')

# Annotate cells
for i in range(len(all_windows)):
    for j in range(len(periods)):
        v = M[i, j]
        n_ = N[i, j]
        if np.isnan(v): continue
        text = f'{v:.0f}'
        col = 'white' if v < 40 else 'black'
        ax.text(j, i, text, ha='center', va='center', fontsize=7, color=col)

plt.tight_layout()
save(fig, 'spatial_flip_rate')

print('\nAll figures done.')
