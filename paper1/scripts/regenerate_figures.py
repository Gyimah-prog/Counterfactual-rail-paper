"""Regenerate polished publication figures into figures/paper_ready/.

Each figure is built standalone from data/RawD.xlsx (no notebook re-run).
Hard-coded stats (XGBoost importance, PID, MI) match the executed notebook.
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, 'figures', 'paper_ready')
os.makedirs(OUT, exist_ok=True)

# ─── Style (matches notebook cell 3) ────────────────────────────────────────
sns.set_theme(style='whitegrid', context='paper', font_scale=1.15)
sns.set_palette('colorblind')
plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica'],
    'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.labelsize': 11, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'axes.linewidth': 1.0, 'axes.edgecolor': '#333333',
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': '#cccccc', 'grid.alpha': 0.55, 'grid.linewidth': 0.5,
    'grid.linestyle': '--',
})

def save(fig, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{stem}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {stem}.{{png,pdf}}')

def annotate_bars(ax, fmt='{:.3f}', dy=0.01, fontsize=9.5, color='#444441'):
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * dy
    for bar in ax.patches:
        h = bar.get_height()
        if np.isnan(h) or h == 0:
            text = '~0' if h == 0 else 'n/a'
        else:
            text = fmt.format(h)
        ax.text(bar.get_x() + bar.get_width()/2, h + pad,
                text, ha='center', va='bottom',
                fontsize=fontsize, color=color)

# ─── Load + preprocess (same pipeline as notebook §1–§2) ───────────────────
print('Loading data...')
xl = pd.read_excel(os.path.join(ROOT, 'data', 'RawD.xlsx'),
                   sheet_name=None, engine='openpyxl')
rx = re.compile(r'^(\d{1,2})_(\d{4})$')
frames = []
for name, d in xl.items():
    m = rx.match(name)
    if not m: continue
    mo, yr = map(int, m.groups())
    d = d.copy()
    d['date']  = pd.Timestamp(year=yr, month=mo, day=1)
    d['year']  = d['date'].dt.year
    d['month'] = d['date'].dt.month
    frames.append(d)
raw_df = pd.concat(frames, ignore_index=True).sort_values('date').reset_index(drop=True)
for c in ('MILE', 'FEET'):
    if c in raw_df.columns:
        raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce').astype('Int64')

FEATURES = ['GAGE', 'ALI_L_62', 'ALI_R_62', 'WARP_62', 'XLEVEL']
df = raw_df.dropna(subset=FEATURES).copy()

# Defect labels (project limits)
df['def_GAGE']   = (df['GAGE'] < 56.3) | (df['GAGE'] > 57.2)
df['def_ALI_L']  = df['ALI_L_62'].abs() > 0.20
df['def_ALI_R']  = df['ALI_R_62'].abs() > 0.20
df['def_WARP']   = df['WARP_62'].abs()  > 0.30
df['def_XLEVEL'] = df['XLEVEL'].abs()   > 0.80
df['defect']     = df[[c for c in df.columns if c.startswith('def_')]].any(axis=1).astype(int)
print(f'  {len(df):,} rows, {df["defect"].sum():,} defects')

# ============================================================================
# 1) defect_triggers — horizontal bars, value-labeled
# ============================================================================
print('1) defect_triggers')
trig = {
    'WARP_62':  int(df['def_WARP'].sum()),
    'XLEVEL':   int(df['def_XLEVEL'].sum()),
    'ALI_R_62': int(df['def_ALI_R'].sum()),
    'ALI_L_62': int(df['def_ALI_L'].sum()),
    'GAGE':     int(df['def_GAGE'].sum()),
}
labels = list(trig.keys()); vals = list(trig.values())

fig, ax = plt.subplots(figsize=(8, 4.2))
colors = ['#1f77b4'] * len(labels)
colors[0] = '#d62728'  # highlight WARP
bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], edgecolor='white')
ax.set_xlabel('Number of triggered defects')
ax.set_title('Defect triggers by feature  (overlap allowed)')
ax.set_xlim(0, max(vals) * 1.15)
for i, v in enumerate(vals[::-1]):
    ax.text(v + max(vals)*0.012, i, f'{v:,}', va='center', fontsize=9.5, color='#444441')
ax.grid(axis='x', alpha=0.4, linewidth=0.5)
sns.despine(ax=ax, left=True)
ax.tick_params(axis='y', length=0)
save(fig, 'defect_triggers')

# ============================================================================
# 2) correlation_matrix — symmetric vmin/vmax, mask upper triangle
# ============================================================================
print('2) correlation_matrix')
corr = df[FEATURES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f',
            cmap='RdBu_r', vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.75, 'label': 'Pearson r'})
ax.set_title('Feature correlation matrix')
plt.setp(ax.get_xticklabels(), rotation=0)
plt.setp(ax.get_yticklabels(), rotation=0)
save(fig, 'correlation_matrix')

# ============================================================================
# 3) feature_distributions — 1×5 violin (REPLACES cell13)
# ============================================================================
print('3) feature_distributions (violins)')
long_df = df[FEATURES + ['defect']].melt(id_vars='defect',
                                          var_name='feature',
                                          value_name='value')
long_df['defect_label'] = long_df['defect'].map({0: 'Non-defective', 1: 'Defective'})

# Per-feature limits for y-axis trimming (drop extreme outliers visually)
fig, axes = plt.subplots(1, 5, figsize=(15, 4.2), sharey=False)
palette = {'Non-defective': '#4c72b0', 'Defective': '#dd8452'}

for ax, feat in zip(axes, FEATURES):
    sub = long_df[long_df['feature'] == feat]
    sns.violinplot(data=sub, x='defect_label', y='value', hue='defect_label',
                   palette=palette, ax=ax, inner='quartile',
                   linewidth=0.8, cut=0, legend=False)
    ax.set_title(feat, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Value' if feat == FEATURES[0] else '')
    # trim to 1st/99th percentile so extreme outliers don't squash the violin
    lo, hi = sub['value'].quantile([0.005, 0.995])
    ax.set_ylim(lo, hi)
    ax.grid(axis='y', alpha=0.4, linewidth=0.5)
    sns.despine(ax=ax)

handles = [mpatches.Patch(color=palette['Non-defective'], label='Non-defective'),
           mpatches.Patch(color=palette['Defective'],     label='Defective')]
fig.legend(handles=handles, loc='upper center', ncol=2,
           frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.suptitle('Geometry feature distributions — defective vs non-defective',
             y=1.08, fontsize=12, fontweight='bold')
plt.tight_layout()
save(fig, 'feature_distributions')

# ============================================================================
# 4) warp_kde_by_year — line-only KDE, bold 2016
# ============================================================================
print('4) warp_kde_by_year')
fig, ax = plt.subplots(figsize=(8.5, 5))
year_colors = {2013: '#4c72b0', 2014: '#55a868',
               2015: '#dd8452', 2016: '#c44e52'}
for yr in [2013, 2014, 2015, 2016]:
    sub = df[df['year'] == yr]['WARP_62']
    lw  = 2.5 if yr == 2016 else 1.5
    a   = 1.0 if yr == 2016 else 0.85
    sns.kdeplot(sub, ax=ax, label=str(yr),
                color=year_colors[yr], linewidth=lw, alpha=a,
                bw_adjust=0.8)
ax.axvline(-0.30, color='#666', linestyle='--', linewidth=1, label='±0.30 limit')
ax.axvline( 0.30, color='#666', linestyle='--', linewidth=1)
ax.set_xlim(-0.7, 0.7)
ax.set_xlabel('WARP_62 (in)')
ax.set_ylabel('Density')
ax.set_title('WARP_62 distribution drift across years\n(2016 highlighted — fattest tails)')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='#cccccc')
sns.despine(ax=ax)
save(fig, 'warp_kde_by_year')

# ============================================================================
# 5) mutual_information — value-labeled
# ============================================================================
print('5) mutual_information')
mi_vals = {'WARP_62': 0.0879, 'XLEVEL': 0.0492,
           'GAGE': 0.0138, 'ALI_R_62': 0.0123, 'ALI_L_62': 0.0120}
feats = list(mi_vals.keys()); mi = list(mi_vals.values())
fig, ax = plt.subplots(figsize=(8, 4.5))
shades = sns.color_palette('Blues_r', n_colors=len(feats))
bars = ax.bar(feats, mi, color=shades, edgecolor='white')
ax.set_ylabel('Mutual information (nats)')
ax.set_title('Mutual information of each geometry feature with defect status')
ax.set_ylim(0, max(mi) * 1.18)
annotate_bars(ax, fmt='{:.4f}', dy=0.012)
ax.grid(axis='y', alpha=0.4, linewidth=0.5)
sns.despine(ax=ax)
save(fig, 'mutual_information')

# ============================================================================
# 6) pid_decomposition — value-labeled, two panels
# ============================================================================
print('6) pid_decomposition')
pid_comp = ['Redundant', 'Unique WARP', 'Unique XLEVEL', 'Synergy']
pid_mi   = [0.0481, 0.0399, 0.0000, 0.0005]
pid_pct  = [54.3, 45.1, 0.0, 0.5]
colors4  = ['#4c72b0', '#dd8452', '#55a868', '#8172b3']

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, vals, ylabel, fmt in [
    (axes[0], pid_mi,  'Mutual information (nats)', '{:.4f}'),
    (axes[1], pid_pct, '% of total information',    '{:.1f}%'),
]:
    bars = ax.bar(pid_comp, vals, color=colors4, edgecolor='white')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(vals) * 1.20 if max(vals) > 0 else 1)
    annotate_bars(ax, fmt=fmt, dy=0.015)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.4, linewidth=0.5)
    sns.despine(ax=ax)

axes[0].set_title('Absolute MI')
axes[1].set_title('Share of total')
fig.suptitle('Partial information decomposition: (WARP, XLEVEL) → defect',
             y=1.04, fontweight='bold')
plt.tight_layout()
save(fig, 'pid_decomposition')

# ============================================================================
# 7) xgb_importance — value-labeled
# ============================================================================
print('7) xgb_importance')
imp = {'WARP_62': 0.4527, 'XLEVEL': 0.2109,
       'ALI_L_62': 0.1580, 'ALI_R_62': 0.1477, 'GAGE': 0.0306}
fig, ax = plt.subplots(figsize=(8, 4.5))
shades = sns.color_palette('Blues_r', n_colors=len(imp))
bars = ax.bar(list(imp.keys()), list(imp.values()),
              color=shades, edgecolor='white')
ax.set_ylabel('Gain importance')
ax.set_title('XGBoost feature importance (trained on 2013–2014)')
ax.set_ylim(0, max(imp.values()) * 1.18)
annotate_bars(ax, fmt='{:.3f}', dy=0.012)
ax.grid(axis='y', alpha=0.4, linewidth=0.5)
sns.despine(ax=ax)
save(fig, 'xgb_importance')

# ============================================================================
# 8) tqi_heatmap_200ft — cleaner row labels, better aspect
# ============================================================================
print('8) tqi_heatmap_200ft')
df['window_200ft'] = (df['FEET'].astype(int) // 200)
tqi = (df.groupby(['date', 'window_200ft'])['WARP_62']
         .std().unstack('window_200ft').T)
fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(tqi, cmap='YlOrRd', ax=ax,
            cbar_kws={'shrink': 0.7, 'label': 'WARP_62 std-dev (in)'},
            linewidths=0)
ax.set_xlabel('Inspection date')
ax.set_ylabel('200-ft window ID')
ax.set_title('Per-window WARP variability over time\n(darker = more deterioration)')
# Show every 3rd date and every 2nd window
xt = ax.get_xticks()
xl = [t.get_text() for t in ax.get_xticklabels()]
keep_x = list(range(0, len(xl), 3))
ax.set_xticks([xt[i] for i in keep_x])
ax.set_xticklabels([xl[i][:7] for i in keep_x], rotation=45, ha='right')
yt = ax.get_yticks()
yl = [t.get_text() for t in ax.get_yticklabels()]
keep_y = list(range(0, len(yl), 2))
ax.set_yticks([yt[i] for i in keep_y])
ax.set_yticklabels([yl[i] for i in keep_y], rotation=0)
save(fig, 'tqi_heatmap_200ft')

print('\nDone. All files written to:', OUT)
