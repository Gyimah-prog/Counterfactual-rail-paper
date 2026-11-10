"""All report figures, PNG (300 dpi) and PDF."""
import numpy as np, pandas as pd, pickle, json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from cf2lib import *
os.makedirs('figs', exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'figure.facecolor': 'white'})
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8984'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
A = pickle.load(open('out/analysis.pkl', 'rb')); T, best, lad = A['T'], A['best'], A['lad']
mon = pickle.load(open('out/monitor.pkl', 'rb')); stress = pickle.load(open('out/stress.pkl', 'rb'))
LBL = {'stale': 'Stale P2', 'inst_ratio': 'Instance (ratio)', 'inst_margin': 'Instance (margin)', 'sc_sigma': 'Scalar (spread)',
       'sc_tail': 'Scalar (tail)', 'sc_excess': 'Scalar (excess)', 'sc_mproxy': 'Scalar (model proxy)', 'dg_sigma': 'Diagonal (spread)',
       'dg_tail': 'Diagonal (tail)', 'dg_excess': 'Diagonal (excess)', 'ot_1d': 'OT (1D WARP)', 'ot_marg': 'OT (marginal)',
       'ot_mv': 'OT (multivariate)', 'ot_mv_ret': 'OT (mv, retrieval)', 'fresh': 'Fresh generation', 'ctrl_threshold': 'Threshold correction',
       'ctrl_p1': 'P1 mean', 'ctrl_none': 'No action'}
KEY = ['stale', best['Instance'], best['Scalar'], best['Diagonal'], best['OT'], 'fresh']
COL = dict(zip(KEY, [MUTED, CAT[0], CAT[1], CAT[2], CAT[6], INK]))
def save(fig, name):
    fig.savefig(f'figs/{name}.png'); fig.savefig(f'figs/{name}.pdf'); plt.close(fig)
def shade(ax):
    ax.axvspan(pd.Timestamp('2014-12-15'), pd.Timestamp('2015-06-15'), color='#f2f1ec', zorder=0)
    ax.axvspan(pd.Timestamp('2015-12-20'), pd.Timestamp('2016-04-15'), color='#e9eef7', zorder=0)

# Fig 1 architecture
fig, ax = plt.subplots(figsize=(11.5, 3.1)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
boxes = [(0.1, 'Reference', 'frozen classifier f0\n(trained 2013 to 2014)\n637 stored 2014 pairs\nbounds and labels'),
         (2.05, 'Monitor', 'past only window Ct\nvalidity, spread, tail,\nexcess, composition'),
         (4.0, 'Adapt', 'Instance  κ(x)δ\nScalar  αt δ\nDiagonal  At δ\nOT  Tt(xcf) − Tt(x0)'),
         (5.95, 'Validate', 'model flip\nproject compliance\nfeasibility in B'),
         (7.9, 'Refresh', 'accept first valid level\nelse regenerate\nelse unresolved')]
for x, h, b in boxes:
    ax.add_patch(FancyBboxPatch((x, 0.35), 1.8, 2.3, boxstyle='round,pad=0.02,rounding_size=0.08', fc='#f6f5f2', ec='#c9c8c2'))
    ax.text(x + 0.9, 2.35, h, ha='center', va='center', weight='bold', color=INK, fontsize=10.5)
    ax.text(x + 0.9, 1.3, b, ha='center', va='center', color=INK2, fontsize=8.2, linespacing=1.5)
for x in (1.92, 3.87, 5.82, 7.77):
    ax.annotate('', xy=(x + 0.12, 1.5), xytext=(x - 0.02, 1.5), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.2))
ax.annotate('', xy=(4.9, 0.3), xytext=(8.8, 0.3), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1, connectionstyle='arc3,rad=0.0'))
ax.text(6.85, 0.08, 'failed level: try the next level', ha='center', color=MUTED, fontsize=8)
save(fig, 'fig01_architecture')

# Fig 2 drift decomposition
fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
m = mon.copy()
for ax, col, ttl, yl in [(axs[0, 0], 'std_ratio', 'WARP spread relative to 2014', 'Std ratio'),
                         (axs[0, 1], 'tail_ratio', 'WARP exceedance frequency relative to 2014', 'P(|WARP| > 0.30) ratio'),
                         (axs[1, 0], 'excess_W', 'Mean WARP excess beyond the project limit', 'Excess (in.)')]:
    shade(ax); ax.plot(m.date, m[col], color=CAT[0], lw=2, marker='o', ms=4); ax.set_title(ttl, loc='left', fontsize=10, color=INK); ax.set_ylabel(yl)
    if col != 'excess_W': ax.axhline(1, color=MUTED, lw=1, ls='--')
ax = axs[1, 1]; shade(ax); bottom = np.zeros(len(m))
for c, lab, k in [('share_warp_only', 'WARP only', 0), ('share_xlevel_only', 'XLEVEL only', 1), ('share_joint', 'WARP and XLEVEL', 2), ('share_other', 'Other', 3)]:
    ax.bar(m.date, m[c], bottom=bottom, width=22, color=CAT[k], label=lab, edgecolor='white', linewidth=0.6); bottom += m[c].values
ax.set_title('Defect group shares', loc='left', fontsize=10, color=INK); ax.set_ylabel('Share of defects'); ax.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.3))
for a in axs.flat: a.tick_params(axis='x', rotation=30)
fig.text(0.5, -0.1, 'Grey band: 2015 development window.  Blue band: January to April 2016 test window.', ha='center', color=INK2, fontsize=8.5)
fig.tight_layout(); save(fig, 'fig02_drift_decomposition')

# Fig 3 action geometry on one 2016 target
df, model, lib = load()
s = lad[(lad.date.dt.year == 2016) & (lad.group == 'WARP only')]
piv = s.pivot_table(index='rid', columns='method', values='js', aggfunc='first')
cand = piv[(piv['stale'] == False) & (piv[best['Instance']] == True) & (piv['fresh'] == True)].index
rid = sorted(cand, key=lambda r: abs(df.set_index('rid').loc[r, 'WARP_62']))[len(cand) // 2]
x = df.set_index('rid').loc[rid, F].values.astype(float)
wg = np.linspace(-0.6, 0.6, 241); xg = np.linspace(-0.9, 1.0, 191); WW, XX = np.meshgrid(wg, xg)
P = np.tile(x, (WW.size, 1)); P[:, W] = WW.ravel(); P[:, X] = XX.ravel()
pr = model.predict_proba(P)[:, 1].reshape(WW.shape)
fig, ax = plt.subplots(figsize=(6.6, 5.2)); ax.grid(False)
ax.contourf(WW, XX, pr >= 0.5, levels=[-0.5, 0.5, 1.5], colors=['white', '#fbe9e1'])
ax.contour(WW, XX, pr, levels=[0.5], colors=[CAT[1]], linewidths=1.6)
ax.add_patch(Rectangle((-0.3, -0.8), 0.6, 1.6, fill=False, ec=INK2, ls='--', lw=1.1))
ax.scatter(x[W], x[X], s=70, color=INK, zorder=5); ax.annotate('target', (x[W], x[X]), xytext=(8, 6), textcoords='offset points', color=INK)
for k, mth in enumerate(KEY):
    r = lad[(lad.rid == rid) & (lad.method == mth)].iloc[0]; z = r[[f'z_{f}' for f in F]].values.astype(float)
    ax.annotate('', xy=(z[W], z[X] + 0.02 * (k - 2.5)), xytext=(x[W], x[X] + 0.02 * (k - 2.5)),
                arrowprops=dict(arrowstyle='-|>', color=COL[mth], lw=1.8))
    ax.scatter(z[W], z[X] + 0.02 * (k - 2.5), s=36, color=COL[mth], edgecolor='white', linewidth=1.5, zorder=6,
               label=f"{LBL[mth]}  ({'valid' if r.js else 'fails'})")
ax.set_xlabel('WARP (in.)'); ax.set_ylabel('XLEVEL (in.)'); ax.set_xlim(x[W] - 0.25, x[W] + 0.05 if x[W] > 0 else x[W] + 0.25); ax.set_ylim(x[X] - 0.2, x[X] + 0.2)
ax.text(0.01, -0.13, 'Two feature slice through the target. Shaded region: f0 predicts defect. Dashed line: project limit.',
        transform=ax.transAxes, fontsize=7.8, color=INK2, va='top')
if False: ax.text(-0.58, 0.93, 'shaded: f0 predicts defect (other features at target values)\ndashed box: project limits', fontsize=7.8, color=INK2, va='top')
ax.legend(fontsize=7.8, loc='center left', bbox_to_anchor=(1.01, 0.5)); save(fig, 'fig03_action_geometry')

# Fig 4 matched validity with block bootstrap intervals
def ci(sub, nb=1000):
    g = sub.groupby('block').js.agg(['sum', 'size']); r = np.random.default_rng(SEED); k = len(g); b = []
    for _ in range(nb):
        i = r.integers(0, k, k); b.append(g['sum'].values[i].sum() / g['size'].values[i].sum())
    return np.quantile(b, [.025, .975])
lad['block'] = lad.MILE.astype(str) + '_' + (lad.FEET // 200).astype(str)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
for ax, per in zip(axs, ['Dev 2015', 'Eval 2015', 'Test 2016']):
    for k, mth in enumerate(KEY):
        sub = lad[(lad.period == per) & (lad.method == mth)]; v = sub.js.mean(); lo, hi = ci(sub)
        ax.bar(k, v, color=COL[mth], width=0.7); ax.errorbar(k, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=3, lw=1)
        ax.text(k, hi + 0.02, f'{100 * v:.0f}', ha='center', fontsize=8.5, color=INK)
    ax.set_xticks(range(len(KEY))); ax.set_xticklabels([LBL[k] for k in KEY], rotation=40, ha='right', fontsize=8.5)
    n = len(lad[(lad.period == per) & (lad.method == 'stale')]); ax.set_title(f'{per}  (N = {n:,})', loc='left', fontsize=10, color=INK)
axs[0].set_ylabel('Joint success (flip, compliant, feasible)'); axs[0].set_ylim(0, 1.08)
fig.tight_layout(); save(fig, 'fig04_matched_validity')

# Fig 5 residual failure heatmap
H = T['residual_heat'].copy(); groups = ['WARP only', 'XLEVEL only', 'WARP and XLEVEL', 'Other']
mat = H.pivot(index='group', columns='dist_bin', values='fail').reindex(groups); nn = H.pivot(index='group', columns='dist_bin', values='n').reindex(groups)
mm = H.pivot(index='group', columns='dist_bin', values='mismatch').reindex(groups)
fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.grid(False)
im = ax.imshow(mat.values.astype(float), cmap='Blues', vmin=0, vmax=1, aspect='auto')
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i, j]; n = nn.values[i, j]
        if n and n > 0:
            ax.text(j, i, f'{100 * v:.0f}%\nn={int(n)}\nmis {100 * mm.values[i, j]:.0f}%', ha='center', va='center', fontsize=7.2,
                    color='white' if v > 0.6 else INK)
e = T['residual_dist_edges'].edge.values
ax.set_xticks(range(5)); ax.set_xticklabels([f'Q{i + 1}\n{e[i]:.2f} to {e[i + 1]:.2f}' for i in range(5)], fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels(groups); ax.set_xlabel('Distance to nearest 2014 reference origin (MAD units, quintiles)')
cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label('Residual failure after all ladder levels')
save(fig, 'fig05_residual_failures')

# Fig 6 strategies: validity and regeneration share
P = T['policy']; strat = ['Stale reuse', 'Universal fresh', 'Policy full ladder', 'Policy no support gate', 'Policy no gate without OT']
lab = ['Stale reuse', 'Universal fresh', 'Policy, registered gate', 'Policy, no gate', 'Policy, no gate, no OT']
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
for ax, col, ttl in [(axs[0], 'success', 'Joint success'), (axs[1], 'regen_frac', 'Share of cases regenerated')]:
    for k, per in enumerate(['Eval 2015', 'Test 2016']):
        v = [P[(P.period == per) & (P.strategy == s)][col].values[0] for s in strat]
        y = np.arange(len(strat)) + (0.2 if k == 0 else -0.2)
        ax.barh(y, v, height=0.38, color=[CAT[0], CAT[1]][k], label=per)
        for yy, vv in zip(y, v): ax.text(vv + 0.01, yy, f'{100 * vv:.0f}%', va='center', fontsize=8, color=INK)
    ax.set_yticks(range(len(strat))); ax.set_yticklabels(lab if ax is axs[0] else []); ax.invert_yaxis(); ax.set_xlim(0, 1.12)
    ax.set_title(ttl, loc='left', fontsize=10, color=INK)
fig.tight_layout(); fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.06)); save(fig, 'fig06_strategy_comparison')

# Fig 7 stress tests
meths = ['stale', best['Instance'], best['Scalar'], best['Diagonal'], 'ot_1d', 'ot_marg', 'ot_mv']
S = stress[stress.method.isin(meths)].pivot(index='scenario', columns='method', values='js')[meths]
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.grid(False)
im = ax.imshow(S.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        ax.text(j, i, f'{100 * S.values[i, j]:.0f}', ha='center', va='center', fontsize=8.5, color='white' if S.values[i, j] > 0.6 else INK)
ax.set_xticks(range(len(meths))); ax.set_xticklabels([LBL[k] for k in meths], rotation=25, ha='right')
ax.set_yticks(range(len(S))); ax.set_yticklabels(S.index); cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label('Joint success')
save(fig, 'fig07_stress_tests')

# Fig 8 monitoring, two panels on separate axes
fig, axs = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
ax = axs[0]; shade(ax)
ax.plot(m.date, m.validity, color=CAT[0], lw=2, marker='o', ms=4, label='Stale P2 transfer validity')
b13 = m[m.date.dt.year == 2013].validity; ax.axhspan(b13.min(), b13.max(), color=CAT[0], alpha=0.10)
ax.text(pd.Timestamp('2013-06-01'), b13.max() + 0.03, '2013 out of library band', fontsize=8, color=INK2)
ax.text(pd.Timestamp('2014-02-01'), 0.86, '2014 targets are the library origins', fontsize=8, color=INK2)
ax.set_ylabel('Feasible flip rate'); ax.set_ylim(0, 1.05); ax.set_title('Transferred explanation validity', loc='left', fontsize=10, color=INK)
ax = axs[1]; shade(ax); ax.plot(m.date, m.auc, color=CAT[1], lw=2, marker='o', ms=4)
ax.set_ylabel('ROC AUC'); ax.set_ylim(0.98, 1.001); ax.set_title('Fixed classifier ROC AUC', loc='left', fontsize=10, color=INK)
ax.tick_params(axis='x', rotation=30); fig.tight_layout(); save(fig, 'fig08_monitoring')

# Fig 9 refresh policy outcomes, registered gate and no gate
pols = A['pols']; fig, ax = plt.subplots(figsize=(8, 3.8))
outs = [best['Instance'], best['Scalar'], best['Diagonal'], best['OT'], 'fresh', 'unresolved']
cols = [CAT[0], CAT[1], CAT[2], CAT[6], INK, '#c9c8c2']; ylab = []
rowsp = [('Policy full ladder', 'Eval 2015', 'Eval 2015, registered gate'), ('Policy no support gate', 'Eval 2015', 'Eval 2015, no gate'),
         ('Policy full ladder', 'Test 2016', 'Test 2016, registered gate'), ('Policy no support gate', 'Test 2016', 'Test 2016, no gate')]
for i, (pk, per, yl) in enumerate(rowsp):
    p = pols[pk]; s = p[p.period == per].outcome.value_counts(normalize=True); left = 0; ylab.append(yl)
    for o, c in zip(outs, cols):
        v = s.get(o, 0); ax.barh(i, v, left=left, color=c, edgecolor='white', linewidth=1.2, label=(LBL.get(o, 'Unresolved') if i == 0 else None))
        if v > 0.05: ax.text(left + v / 2, i, f'{100 * v:.0f}%', ha='center', va='center', fontsize=8.5, color='white' if c in (INK, CAT[0], CAT[6]) else INK)
        left += v
ax.set_yticks(range(4)); ax.set_yticklabels(ylab); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel('Share of target cases resolved by each step')
ax.legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.22)); save(fig, 'fig09_policy_outcomes')
print('figures written', sorted(os.listdir('figs')))
