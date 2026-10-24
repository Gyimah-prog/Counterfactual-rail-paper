"""Drift figure: exceedance curves of |WARP_62| by year on a log scale.
Usage: python fig_warp_exceedance.py <path to .cache/data.pkl> <output stem>"""
import pickle, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
data = sys.argv[1] if len(sys.argv) > 1 else '.cache/data.pkl'
out = sys.argv[2] if len(sys.argv) > 2 else 'warp_exceedance'
df = pickle.load(open(data, 'rb'))
cols = ['#9ecae1', '#6baed6', '#2171b5', '#08306b']   # one hue, light to dark over time
plt.rcParams.update({'font.family': 'serif', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(3.4, 2.6))
xs = np.linspace(0, 0.7, 400); LIM = 0.30
for (y, g), c in zip(df.groupby('year'), cols):
    v = np.sort(np.abs(g['WARP_62'].values))
    ex = 1 - np.searchsorted(v, xs, side='right') / len(v)
    ax.plot(xs, np.where(ex > 0, ex * 100, np.nan), color=c, lw=1.8,
            label=str(y))
    print(y, 'share beyond 0.30 in. = %.2f%%' % ((np.abs(g['WARP_62']) > LIM).mean() * 100))
ax.axvline(LIM, color='0.35', ls='--', lw=1)
ax.text(LIM, 60, ' project limit', fontsize=8, color='0.3', va='top')
ax.set_yscale('log'); ax.set_ylim(0.005, 100); ax.set_xlim(0, 0.7)
ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
ax.yaxis.set_major_formatter(FixedFormatter(['0.01', '0.1', '1', '10', '100']))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('|WARP| (in.)'); ax.set_ylabel('Measurements exceeding value (%)')
ax.grid(axis='y', color='0.9', lw=0.6)
ax.legend(frameon=False, fontsize=7.5, loc='lower left')
fig.tight_layout(); fig.savefig(out + '.png', dpi=300); fig.savefig(out + '.pdf')
