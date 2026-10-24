"""Defect composition by year (paper figure defect_composition_by_year.pdf).

Generated 2026-06-05 from the canonical cache (data.pkl). Categories:
WARP-driven (any WARP violation, incl. combined), XLEVEL-only
(XLEVEL violation without WARP), alignment/gage only (neither).
Reproduces locked text values: 2014 WARP-driven 78.7%, XLEVEL-only
21.3% -> 37.2% (2016); XLEVEL-any reaches 44.4% (2016).
"""
import os, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
df = pickle.load(open(os.path.join(HERE, '.cache', 'data.pkl'), 'rb'))
PL = {'GAGE': (56.3, 57.2), 'ALI_L_62': (-0.20, 0.20), 'ALI_R_62': (-0.20, 0.20),
      'WARP_62': (-0.30, 0.30), 'XLEVEL': (-0.80, 0.80)}
viol = {f: ((df[f] < lo) | (df[f] > hi)) for f, (lo, hi) in PL.items()}
df['defect_any'] = np.logical_or.reduce(list(viol.values()))

rows = []
for y in (2013, 2014, 2015, 2016):
    d = df[(df['year'] == y) & df['defect_any']]
    w = viol['WARP_62'][d.index]; x = viol['XLEVEL'][d.index]
    rows.append(dict(year=y, n=len(d), warp_driven=100*w.mean(),
                     xlevel_only=100*(x & ~w).mean(), other_only=100*(~x & ~w).mean()))
t = pd.DataFrame(rows)
print(t.round(1).to_string(index=False))

years = t['year'].astype(str)
fig, ax = plt.subplots(figsize=(6.8, 4.4))
b = np.zeros(4)
for col, lab, c in [('warp_driven', 'WARP-driven (incl. combined)', '#cc2a2a'),
                    ('xlevel_only', 'XLEVEL-only', '#1f6fb4'),
                    ('other_only', 'Alignment or gage only', '#999999')]:
    ax.bar(years, t[col], bottom=b, label=lab, color=c, alpha=0.88)
    for i, (v, bb) in enumerate(zip(t[col], b)):
        if v > 4:
            ax.text(i, bb + v/2, f'{v:.1f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
    b += t[col].values
ax.set_ylabel('Share of defective segments (%)'); ax.set_ylim(0, 100)
ax.set_title('Defect composition by year\nXLEVEL-only defects grow as the deterioration regime shifts',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5, loc='lower left', framealpha=0.9)
for i, n in enumerate(t['n']):
    ax.text(i, -9, f'n = {n}', ha='center', fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(HERE, '..', '..', 'defect_composition_by_year.pdf'))
print('saved')
