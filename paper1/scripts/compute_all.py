"""Computes the CFs and persistence labels needed for figs #2–#6.

Caches each phase to scripts/.cache/ so reruns are cheap.

Phases:
  A) CFs for 2014 cohort (frozen base model)        — for #3, #4
  B) CFs for 2015 cohort (frozen base model)        — for #2, #6
  C) CFs for 2016 cohort (frozen base model)        — for #2, #6
  D) Persistence labels per (MILE, FEET, date)      — for #2
  E) Recalibrated model trained on 2014+2015        — for #5
  F) CFs for 2016 under recalibrated model          — for #5

Cohorts are capped to keep wall time under ~30 min.
"""
import os, sys, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from _cf_lib import (load_data, train_model, generate_cfs_for_cohort,
                     FEATURES, ROOT, CACHE, PROJECT_LIMITS)

COHORT_CAP = 800  # max defective samples per cohort year

print('='*60); print('PHASE A — data + base model'); print('='*60)
df    = load_data()
model = train_model(df, train_years=(2013, 2014), tag='base')
print(f'  rows={len(df):,}  defects={df["defect"].sum():,}')

print('\n'+'='*60); print('PHASE A — CFs for 2014 cohort'); print('='*60)
cf_2014 = generate_cfs_for_cohort(model, df, year=2014, tag='2014_base',
                                  max_samples=COHORT_CAP)

print('\n'+'='*60); print('PHASE B — CFs for 2015 cohort'); print('='*60)
cf_2015 = generate_cfs_for_cohort(model, df, year=2015, tag='2015_base',
                                  max_samples=COHORT_CAP)

print('\n'+'='*60); print('PHASE C — CFs for 2016 cohort'); print('='*60)
cf_2016 = generate_cfs_for_cohort(model, df, year=2016, tag='2016_base',
                                  max_samples=COHORT_CAP)

print('\n'+'='*60); print('PHASE D — persistence labels'); print('='*60)
pkl_p = os.path.join(CACHE, 'persistence.pkl')
if os.path.exists(pkl_p):
    print('  cached.')
else:
    out = df.sort_values(['MILE', 'FEET', 'date']).copy()
    def per_seg(g):
        g = g.sort_values('date').copy()
        d = g['defect'].astype(int).values
        n = len(g)
        for h in (1, 3, 6):
            any_list, cnt_list = [], []
            for i in range(n):
                fut = d[i+1:i+1+h]
                if len(fut) == 0:
                    any_list.append(np.nan); cnt_list.append(np.nan)
                else:
                    any_list.append(int(fut.any()))
                    cnt_list.append(int(fut.sum()))
            g[f'persist_{h}m']       = any_list
            g[f'persist_count_{h}m'] = cnt_list
        return g
    out = out.groupby(['MILE', 'FEET'], group_keys=False).apply(per_seg)
    pickle.dump(out.reset_index(drop=True), open(pkl_p, 'wb'))
    print(f'  built {len(out):,} rows.')

print('\n'+'='*60); print('PHASE E — recalibrated model (train 2014+2015)'); print('='*60)
model_recal = train_model(df, train_years=(2014, 2015), tag='recal')

print('\n'+'='*60); print('PHASE F — CFs for 2016 under recalibrated model'); print('='*60)
cf_2016_recal = generate_cfs_for_cohort(model_recal, df, year=2016, tag='2016_recal',
                                        max_samples=COHORT_CAP)

print('\nALL PHASES COMPLETE.')
print('Cached files:')
for f in sorted(os.listdir(CACHE)):
    sz = os.path.getsize(os.path.join(CACHE, f)) / 1024
    print(f'  {sz:7.1f} KB  {f}')
