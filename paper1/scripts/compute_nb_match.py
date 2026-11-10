"""Reproduce the notebook's exact XGBoost hyperparameters and regenerate CFs.

Notebook cell 29 used:
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=SPW (= n_neg / n_pos on 2013+2014 training data)

This script trains BOTH the base and recal models with those hyperparameters,
generates 2014/2015 CFs under each, and saves into the existing cache with
the 'nb_match' suffix so the original caches stay untouched.
"""
import os, sys, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from _cf_lib import (load_data, generate_cfs_for_cohort, FEATURES, CACHE, SEED)


def train_notebook_match(df, train_years, tag):
    pkl = os.path.join(CACHE, f'model_{tag}.pkl')
    if os.path.exists(pkl):
        print(f'  reusing cached {pkl}')
        return pickle.load(open(pkl, 'rb'))

    train = df[df['year'].isin(train_years)]
    X = train[FEATURES].values
    y = train['defect'].values
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    spw = n_neg / max(1, n_pos)
    print(f'  training on years {train_years}: n_pos={n_pos}, n_neg={n_neg}, '
          f'scale_pos_weight={spw:.2f}')

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=SEED,
        eval_metric='logloss',
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X, y)
    pickle.dump(model, open(pkl, 'wb'))
    return model


df = load_data()

print('=' * 60)
print('PHASE 1 — base_nb (notebook-hyperparam model, train 2013–14)')
print('=' * 60)
base_nb = train_notebook_match(df, (2013, 2014), 'base_nb')

print('\n  CFs on 2014 (under base_nb)')
cf_2014_base_nb = generate_cfs_for_cohort(base_nb, df, year=2014,
                                          tag='2014_base_nb', max_samples=800)
print('\n  CFs on 2015 (under base_nb)')
cf_2015_base_nb = generate_cfs_for_cohort(base_nb, df, year=2015,
                                          tag='2015_base_nb', max_samples=800)
print('\n  CFs on 2016 (under base_nb)')
cf_2016_base_nb = generate_cfs_for_cohort(base_nb, df, year=2016,
                                          tag='2016_base_nb', max_samples=800)

print('\n' + '=' * 60)
print('PHASE 2 — recal_nb (notebook-hyperparam model, train 2014–15)')
print('=' * 60)
recal_nb = train_notebook_match(df, (2014, 2015), 'recal_nb')

print('\n  CFs on 2014 (under recal_nb)')
cf_2014_recal_nb = generate_cfs_for_cohort(recal_nb, df, year=2014,
                                            tag='2014_recal_nb', max_samples=800)
print('\n  CFs on 2015 (under recal_nb)')
cf_2015_recal_nb = generate_cfs_for_cohort(recal_nb, df, year=2015,
                                            tag='2015_recal_nb', max_samples=800)
print('\n  CFs on 2016 (under recal_nb)')
cf_2016_recal_nb = generate_cfs_for_cohort(recal_nb, df, year=2016,
                                            tag='2016_recal_nb', max_samples=800)

print('\nAll done. Caches:')
for f in sorted(os.listdir(CACHE)):
    if '_nb' in f:
        sz = os.path.getsize(os.path.join(CACHE, f)) / 1024
        print(f'  {sz:7.1f} KB  {f}')
