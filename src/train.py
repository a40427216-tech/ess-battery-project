"""Select on Batch 1 development only, then evaluate a frozen model.

Run from the repository root: python -m src.train
The outer five group folds are frozen from DAY 1. Every hyperparameter search
uses four inner policy-group folds with imputation/scaling fitted inside them.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .features import FEATURE_SETS, generate, validate_feature_columns
from .preprocess import ROOT, load_features
from .transforms import power10

SEED = 42
SCORER = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
TARGET_MAPE = 9.1


def pipeline(model, log_target=True):
    steps = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()), ('model', model),
    ])
    return TransformedTargetRegressor(regressor=steps, func=np.log10,
                                      inverse_func=power10) if log_target else steps


def candidate_specs():
    candidates = [
        {'name': 'Dummy median', 'family': 'Dummy', 'feature_set': 'variance',
         'estimator': pipeline(DummyRegressor(strategy='median'), log_target=False), 'grid': {}},
        {'name': 'Variance linear', 'family': 'Linear', 'feature_set': 'variance',
         'estimator': pipeline(LinearRegression()), 'grid': {}},
    ]
    for feature_set in FEATURE_SETS:
        candidates.append({'name': f'Ridge / {feature_set}', 'family': 'Ridge',
            'feature_set': feature_set, 'estimator': pipeline(Ridge()),
            'grid': {'regressor__model__alpha': [.01, .1, 1., 10., 100.]}})
    for feature_set in ['discharge', 'state', 'policy', 'current']:
        candidates.append({'name': f'ElasticNet / {feature_set}', 'family': 'ElasticNet',
            'feature_set': feature_set,
            'estimator': pipeline(ElasticNet(max_iter=100_000, tol=1e-7, random_state=SEED)),
            'grid': {'regressor__model__alpha': [.001, .01, .1],
                     'regressor__model__l1_ratio': [.1, .5, .9]}})
    for feature_set in ['variance', 'discharge', 'state']:
        candidates.append({'name': f'SVR / {feature_set}', 'family': 'SVR',
            'feature_set': feature_set, 'estimator': pipeline(SVR(kernel='rbf')),
            'grid': {'regressor__model__C': [.1, 1., 10.],
                     'regressor__model__gamma': ['scale', .1],
                     'regressor__model__epsilon': [.02, .05]}})
    for feature_set in ['discharge', 'state', 'policy']:
        candidates.append({'name': f'RandomForest / {feature_set}', 'family': 'RandomForest',
            'feature_set': feature_set,
            'estimator': pipeline(RandomForestRegressor(n_estimators=120, random_state=SEED, n_jobs=1)),
            'grid': {'regressor__model__max_depth': [2, 3],
                     'regressor__model__min_samples_leaf': [2, 4]}})
    return candidates


def metrics(actual, prediction):
    y, p = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(p).all() or np.any(y <= 0):
        raise ValueError('Metrics require finite predictions and positive targets')
    error = p - y
    return {'n': len(y), 'MAPE (%)': float(100 * np.mean(np.abs(error) / y)),
            'MAE (cycles)': float(np.mean(np.abs(error))),
            'RMSE (cycles)': float(np.sqrt(np.mean(error ** 2))),
            'R2': float(r2_score(y, p)) if len(y) > 1 else None,
            'Bias (cycles)': float(np.mean(error)),
            'overprediction_count': int(np.sum(error > 0)),
            'predicted_life_le_100_count': int(np.sum(p <= 100))}


def tune(spec, frame):
    columns = FEATURE_SETS[spec['feature_set']]
    validate_feature_columns(columns)
    if not spec['grid']:
        return clone(spec['estimator']).fit(frame[columns], frame.cycle_life), {}
    groups = frame.charging_policy
    cv = GroupKFold(n_splits=min(4, groups.nunique()))
    search = GridSearchCV(clone(spec['estimator']), spec['grid'], scoring=SCORER,
                          cv=cv, n_jobs=1, error_score='raise', refit=True)
    search.fit(frame[columns], frame.cycle_life, groups=groups)
    return search.best_estimator_, search.best_params_


def prediction_rows(frame, prediction, model_name, subset, fold=None):
    keep = ['cell_id', 'batch', 'charging_policy', 'cycle_life']
    out = frame[keep].copy().reset_index(drop=True).rename(columns={'cycle_life': 'actual'})
    out['model'] = model_name
    out['subset'] = subset
    out['fold'] = fold
    out['predicted'] = np.asarray(prediction)
    out['error_cycles'] = out.predicted - out.actual
    out['absolute_percentage_error'] = 100 * np.abs(out.error_cycles) / out.actual
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def performance_rows(cv_mean, cv_std, hold, test2, test3):
    return [
        {'구분': 'Train (Batch 1 CV)', 'MAPE (%) 또는 Gap (%p)': cv_mean,
         '비고': f'정책 그룹 5-fold, 평균 ± 표준편차 {cv_mean:.3f} ± {cv_std:.3f}%'},
        {'구분': 'Valid (Batch 1 Hold-out)', 'MAPE (%) 또는 Gap (%p)': hold,
         '비고': '개발 데이터와 정책이 겹치지 않는 8개 셀'},
        {'구분': 'Test (Batch 2)', 'MAPE (%) 또는 Gap (%p)': test2,
         '비고': '과제 지정 2018-02-20, 39개 셀'},
        {'구분': 'Gap (Train-Valid)', 'MAPE (%) 또는 Gap (%p)': hold - cv_mean,
         '비고': 'Valid − CV; 양수이면 내부 검증 오차 증가'},
        {'구분': 'Gap (Valid-Test)', 'MAPE (%) 또는 Gap (%p)': test2 - hold,
         '비고': 'Batch 2 − Valid; 양수이면 외부 평가 오차 증가'},
        {'구분': 'Gap (Target-Test)', 'MAPE (%) 또는 Gap (%p)': test2 - TARGET_MAPE,
         '비고': 'Batch 2 − 9.1; 서로 다른 코호트·분리 조건의 참고 비교'},
        {'구분': 'Test (Batch 3)', 'MAPE (%) 또는 Gap (%p)': test3,
         '비고': '추가 평가, 2018-04-12, 40개 셀'},
        {'구분': 'Gap (Batch2-Batch3)', 'MAPE (%) 또는 Gap (%p)': test3 - test2,
         '비고': 'Batch 3 − Batch 2; 양수이면 Batch 3 오차가 더 큼'},
        {'구분': 'Gap (Target-Test, Batch 3)', 'MAPE (%) 또는 Gap (%p)': test3 - TARGET_MAPE,
         '비고': 'Batch 3 − 9.1; 논문 목표와의 참고 비교'},
    ]


def run(root=ROOT):
    root = Path(root)
    generate(root)
    table, split = load_features(root)
    by_id = table.set_index('cell_id', drop=False)
    dev = by_id.loc[split['development_cells']]
    specs = candidate_specs()
    folds, comparisons, all_oof = [], [], []
    start = time.monotonic()
    for spec in specs:
        columns = FEATURE_SETS[spec['feature_set']]
        candidate_folds, candidate_predictions = [], []
        for i, fold in enumerate(split['cv'], 1):
            train = by_id.loc[fold['train_cells']]
            valid = by_id.loc[fold['valid_cells']]
            fitted, params = tune(spec, train)
            predicted = fitted.predict(valid[columns])
            result = {'model': spec['name'], 'family': spec['family'],
                      'feature_set': spec['feature_set'], 'fold': i,
                      'best_params': json.dumps(params, sort_keys=True),
                      **metrics(valid.cycle_life, predicted)}
            candidate_folds.append(result)
            candidate_predictions.append(prediction_rows(valid, predicted, spec['name'], 'CV OOF', i))
        prediction = pd.concat(candidate_predictions, ignore_index=True)
        mean = float(np.mean([x['MAPE (%)'] for x in candidate_folds]))
        std = float(np.std([x['MAPE (%)'] for x in candidate_folds], ddof=1))
        comparison = {'model': spec['name'], 'family': spec['family'],
                      'feature_set': spec['feature_set'], 'n_features': len(columns),
                      'CV MAPE mean (%)': mean, 'CV MAPE std (%)': std,
                      'OOF pooled MAPE (%)': metrics(prediction.actual, prediction.predicted)['MAPE (%)'],
                      'OOF MAE (cycles)': metrics(prediction.actual, prediction.predicted)['MAE (cycles)']}
        comparisons.append(comparison)
        folds.extend(candidate_folds)
        all_oof.append(prediction)
        print(f"{spec['name']}: CV MAPE {mean:.3f} ± {std:.3f}% ({time.monotonic()-start:.1f}s)", flush=True)
    comparisons = pd.DataFrame(comparisons).sort_values(
        ['CV MAPE mean (%)', 'n_features', 'model']).reset_index(drop=True)
    selected_name = comparisons.iloc[0]['model']
    selected_spec = next(x for x in specs if x['name'] == selected_name)
    final_model, final_params = tune(selected_spec, dev)
    chosen_features = FEATURE_SETS[selected_spec['feature_set']]
    selection = {
        'selected_model': selected_name, 'feature_set': selected_spec['feature_set'],
        'features': chosen_features, 'best_params': final_params,
        'selection_criterion': 'Minimum mean outer group-CV MAPE on Batch 1 development; ties by fewer features/name',
        'selection_uses_only_cells': dev.cell_id.tolist(),
        'fit_cells': dev.cell_id.tolist(), 'n_fit': len(dev),
        'target': 'total cycle_life; not remaining life',
        'target_transform': 'identity' if selected_spec['family'] == 'Dummy' else 'log10 / inverse 10**x',
        'random_state': SEED,
        'feature_table_sha256': sha256(root / 'results/features.csv'),
        'split_sha256': sha256(root / 'results/split_design.json'),
        'candidate_grids': [{'name': x['name'], 'feature_set': x['feature_set'], 'grid': x['grid']} for x in specs],
        'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'frozen_at_utc_before_external_evaluation': datetime.now(timezone.utc).isoformat(),
        'cv_limitation': 'Outer CV ranks candidate families/feature sets, so selected CV performance can be optimistic. Hold-out is kept separate.',
        'external_eda_limitation': 'All batches were inspected in DAY 1 EDA; this is not a fully blinded study.',
    }
    results = root / 'results'
    comparisons.to_csv(results / 'cv_comparison.csv', index=False)
    pd.DataFrame(folds).to_csv(results / 'cv_fold_metrics.csv', index=False)
    pd.concat(all_oof, ignore_index=True).to_csv(results / 'cv_oof_predictions.csv', index=False)
    (results / 'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2))
    (root / 'models').mkdir(exist_ok=True)
    joblib.dump({'model': final_model, 'features': chosen_features, 'selection': selection},
                root / 'models/final_model.joblib')
    # The selected model and hyperparameters are frozen before external targets are used for evaluation.
    eval_sets = [('Hold-out', split['holdout_cells']), ('Batch 2', split['test_batch2_cells']),
                 ('Batch 3', split['additional_batch3_cells'])]
    eval_models = [(selected_spec, final_model)]
    for reference in specs[:2]:
        if reference['name'] != selected_name:
            fitted, _ = tune(reference, dev)
            eval_models.append((reference, fitted))
    evaluated, predictions = [], []
    for spec, model in eval_models:
        for subset, ids in eval_sets:
            frame = by_id.loc[ids]
            p = model.predict(frame[FEATURE_SETS[spec['feature_set']]])
            evaluated.append({'model': spec['name'], 'subset': subset,
                              **metrics(frame.cycle_life, p)})
            predictions.append(prediction_rows(frame, p, spec['name'], subset))
    metrics_table = pd.DataFrame(evaluated)
    metrics_table.to_csv(results / 'evaluation_metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(results / 'predictions.csv', index=False)
    selected_metrics = metrics_table.loc[metrics_table.model == selected_name].set_index('subset')
    selected_cv = comparisons.loc[comparisons.model == selected_name].iloc[0]
    rows = performance_rows(float(selected_cv['CV MAPE mean (%)']), float(selected_cv['CV MAPE std (%)']),
                            float(selected_metrics.loc['Hold-out', 'MAPE (%)']),
                            float(selected_metrics.loc['Batch 2', 'MAPE (%)']),
                            float(selected_metrics.loc['Batch 3', 'MAPE (%)']))
    pd.DataFrame(rows).to_csv(results / 'model_performance.csv', index=False)
    print('\nSelected:', selected_name, final_params, flush=True)
    print(metrics_table.round(4).to_string(index=False), flush=True)
    return selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)
