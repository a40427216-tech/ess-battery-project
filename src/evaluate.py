"""Diagnostics for the frozen model; this module never refits or selects models."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .features import ALLOWED_FEATURES
from .preprocess import ROOT, load_features
from .train import metrics, sha256

COLORS = {'CV OOF': '#2463A6', 'Hold-out': '#718096', 'Batch 2': '#D56A21', 'Batch 3': '#288C79'}


def configure(root):
    font = root / 'references/NanumGothic-Regular.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = [font_manager.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans']
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, root, name):
    fig.tight_layout()
    fig.savefig(root / f'results/figures/{name}.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def group_bootstrap(frame, repeats=2000):
    """Approximate sampling interval conditional on this fixed model, by policy clusters."""
    groups = [g.absolute_percentage_error.to_numpy() for _, g in frame.groupby('charging_policy')]
    rng = np.random.default_rng(42)
    values = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
              for _ in range(repeats)]
    return np.percentile(values, [2.5, 97.5])


def analyze(root=ROOT):
    root = Path(root)
    configure(root)
    features, split = load_features(root)
    saved = joblib.load(root / 'models/final_model.joblib')
    selected = saved['selection']['selected_model']
    selection = json.loads((root / 'results/selection.json').read_text())
    if sha256(root / 'results/features.csv') != selection['feature_table_sha256']:
        raise ValueError('Feature table changed after model selection')
    if sha256(root / 'results/split_design.json') != selection['split_sha256']:
        raise ValueError('Frozen split changed after model selection')
    predictions = pd.read_csv(root / 'results/predictions.csv')
    predictions = predictions.loc[predictions.model == selected].copy()
    oof = pd.read_csv(root / 'results/cv_oof_predictions.csv')
    oof = oof.loc[oof.model == selected].copy()
    all_pred = pd.concat([oof, predictions], ignore_index=True)
    by_id = features.set_index('cell_id')
    for subset, frame in predictions.groupby('subset'):
        again = saved['model'].predict(by_id.loc[frame.cell_id, saved['features']])
        np.testing.assert_allclose(again, frame.predicted, rtol=1e-10, atol=1e-8)
    dev = by_id.loc[split['development_cells']]
    selected_policies = set(dev.charging_policy)
    all_pred['seen_training_policy'] = all_pred.charging_policy.isin(selected_policies)
    all_pred['outside_training_target_range'] = ~all_pred.actual.between(dev.cycle_life.min(), dev.cycle_life.max())
    # OOF predictions come from each fold's training cells, not the final 28-cell fit.
    for i, fold in enumerate(split['cv'], 1):
        mask = (all_pred.subset == 'CV OOF') & (all_pred.fold == i)
        training = by_id.loc[fold['train_cells']]
        all_pred.loc[mask, 'seen_training_policy'] = all_pred.loc[mask, 'charging_policy'].isin(set(training.charging_policy))
        all_pred.loc[mask, 'outside_training_target_range'] = ~all_pred.loc[mask, 'actual'].between(training.cycle_life.min(), training.cycle_life.max())
    all_pred['life_range'] = np.select([all_pred.actual < 500, all_pred.actual > 1000],
                                      ['<500', '>1000'], default='500–1000')
    all_pred.to_csv(root / 'results/selected_predictions.csv', index=False)
    error_tables = []
    for subset, frame in all_pred.groupby('subset', sort=False):
        for dimension in ['life_range', 'seen_training_policy', 'outside_training_target_range']:
            for value, part in frame.groupby(dimension):
                error_tables.append({'subset': subset, 'dimension': dimension, 'group': value,
                                     **metrics(part.actual, part.predicted)})
    pd.DataFrame(error_tables).to_csv(root / 'results/error_groups.csv', index=False)
    policy_tables = [{'subset': subset, 'charging_policy': policy,
                      **metrics(part.actual, part.predicted)}
                     for (subset, policy), part in predictions.groupby(['subset', 'charging_policy'])]
    pd.DataFrame(policy_tables).to_csv(root / 'results/error_by_policy.csv', index=False)
    predictions.nlargest(12, 'absolute_percentage_error').to_csv(root / 'results/largest_errors.csv', index=False)
    drift = []
    for column in sorted(ALLOWED_FEATURES):
        values = dev[column].dropna()
        mean, std = values.mean(), values.std(ddof=1)
        for batch, part in features.groupby('batch'):
            p = part[column]
            drift.append({'batch': int(batch), 'feature': column, 'n_valid': int(p.notna().sum()),
                          'missing_count': int(p.isna().sum()), 'dev_mean': mean, 'dev_std': std,
                          'batch_mean': p.mean(), 'mean_shift_in_dev_sd': (p.mean()-mean)/std if std > 0 else np.nan,
                          'outside_dev_range_pct': 100*((p < values.min()) | (p > values.max())).sum()/p.notna().sum(),
                          'selected_feature': column in saved['features']})
    pd.DataFrame(drift).to_csv(root / 'results/feature_drift.csv', index=False)
    intervals = []
    for subset, part in predictions.groupby('subset'):
        low, high = group_bootstrap(part)
        intervals.append({'subset': subset, 'n_cells': len(part), 'n_policy_groups': part.charging_policy.nunique(),
                          'MAPE (%)': part.absolute_percentage_error.mean(),
                          'bootstrap_2.5%': low, 'bootstrap_97.5%': high,
                          'method': '2000 policy-cluster bootstrap resamples; fixed model; approximate sampling interval'})
    pd.DataFrame(intervals).to_csv(root / 'results/mape_bootstrap.csv', index=False)
    fitted = saved['model'].regressor_ if hasattr(saved['model'], 'regressor_') else saved['model']
    if hasattr(fitted['model'], 'coef_'):
        coef = np.ravel(fitted['model'].coef_)
        pd.DataFrame({'feature': saved['features'], 'standardized_log10_coefficient': coef,
                      'coefficient_per_original_unit': coef / fitted['scaler'].scale_}).to_csv(
                          root / 'results/model_coefficients.csv', index=False)
        z = fitted['scaler'].transform(fitted['imputer'].transform(by_id.loc[predictions.cell_id, saved['features']]))
        contributions = pd.DataFrame(z * coef, columns=saved['features'])
        contributions.insert(0, 'cell_id', predictions.cell_id.to_numpy())
        contributions.insert(1, 'subset', predictions.subset.to_numpy())
        contributions.to_csv(root / 'results/prediction_log_contributions.csv', index=False)
    comparison = pd.read_csv(root / 'results/cv_comparison.csv').sort_values('CV MAPE mean (%)', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    color = ['#2463A6' if x == selected else '#A9BACD' for x in comparison.model]
    ax.barh(comparison.model, comparison['CV MAPE mean (%)'], xerr=comparison['CV MAPE std (%)'], color=color, capsize=3)
    ax.set(xlabel='그룹 CV MAPE (%) · 오차막대: 5-fold 표준편차', title='Batch 1 개발 데이터에서만 후보 비교')
    save(fig, root, '10_cv_candidates')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, subset in zip(axes, ['Hold-out', 'Batch 2', 'Batch 3']):
        part = predictions.loc[predictions.subset == subset]
        ax.scatter(part.actual, part.predicted, c=COLORS[subset], alpha=.8, s=30)
        lo, hi = min(part.actual.min(), part.predicted.min())*.9, max(part.actual.max(), part.predicted.max())*1.05
        ax.plot([lo, hi], [lo, hi], color='#666', ls='--')
        ax.set(xlabel='실제 총수명 (사이클)', ylabel='예측 총수명 (사이클)',
               title=f"{subset}: MAPE {part.absolute_percentage_error.mean():.2f}%", xlim=(lo,hi), ylim=(lo,hi))
    save(fig, root, '11_model_parity')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for subset, part in predictions.groupby('subset'):
        axes[0].scatter(part.actual, part.error_cycles, color=COLORS[subset], label=subset, s=25, alpha=.75)
        axes[1].scatter(part.actual, part.absolute_percentage_error, color=COLORS[subset], label=subset, s=25, alpha=.75)
    axes[0].axhline(0, c='#666', ls='--');axes[0].legend(fontsize=8)
    axes[0].set(xlabel='실제 총수명 (사이클)', ylabel='예측 − 실제 (사이클)', title='배치별 예측 편향')
    axes[1].set(xlabel='실제 총수명 (사이클)', ylabel='개별 절대 상대오차 (%)', title='평균 MAPE에 가려지는 셀별 오차')
    save(fig, root, '12_model_errors')
    fig, axes = plt.subplots(1, len(saved['features']), figsize=(12, 3.5))
    for ax, column in zip(np.atleast_1d(axes), saved['features']):
        parts = [dev[column].dropna(), features.loc[features.batch==2,column].dropna(), features.loc[features.batch==3,column].dropna()]
        box = ax.boxplot(parts, tick_labels=['개발', 'B2', 'B3'], patch_artist=True)
        for patch, color in zip(box['boxes'], ['#2463A6','#D56A21','#288C79']):patch.set_facecolor(color);patch.set_alpha(.6)
        ax.set(title=column)
    fig.suptitle('최종 피처의 입력 분포 차이')
    save(fig, root, '13_feature_drift')
    evaluation = pd.read_csv(root / 'results/evaluation_metrics.csv')
    fig, ax = plt.subplots(figsize=(9, 4))
    model_order = [selected, 'Variance linear', 'Dummy median']
    for i, name in enumerate(model_order):
        part = evaluation.loc[evaluation.model == name].set_index('subset')
        values = [part.loc[s, 'MAPE (%)'] for s in ['Hold-out','Batch 2','Batch 3']]
        ax.bar(np.arange(3)+(i-1)*.24, values, width=.24, label=name)
    ax.set_xticks(np.arange(3), ['Hold-out','Batch 2','Batch 3']);ax.set_ylabel('MAPE (%)')
    ax.set_title('고정한 최종 모델과 사전 지정 기준 모델의 외부 평가');ax.legend(fontsize=8)
    save(fig, root, '14_baseline_comparison')
    if (root / 'results/model_coefficients.csv').exists():
        coefficients = pd.read_csv(root / 'results/model_coefficients.csv')
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.barh(coefficients.feature, coefficients.standardized_log10_coefficient, color='#2463A6')
        ax.axvline(0, c='#666', lw=.8)
        ax.set(xlabel='표준화 피처 1단위당 log10 총수명 계수', title='최종 모델의 계수 · 인과 효과를 의미하지 않음')
        save(fig, root, '15_model_coefficients')
    print('Saved error, drift, coefficient and uncertainty diagnostics; model unchanged.')


if __name__ == '__main__':
    analyze()
