"""Validate saved predictions, target metrics, group splits and notebook execution."""
from pathlib import Path
import json
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import nbformat
from src.features import FEATURE_SETS, validate_feature_columns
from src.preprocess import load_features, validate_input_manifest
from src.train import metrics, sha256


def validate():
    checks = []
    def check(condition, text):
        if not condition:
            raise AssertionError(text)
        checks.append(text)
    table, split = load_features(ROOT)
    check(validate_input_manifest(ROOT), 'All six processed input fingerprints match their provenance manifest')
    check(table.groupby('batch').size().to_dict() == {1:36,2:39,3:40}, 'Correct assignment batches and frozen cohort')
    original = pd.read_csv(ROOT / 'data/processed/cells_analysis.csv')
    check(set(original.cell_id) == set(table.cell_id), 'Features cover exactly the selected cohort')
    truth = original.set_index('cell_id').cycle_life
    check(np.allclose(truth.loc[table.cell_id], table.cycle_life), 'Target labels were not replaced or corrected from another batch')
    selection = json.loads((ROOT / 'results/selection.json').read_text())
    dev = set(split['development_cells']); hold = set(split['holdout_cells'])
    check(set(selection['selection_uses_only_cells']) == dev, 'Candidate selection uses only the 28 development cells')
    check(set(selection['fit_cells']) == dev and selection['n_fit'] == 28, 'Final fitted model excludes the hold-out and external batches')
    check(selection['feature_table_sha256'] == sha256(ROOT / 'results/features.csv'), 'Frozen feature-table fingerprint matches')
    check(selection['split_sha256'] == sha256(ROOT / 'results/split_design.json'), 'DAY 1 split fingerprint matches')
    check(validate_feature_columns(selection['features']), 'Selected model has only whitelisted early-cycle features')
    if (ROOT/'results/inner_cv_audit.json').exists():
        audit=json.loads((ROOT/'results/inner_cv_audit.json').read_text())
        check(audit['candidate_grids_match_frozen_selection'], 'Current 17 candidate grids match the frozen selection record')
        check(len(audit['inner_folds'])==24, 'All 24 inner tuning folds are accounted for')
        by_id=table.set_index('cell_id')
        for record in audit['inner_folds']:
            train=set(record['train_cells']); valid=set(record['valid_cells'])
            label=f"{record['context']}/inner fold {record['fold']}"
            check(train.isdisjoint(valid) and (train|valid)<=dev and (train|valid).isdisjoint(hold), f'{label}: only independent development cells')
            check(set(by_id.loc[list(train)].charging_policy).isdisjoint(by_id.loc[list(valid)].charging_policy), f'{label}: policies do not overlap')
    model = joblib.load(ROOT / 'models/final_model.joblib')
    by_id = table.set_index('cell_id')
    pred = pd.read_csv(ROOT / 'results/predictions.csv')
    evaluation = pd.read_csv(ROOT / 'results/evaluation_metrics.csv')
    for (name, subset), part in pred.groupby(['model','subset']):
        row = evaluation.loc[(evaluation.model == name) & (evaluation.subset == subset)].iloc[0]
        check(np.allclose(truth.loc[part.cell_id], part.actual), f'{name}/{subset}: original targets match')
        calc = metrics(part.actual, part.predicted)
        for key in ['MAPE (%)','MAE (cycles)','RMSE (cycles)','Bias (cycles)','R2']:
            check(np.isclose(calc[key], row[key], atol=1e-9), f'{name}/{subset}: {key} recalculated from cell predictions')
        if name == selection['selected_model']:
            again = model['model'].predict(by_id.loc[part.cell_id, model['features']])
            check(np.allclose(again, part.predicted, atol=1e-8), f'{subset}: persisted model reload reproduces predictions')
            key = {'Hold-out':'holdout_cells','Batch 2':'test_batch2_cells','Batch 3':'additional_batch3_cells'}[subset]
            check(set(part.cell_id) == set(split[key]), f'{subset}: exact frozen evaluation cells')
    comp = pd.read_csv(ROOT / 'results/cv_comparison.csv')
    check(comp.iloc[0].model == selection['selected_model'], 'Final model is the lowest development CV MAPE candidate')
    folds = pd.read_csv(ROOT / 'results/cv_fold_metrics.csv')
    oof = pd.read_csv(ROOT / 'results/cv_oof_predictions.csv')
    for name, part in oof.groupby('model'):
        check(len(part) == 28 and set(part.cell_id) == dev and part.cell_id.is_unique, f'{name}: one OOF prediction per development cell')
        check(set(part.cell_id).isdisjoint(hold), f'{name}: hold-out never appears in CV')
        for number, fold in enumerate(split['cv'], 1):
            p = part.loc[part.fold == number]
            check(set(p.cell_id) == set(fold['valid_cells']), f'{name}/fold {number}: frozen validation IDs')
            saved = folds.loc[(folds.model == name) & (folds.fold == number)].iloc[0]
            check(np.isclose(metrics(p.actual,p.predicted)['MAPE (%)'],saved['MAPE (%)']), f'{name}/fold {number}: fold MAPE matches predictions')
        c = comp.loc[comp.model == name].iloc[0]
        f = folds.loc[folds.model == name, 'MAPE (%)']
        check(np.isclose(f.mean(),c['CV MAPE mean (%)']) and np.isclose(f.std(ddof=1),c['CV MAPE std (%)']), f'{name}: CV mean and standard deviation match')
    perf = pd.read_csv(ROOT / 'results/model_performance.csv')['MAPE (%) 또는 Gap (%p)'].to_numpy()
    check(len(perf)==9, 'All six mandatory performance rows and three additional rows exist')
    check(np.allclose(perf[[3,4,5,7,8]], [perf[1]-perf[0],perf[2]-perf[1],perf[2]-9.1,perf[6]-perf[2],perf[6]-9.1]), 'Gap signs and percentage-point units are consistent')
    notebooks = list((ROOT / 'notebooks').glob('*.ipynb'))
    check(len(notebooks)==3, 'Three required workflow notebooks exist')
    count = 0
    for path in notebooks:
        nb = nbformat.read(path,as_version=4);nbformat.validate(nb)
        code = [c for c in nb.cells if c.cell_type=='code'];count += len(code)
        check(all(c.execution_count is not None for c in code), f'{path.name}: all code cells executed')
        check(not any(o.output_type=='error' for c in code for o in c.outputs), f'{path.name}: no error outputs')
    code_files = [*list((ROOT/'src').glob('*.py')),*list((ROOT/'scripts').glob('*.py'))]
    for path in code_files:
        compile(path.read_text(), str(path), 'exec')
    check(True, 'All source and reproduction scripts compile')
    for file in ['docs/모델_탐색_및_피처_정의.md','README.md','docs/DAY2_분석보고서.md']:
        check((ROOT/file).is_file(), f'{file}: required explanation exists')
    audit = {'status':'passed','total_checks':len(checks),'notebook_code_cells':count,'checks':checks}
    (ROOT/'results/validation_summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(f'Validation passed: {len(checks)} checks; {count} executed notebook cells.')
    return audit


if __name__ == '__main__':
    validate()
