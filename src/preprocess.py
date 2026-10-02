"""Load the frozen DAY 1 cohort and split without fitting preprocessing globally."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def validate_input_manifest(root=ROOT):
    root=Path(root)
    records=json.loads((root/'data/processed_manifest.json').read_text())
    for record in records:
        path=root/'data'/record['file']
        if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f"Processed data changed: {record['file']}; rebuild and record provenance explicitly")
    return True


def load_inputs(root=ROOT):
    root = Path(root)
    validate_input_manifest(root)
    data = root / 'data/processed'
    cells = pd.read_csv(data / 'cells_analysis.csv')
    summary = pd.read_csv(data / 'summary_analysis.csv.gz')
    curves = np.load(data / 'early_curves.npz')
    if not cells.cell_id.is_unique:
        raise ValueError('Duplicate cell IDs')
    if not np.isfinite(cells.cycle_life).all() or not (cells.cycle_life > 100).all():
        raise ValueError('Targets must be finite and greater than 100')
    # Later observations can be used in EDA, but are never passed to feature extraction.
    summary = summary.loc[summary.cycle.between(2, 100)].copy()
    return cells, summary, curves


def validate_split(features, split):
    by_id = features.set_index('cell_id')
    if not by_id.index.is_unique:
        raise ValueError('Duplicate cell IDs')
    dev = set(split['development_cells'])
    hold = set(split['holdout_cells'])
    if dev & hold or dev | hold != set(features.loc[features.batch == 1, 'cell_id']):
        raise ValueError('Development and hold-out must partition Batch 1')
    groups = by_id.charging_policy
    if set(groups.loc[list(dev)]) & set(groups.loc[list(hold)]):
        raise ValueError('Hold-out charging policies overlap development')
    valid_all = []
    for fold in split['cv']:
        train, valid = set(fold['train_cells']), set(fold['valid_cells'])
        if train & valid or train | valid != dev:
            raise ValueError('CV fold does not partition development')
        if set(groups.loc[list(train)]) & set(groups.loc[list(valid)]):
            raise ValueError('CV charging policies overlap')
        valid_all.extend(valid)
    if len(valid_all) != len(dev) or set(valid_all) != dev:
        raise ValueError('Every development cell needs exactly one OOF prediction')
    for batch, key in [(2, 'test_batch2_cells'), (3, 'additional_batch3_cells')]:
        if set(split[key]) != set(features.loc[features.batch == batch, 'cell_id']):
            raise ValueError(f'Batch {batch} does not match the frozen split')
    return True


def load_features(root=ROOT):
    root = Path(root)
    table = pd.read_csv(root / 'results/features.csv')
    split = json.loads((root / 'results/split_design.json').read_text())
    validate_split(table, split)
    return table, split
