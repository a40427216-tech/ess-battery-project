"""Tests for target metrics, group separation and future-information isolation."""
import copy
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.features import ALLOWED_FEATURES, build_features, validate_feature_columns
from src.preprocess import ROOT, load_features, load_inputs, validate_split, validate_input_manifest
from src.train import metrics, performance_rows, pipeline


class PipelineTests(unittest.TestCase):
    def test_mape_and_gap_units(self):
        result = metrics([100, 200], [110, 180])
        self.assertAlmostEqual(result['MAPE (%)'], 10)
        self.assertAlmostEqual(result['Bias (cycles)'], -5)
        rows = performance_rows(5, 1, 8, 12, 10)
        self.assertAlmostEqual(rows[3]['MAPE (%) 또는 Gap (%p)'], 3)
        self.assertAlmostEqual(rows[4]['MAPE (%) 또는 Gap (%p)'], 4)
        self.assertAlmostEqual(rows[5]['MAPE (%) 또는 Gap (%p)'], 2.9)
        self.assertAlmostEqual(rows[7]['MAPE (%) 또는 Gap (%p)'], -2)

    def test_reject_policy_overlap(self):
        features, split = load_features()
        bad = copy.deepcopy(split)
        bad['development_cells'][0], bad['holdout_cells'][0] = (
            bad['holdout_cells'][0], bad['development_cells'][0])
        with self.assertRaises(ValueError):
            validate_split(features, bad)

    def test_future_and_identity_columns_are_rejected(self):
        for name in ['cycle_life', 'last_QD', 'last_cycle', 'n_summary', 'cell_id', 'batch', 'knee']:
            with self.assertRaises(ValueError):
                validate_feature_columns(['delta_q_logvar', name])

    def test_imputer_scaler_fit_training_only(self):
        x = pd.DataFrame({'a': [1., np.nan, 3.], 'b': [2., 4., 6.]})
        model = pipeline(Ridge()).fit(x, [500., 600., 700.])
        fitted = model.regressor_
        median = fitted['imputer'].statistics_.copy()
        mean = fitted['scaler'].mean_.copy()
        model.predict(pd.DataFrame({'a': [100., np.nan], 'b': [-100., 100.]}))
        np.testing.assert_allclose(median, [2., 4.])
        np.testing.assert_allclose(mean, [2., 4.])
        np.testing.assert_array_equal(median, fitted['imputer'].statistics_)
        np.testing.assert_array_equal(mean, fitted['scaler'].mean_)

    def test_future_summary_changes_do_not_change_early_features(self):
        cells, summary, curves = load_inputs()
        cells = cells.iloc[:1].copy()
        early = summary.loc[summary.cell_id == cells.iloc[0].cell_id].copy()
        future = early.copy()
        future['cycle'] += 1000
        for column in ['QD', 'IR', 'Tavg', 'Tmax', 'Tmin', 'chargetime']:
            future[column] = 1e9
        original, _, _ = build_features(cells, early, curves)
        altered, _, _ = build_features(cells, pd.concat([early, future]), curves)
        curves.close()
        columns = sorted(ALLOWED_FEATURES)
        np.testing.assert_allclose(original[columns], altered[columns], equal_nan=True)

    def test_processed_data_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='ess-provenance-test-') as directory:
            root=Path(directory);(root/'data/processed').mkdir(parents=True)
            original=b'a'
            record={'file':'processed/example.csv','bytes':1,'sha256':hashlib.sha256(original).hexdigest()}
            (root/'data/processed_manifest.json').write_text(json.dumps([record]))
            (root/'data/processed/example.csv').write_bytes(original)
            self.assertTrue(validate_input_manifest(root))
            (root/'data/processed/example.csv').write_bytes(b'b')
            with self.assertRaises(ValueError):
                validate_input_manifest(root)


if __name__ == '__main__':
    unittest.main()
