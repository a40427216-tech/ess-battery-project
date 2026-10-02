"""Deterministic cycle-2–100 cell features; no cross-cell fitting."""
from pathlib import Path
import re
import json
import numpy as np
import pandas as pd

DISCHARGE = ['delta_q_logvar', 'delta_q_min', 'QD_2', 'QD_slope']
STATE = DISCHARGE + ['IR_mean', 'IR_change_100_2', 'Tavg_mean', 'chargetime_mean']
FEATURE_SETS = {
    'variance': ['delta_q_logvar'],
    'discharge': DISCHARGE,
    'state': STATE,
    'policy': STATE + ['c_rate_1', 'c_rate_2', 'switch_soc'],
    'current': STATE + ['charge_I_rms_c10', 'heat_proxy_c10_J'],
}
ALLOWED_FEATURES = set().union(*map(set, FEATURE_SETS.values()))


def validate_feature_columns(columns):
    if not columns or len(columns) != len(set(columns)) or not set(columns) <= ALLOWED_FEATURES:
        raise ValueError('Only explicitly whitelisted early-cycle features are allowed')
    return True

def policy_values(policy):
    match = re.search(r'([\d.]+)C\(([\d.]+)%\)-([\d.]+)C', policy)
    if not match:
        return np.nan, np.nan, np.nan, np.nan
    c1, soc, c2 = map(float, match.groups())
    # The named stages cover 0–80% SOC; after 80% all cells use a common CC-CV step.
    f = soc / 100
    effective = 0.8 / (f / c1 + (0.8 - f) / c2) if 0 < f <= 0.8 else np.nan
    return c1, soc, c2, effective

def interpolate_discharge(curves, cell_id, cycle, voltage):
    prefix = f'{cell_id}_c{cycle}_'
    i, v, q, t = (curves[prefix + field] for field in ['I', 'V', 'Qd', 't'])
    # Longest contiguous negative-current region avoids brief resistance pulses.
    mask = (i < -0.1) & np.isfinite(v) & np.isfinite(q) & np.isfinite(t)
    indices = np.flatnonzero(mask)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    runs = [r for r in runs if len(r) > 10]
    if not runs:
        return np.full(len(voltage), np.nan)
    run = max(runs, key=lambda r: t[r[-1]]-t[r[0]])
    # Some Batch 2 raw Qd points reset to exactly zero at low voltage while
    # current remains negative. They are not valid cumulative discharge charge.
    # Exclude those structural zeros before interpolation; retain the original
    # data and audit their count separately.
    run = run[q[run] > 1e-8]
    d = pd.DataFrame({'V': v[run], 'Q': q[run]}).groupby('V')['Q'].median().sort_index()
    out = np.interp(voltage, d.index.to_numpy(), d.to_numpy(), left=np.nan, right=np.nan)
    return out

def build_features(cells, summary, curves):
    rows, dq_arrays, quality = [], {}, []
    for _, cell in cells.iterrows():
        cid = cell.cell_id
        s = summary.loc[summary.cell_id == cid].sort_values('cycle')
        early = s.loc[(s.cycle >= 2) & (s.cycle <= 100)].copy()
        valid_q = early.QD.where((early.QD > 0) & (early.QD < 1.5))
        early.loc[early.IR <= 0, 'IR'] = np.nan
        # Unrealistic temperatures are missing sensor readings, not cool cells.
        for field in ['Tavg', 'Tmin', 'Tmax']:
            early.loc[(early[field] < -20) | (early[field] > 100), field] = np.nan
        row = cell.to_dict()
        row.update(dict(zip(['c_rate_1', 'switch_soc', 'c_rate_2', 'effective_c_rate'], policy_values(cell.charging_policy))))
        row['QD_mean'] = valid_q.mean()
        row['QD_std'] = valid_q.std()
        row['IR_mean'] = early.IR.mean()
        row['Tavg_mean'] = early.Tavg.mean()
        row['Tmax_mean'] = early.Tmax.mean()
        row['temperature_span'] = (early.Tmax - early.Tmin).mean()
        row['chargetime_mean'] = early.chargetime.where(early.chargetime > 0).mean()
        good = valid_q.notna()
        row['QD_slope'] = np.polyfit(early.loc[good, 'cycle'], valid_q[good], 1)[0] if good.sum() > 3 else np.nan
        qd_by_cycle = s.set_index('cycle')['QD']
        row['QD_2'] = qd_by_cycle.get(2, np.nan)
        row['QD_100'] = qd_by_cycle.get(100, np.nan)
        row['QD_change_100_2'] = row['QD_100'] - row['QD_2']
        row['QD_ratio_100_2'] = row['QD_100']/row['QD_2'] if row['QD_2'] > 0 else np.nan
        ir_by_cycle = early.set_index('cycle')['IR']
        row['IR_change_100_2'] = ir_by_cycle.get(100, np.nan) - ir_by_cycle.get(2, np.nan)
        prefix = f'{cid}_c10_'
        current, time = curves[prefix+'I'], curves[prefix+'t']
        positive = (current > 0.1) & np.isfinite(current) & np.isfinite(time)
        dt = np.diff(time)*60
        valid_segments = positive[:-1] & positive[1:] & (dt > 0)
        i2dt = ((current[:-1]**2 + current[1:]**2)/2*dt)[valid_segments].sum()
        duration = dt[valid_segments].sum()
        row['charge_I_rms_c10'] = float(np.sqrt(i2dt/duration)) if duration > 0 else np.nan
        row['charge_I_max_c10'] = float(np.max(current[positive])) if positive.any() else np.nan
        # Engineering proxy, not measured total heat. t is in minutes.
        row['heat_proxy_c10_J'] = float(i2dt*row['IR_mean'])
        for field in ['delta_q_logvar', 'delta_q_min', 'delta_q_mean', 'raw_delta_q_logvar']:
            row[field] = np.nan
        k10, k100 = f'{cid}_c10_Qdlin', f'{cid}_c100_Qdlin'
        if k10 in curves and k100 in curves:
            q10, q100 = curves[k10], curves[k100]
            if len(q10) == len(q100) and len(q10) > 20:
                delta = q100 - q10
                if np.isfinite(delta).all():
                    # Median-centering removes scalar capacity origin differences.
                    centered = delta - np.median(delta)
                    dq_arrays[cid] = {'raw': delta, 'centered': centered}
                    row['delta_q_logvar'] = np.log10(max(np.var(delta, ddof=1), 1e-15))
                    row['delta_q_min'] = float(np.min(centered))
                    row['delta_q_mean'] = float(np.mean(centered))
                    quality.append({'cell_id': cid, 'q10_first': float(q10[0]), 'q100_first': float(q100[0]),
                                    'delta_q_offset': float(np.median(delta)), 'finite': True})
            grid = np.linspace(2.1, 3.2, 500)
            raw10 = interpolate_discharge(curves, cid, 10, grid)
            raw100 = interpolate_discharge(curves, cid, 100, grid)
            rd = raw100 - raw10
            if np.isfinite(rd).all():
                row['raw_delta_q_logvar'] = float(np.log10(max(np.var(rd, ddof=1), 1e-15)))
        rows.append(row)
    return pd.DataFrame(rows), dq_arrays, pd.DataFrame(quality)


def generate(root=None):
    from .preprocess import ROOT, load_inputs
    root = Path(root) if root is not None else ROOT
    cells, summary, curves = load_inputs(root)
    table, _, quality = build_features(cells, summary, curves)
    curves.close()
    table['log_cycle_life'] = np.log10(table.cycle_life)
    for columns in FEATURE_SETS.values():
        validate_feature_columns(columns)
    (root / 'results').mkdir(exist_ok=True)
    table.to_csv(root / 'results/features.csv', index=False)
    quality.to_csv(root / 'results/delta_q_quality.csv', index=False)
    print(f'Generated {len(table)} independent cell rows; feature window: cycles 2–100.', flush=True)
    return table


if __name__ == '__main__':
    generate()
