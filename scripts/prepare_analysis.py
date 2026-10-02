"""Define an EOL-labelled cohort, without copying Batch 2 index corrections."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
RESULTS = ROOT/'results'

def prepare():
    RESULTS.mkdir(exist_ok=True)
    cells = pd.read_csv(DATA/'cells_raw.csv')
    summaries = pd.read_csv(DATA/'summary_raw.csv.gz')
    curves = np.load(DATA/'early_curves.npz')
    # Only Batch 3 has the exact source date used in the published loader.
    published_batch3_noisy = {'b3c37','b3c2','b3c23','b3c32','b3c42','b3c43'}
    exclusions, kept = [], []
    for _, cell in cells.iterrows():
        reasons=[]
        if not np.isfinite(cell.cycle_life_raw):
            reasons.append('missing_cycle_life')
        # Source Batch 1 labels often mark one past the last observed cycle,
        # even when the trajectory ends far above the nominal 80% threshold.
        # 0.885Ah tolerates one-cycle discretisation near 0.88Ah.
        if cell.last_QD > 0.885:
            reasons.append('EOL_not_observed_or_incomplete_record')
        if cell.cell_id in published_batch3_noisy:
            reasons.append('published_batch3_noisy_channel')
        if np.isfinite(cell.cycle_life_raw) and cell.cycle_life_raw <= 100:
            reasons.append('insufficient_life_for_100_cycle_prediction')
        for c in [10,100]:
            key=f'{cell.cell_id}_c{c}_Qdlin'
            if key not in curves or not np.isfinite(curves[key]).all() or len(curves[key])<100:
                reasons.append(f'invalid_Qdlin_cycle_{c}')
        if reasons:
            exclusions.append({**cell.to_dict(),'reasons':';'.join(reasons)})
        else:
            kept.append({**cell.to_dict(),'cycle_life':cell.cycle_life_raw})
    cohort=pd.DataFrame(kept)
    cohort.to_csv(DATA/'cells_analysis.csv',index=False)
    summaries.loc[summaries.cell_id.isin(cohort.cell_id)].to_csv(DATA/'summary_analysis.csv.gz',index=False)
    pd.DataFrame(exclusions).to_csv(RESULTS/'cell_exclusions.csv',index=False)
    audits=json.loads((DATA/'extraction_audit.json').read_text())
    errors=[abs(a['raw_max_qd']-a['summary_qd']) for record in audits for a in record['alignment'] if np.isfinite(a['raw_max_qd']) and np.isfinite(a['summary_qd'])]
    assert max(errors)<1e-4, f'Actual cycle-number alignment failed: {max(errors)}'
    b2=cells.loc[cells.batch==2]
    raw_stats=cells.groupby('batch').cycle_life_raw.agg(['count','mean','median','min','max'])
    raw_stats.to_csv(RESULTS/'raw_label_statistics.csv')
    audit={'raw_cells':len(cells),'analysis_cells':len(cohort),
        'raw_counts':{str(k):int(v) for k,v in cells.groupby('batch').size().items()},
        'analysis_counts':{str(k):int(v) for k,v in cohort.groupby('batch').size().items()},
        'excluded_cells':len(exclusions),'missing_labels':int(cells.cycle_life_raw.isna().sum()),
        'eol_tolerance_Ah':0.885,'eol_nominal_Ah':0.88,
        'raw_vs_summary_qd_max_error_Ah':float(max(errors)),
        'batch2_newstructure_cells':int(b2.charging_policy.str.contains('newstructure',case=False).sum()),
        'batch2_missing_labels':b2.loc[b2.cycle_life_raw.isna(),'cell_id'].tolist(),
        'batch1_incomplete_cells':cells.loc[(cells.batch==1)&(cells.last_QD>0.885),'cell_id'].tolist(),
        'batch2_continuation_merge_applied':False,
        'reason_for_no_merge':'2018-02-20 source is not the 2017-06-30 continuation batch used in the published loader; policies differ and barcode-like descriptors repeat.',
        'cycle_number_rule':'summary.cycle labels are matched directly to HDF5 cycles; discharge capacities verified.',
        'feature_window':'2 through 100 inclusive; the structural zero in Batch 1 cycle 1 is excluded.',
        'batch3_exclusion_source':'https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/Load%20Data.ipynb'}
    (RESULTS/'quality_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':prepare()
