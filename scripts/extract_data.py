"""Read only the needed HDF5 fields; retain raw provenance and real cycle labels."""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
OUT = ROOT / 'data/processed'
FILES = ['2017-05-12', '2018-02-20', '2018-04-12']
FIELDS = {'cycle': 'cycle', 'QDischarge': 'QD', 'QCharge': 'QC', 'IR': 'IR',
          'Tmax': 'Tmax', 'Tavg': 'Tavg', 'Tmin': 'Tmin', 'chargetime': 'chargetime'}

def array(f, reference):
    return np.asarray(f[reference][()]).ravel().astype(float)

def extract():
    OUT.mkdir(parents=True, exist_ok=True)
    cell_rows, summary_rows, audit = [], [], []
    arrays = {}
    for batch_id, date in enumerate(FILES, 1):
        path = RAW / f'{date}_batchdata_updated_struct_errorcorrect.mat'
        with h5py.File(path, 'r') as f:
            batch = f['batch']
            print('BATCH', batch_id, 'keys', list(batch), flush=True)
            n_cells = batch['summary'].shape[0]
            for i in range(n_cells):
                cell_id = f'b{batch_id}c{i}'  # Explicitly zero-based cell numbering.
                policy = ''.join(chr(int(x)) for x in f[batch['policy_readable'][i, 0]][()].ravel())
                cycle_life = float(array(f, batch['cycle_life'][i, 0])[0])
                sg = f[batch['summary'][i, 0]]
                summaries = {name: np.asarray(sg[key][()]).ravel().astype(float) for key, name in FIELDS.items()}
                lengths = {key: len(value) for key, value in summaries.items()}
                assert len(set(lengths.values())) == 1, (cell_id, lengths)
                frame = pd.DataFrame(summaries)
                frame['cell_id'], frame['batch'] = cell_id, batch_id
                summary_rows.append(frame)
                cg = f[batch['cycles'][i, 0]]
                cycle_count = cg['I'].shape[0]
                # Summary.cycle gives the true cycle number; no c+1 relabelling.
                # Check alignment using raw and summary discharge capacities.
                checks = []
                for cycle_number in (2, 10, 100):
                    matches = np.flatnonzero(summaries['cycle'] == cycle_number)
                    if len(matches) != 1:
                        continue
                    j = int(matches[0])
                    if j >= cycle_count:
                        continue
                    for field in ['I', 'V', 't', 'Qd', 'Qc', 'T', 'Qdlin']:
                        arrays[f'{cell_id}_c{cycle_number}_{field}'] = array(f, cg[field][j, 0])
                    raw_qd = arrays[f'{cell_id}_c{cycle_number}_Qd']
                    checks.append({'cycle': cycle_number, 'index': j, 'raw_max_qd': float(np.nanmax(raw_qd)),
                                   'summary_qd': float(summaries['QD'][j])})
                if 'Vdlin' in batch:
                    arrays[f'{cell_id}_Vdlin'] = array(f, batch['Vdlin'][i, 0])
                descriptor = {'cell_id': cell_id, 'batch': batch_id, 'source_file': path.name,
                              'charging_policy': policy, 'cycle_life_raw': cycle_life,
                              'n_summary': len(frame), 'n_cycles': cycle_count,
                              'last_cycle': float(frame['cycle'].iloc[-1]),
                              'last_QD': float(frame['QD'].iloc[-1])}
                # Barcode/channel are supplementary identity evidence when present.
                for field in ['barcode', 'channel_id']:
                    if field in batch:
                        obj = f[batch[field][i, 0]][()]
                        if f[batch[field][i, 0]].attrs.get('MATLAB_class') == b'char':
                            descriptor[field] = ''.join(chr(int(x)) for x in obj.ravel())
                        else:
                            descriptor[field] = np.asarray(obj).ravel().tolist()
                cell_rows.append(descriptor)
                audit.append({'cell_id': cell_id, 'alignment': checks,
                              'first_cycle_values': summaries['cycle'][:5].tolist(),
                              'last_cycle_values': summaries['cycle'][-5:].tolist()})
            print('EXTRACTED', date, n_cells, flush=True)
    pd.DataFrame(cell_rows).to_csv(OUT / 'cells_raw.csv', index=False)
    pd.concat(summary_rows, ignore_index=True).to_csv(OUT / 'summary_raw.csv.gz', index=False)
    np.savez_compressed(OUT / 'early_curves.npz', **arrays)
    (OUT / 'extraction_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')

if __name__ == '__main__':
    extract()
