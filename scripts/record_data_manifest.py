"""Record processed-data fingerprints after an explicit raw-data rebuild."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def record():
    files = sorted((ROOT/'data/processed').glob('*'))
    records = [{'file':str(p.relative_to(ROOT/'data')), 'bytes':p.stat().st_size,
                'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in files if p.is_file()]
    (ROOT/'data/processed_manifest.json').write_text(json.dumps(records,indent=2))
    print(f'Recorded {len(records)} processed input files.')


if __name__=='__main__':
    record()
