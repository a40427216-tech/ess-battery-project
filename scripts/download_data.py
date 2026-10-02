"""Download the three assignment batches from the instructor's Kaggle source."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import time
import threading
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw'
DATA.mkdir(parents=True, exist_ok=True)
FILES = [
    '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
]

def download(name):
    dest = DATA / name
    url = 'https://www.kaggle.com/api/v1/datasets/download/itshpark/data-driven-prediction-of-battery-cycle/' + name
    archive = DATA / (name + '.zip')
    if not dest.exists():
        req = urllib.request.Request(url, headers={'Range': 'bytes=0-0', 'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as response:
            size = int(response.headers['Content-Range'].split('/')[-1])
            direct = response.geturl()
        state_path = archive.with_suffix('.progress.json')
        if state_path.exists():
            state = json.loads(state_path.read_text())
        else:
            state = {'prefix': archive.stat().st_size if archive.exists() else 0, 'completed': []}
            state_path.write_text(json.dumps(state))
        with archive.open('a+b') as output:
            output.truncate(size)
        done = state['prefix'] + sum(end - start + 1 for start, end in state['completed'])
        print(f'START {name} archive_bytes={size} already={done}', flush=True)
        lock = threading.Lock()
        last = time.monotonic()
        chunks = [(start, min(size - 1, start + 16 * 1024**2 - 1)) for start in range(state['prefix'], size, 16 * 1024**2)]
        chunks = [part for part in chunks if list(part) not in state['completed']]
        def part_download(part):
            nonlocal done, last
            start, end = part
            for attempt in range(4):
                try:
                    request = urllib.request.Request(direct, headers={'Range': f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        assert response.status == 206
                        payload = response.read()
                    assert len(payload) == end - start + 1
                    with archive.open('r+b') as output:
                        output.seek(start)
                        output.write(payload)
                    with lock:
                        state['completed'].append(list(part))
                        state_path.write_text(json.dumps(state))
                        done += len(payload)
                        if time.monotonic() - last > 30:
                            print(f'DOWNLOAD {name[:10]} {done/1024**2:.0f} MiB / {size/1024**2:.0f}', flush=True)
                            last = time.monotonic()
                    return
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
        with ThreadPoolExecutor(max_workers=4) as parts:
            list(parts.map(part_download, chunks))
        with zipfile.ZipFile(archive) as z:
            assert z.namelist() == [name], z.namelist()
            z.extract(name, DATA)
        archive.unlink()
        state_path.unlink()
    h = hashlib.sha256()
    with dest.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    print(f'READY {name} bytes={dest.stat().st_size}', flush=True)
    return {'file': name, 'source_url': url, 'bytes': dest.stat().st_size, 'sha256': h.hexdigest()}

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, FILES))
    (DATA.parent / 'source_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
