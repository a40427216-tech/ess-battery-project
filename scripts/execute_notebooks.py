"""Execute bundled notebooks using the current Python in a temporary kernel spec."""
from pathlib import Path
import json
import os
import sys
import tempfile
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def execute():
    with tempfile.TemporaryDirectory(prefix='ess-kernel-') as directory:
        kernel=Path(directory)/'kernels/ess-project'
        kernel.mkdir(parents=True)
        (kernel/'kernel.json').write_text(json.dumps({
            'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'ESS Project','language':'python'}))
        original=os.environ.get('JUPYTER_PATH')
        os.environ['JUPYTER_PATH']=directory
        try:
            for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
                notebook=nbformat.read(path,as_version=4)
                NotebookClient(notebook,timeout=120,kernel_name='ess-project',
                               resources={'metadata':{'path':str(ROOT)}}).execute()
                nbformat.write(notebook,path)
                print('Executed',path.name,flush=True)
        finally:
            if original is None:os.environ.pop('JUPYTER_PATH',None)
            else:os.environ['JUPYTER_PATH']=original


if __name__=='__main__':
    execute()
