"""Acquire frozen public benchmark files and reject changed bytes.

Public mirrors are download conveniences, not evidence of institution-signed
byte identity. Dataset-provider terms and citations are in DATA_TERMS.md.
"""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['bot','cic'],required=True);p.add_argument('--raw-root',type=Path,default=ROOT/'data/raw');a=p.parse_args()
    url={'bot':'https://www.kaggle.com/api/v1/datasets/download/majedjaber/bot-iot-all-features-5-sample',
         'cic':'https://www.kaggle.com/api/v1/datasets/download/himadri07/ciciot2023'}[a.dataset]
    expected=json.loads((ROOT/'data'/f'{a.dataset}_raw_manifest.json').read_text())
    expected={(r.get('file') or r.get('filename')):r for r in expected}
    destination=a.raw_root/a.dataset;destination.mkdir(parents=True,exist_ok=True)
    archive=destination/'download.zip'
    if not archive.exists():
        with urlopen(url,timeout=180) as r,archive.open('wb') as out:shutil.copyfileobj(r,out,4*1024*1024)
    with ZipFile(archive) as z:
        for name,record in expected.items():
            matches=[i for i in z.infolist() if Path(i.filename).name==name]
            if len(matches)!=1:raise RuntimeError('archive member missing or ambiguous '+name)
            out=destination/name
            with z.open(matches[0]) as src,out.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
            digest=hashlib.file_digest(out.open('rb'),'sha256').hexdigest()
            if out.stat().st_size!=record['bytes'] or digest!=record['sha256']:raise RuntimeError('frozen dataset checksum mismatch '+name)
            print(name,'verified',out.stat().st_size,flush=True)

if __name__=='__main__':main()
