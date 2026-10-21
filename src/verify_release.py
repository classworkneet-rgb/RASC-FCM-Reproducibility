"""Verify the frozen SHA-256 member manifest before reuse."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 manifest=ROOT/'SHA256SUMS.txt'
 if not manifest.exists():raise SystemExit('SHA256SUMS.txt is missing')
 count=0
 for line in manifest.read_text().splitlines():
  digest,rel=line.split('  ',1);path=ROOT/rel
  if not path.is_file():raise SystemExit('Missing release member: '+rel)
  actual=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
  if actual!=digest:raise SystemExit('Release checksum mismatch: '+rel)
  count+=1
 print('Verified SHA-256 for',count,'release members.')
if __name__=='__main__':main()
