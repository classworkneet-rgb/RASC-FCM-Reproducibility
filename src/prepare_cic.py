"""CICIoT2023 external cohort using the public mirror's train/val/test files.

All three file hashes are frozen. Exact cleaned numeric feature vectors are
deduplicated across the complete acquired files, with train then validation
then test priority. Contradictory eight-category labels for identical vectors
are removed. Caps are applied only after that audit. No chronological,
capture-independent, or cross-dataset transfer claim is made.
"""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd

import argparse
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw/cic');args=p.parse_args();RAW=args.raw_dir
frozen_files=json.loads((ROOT/'data/cic_raw_manifest.json').read_text())
for record in frozen_files:
 raw_path=RAW/record['filename']
 assert raw_path.stat().st_size==record['bytes'],'raw CIC byte count mismatch'
 assert hashlib.file_digest(raw_path.open('rb'),'sha256').hexdigest()==record['sha256'],'raw CIC checksum mismatch'
classes=['Benign','DDoS','DoS','Mirai','Recon','Spoofing','Web','BruteForce']
mapping={c:i for i,c in enumerate(classes)}
def group(label):
 s=label.strip().lower()
 if s.startswith('benign'):return 'Benign'
 if s.startswith('ddos'):return 'DDoS'
 if s.startswith('dos'):return 'DoS'
 if s.startswith('mirai'):return 'Mirai'
 if s.startswith('recon') or s=='vulnerabilityscan':return 'Recon'
 if s in ['dns_spoofing','mitm-arpspoofing']:return 'Spoofing'
 if s=='dictionarybruteforce':return 'BruteForce'
 if s in ['backdoor_malware','browserhijacking','commandinjection','sqlinjection','uploading_attack','xss']:return 'Web'
 raise ValueError('unknown CICIoT2023 label '+label)

features=pd.read_csv(RAW/'test.csv',nrows=0).columns[:-1].tolist()
(ROOT/'data/cache').mkdir(parents=True,exist_ok=True)
metadata=[];cache=ROOT/'data/cache/cic_metadata.pkl';offset=0
if cache.exists():meta=pd.read_pickle(cache)
else:
 for split in ['train','validation','test']:
  for chunk in pd.read_csv(RAW/f'{split}.csv',chunksize=100000,low_memory=False):
   labels=chunk['label'].astype(str).map(group).map(mapping).to_numpy(np.int64)
   numeric=chunk[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0.).astype(np.float64)
   h=pd.util.hash_pandas_object(numeric,index=False,hash_key='rascfcmgroup2026').to_numpy(np.uint64)
   # Numeric columns use the same stable hashing algorithm irrespective of
   # hash_key, so the collision guard is a second SHA-256 row digest for
   # selected rows during verification, not a claimed independent pandas hash.
   metadata.append(pd.DataFrame({'id':np.arange(offset,offset+len(chunk)),'split':split,'y':labels,'h':h}))
   offset+=len(chunk);print('scan',split,offset,flush=True)
 meta=pd.concat(metadata,ignore_index=True);meta.to_pickle(cache)
conflicts=meta.groupby('h',sort=False)['y'].transform('nunique')>1
clean=meta.loc[~conflicts].drop_duplicates('h',keep='first')
caps={'train':10000,'validation':2500,'test':10000};splits={}
for split in caps:
 ids=[]
 for c in range(len(classes)):
  values=clean.loc[(clean['split']==split)&(clean['y']==c),'id'].to_numpy(np.int64)
  cap=caps[split];ids.extend(values if len(values)<=cap else values[np.linspace(0,len(values)-1,cap,dtype=int)])
 splits[split]=np.sort(ids)
selected=np.sort(np.concatenate(list(splits.values())));frames=[];offset=0
for split in ['train','validation','test']:
 for chunk in pd.read_csv(RAW/f'{split}.csv',chunksize=100000,low_memory=False):
  a=np.searchsorted(selected,offset);b=np.searchsorted(selected,offset+len(chunk));ids=selected[a:b]
  if len(ids):
   frame=chunk.iloc[ids-offset].copy();frame['__id']=ids;frame['__split']=split;frames.append(frame)
  offset+=len(chunk)
rows=pd.concat(frames,ignore_index=True);train=rows['__split']=='train'
x=rows[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
med=x.loc[train].median().fillna(0.);x=x.fillna(med)
lo=x.loc[train].quantile(.005);hi=x.loc[train].quantile(.995);x=x.clip(lo,hi,axis=1)
center=x.loc[train].median();scale=x.loc[train].quantile(.75)-x.loc[train].quantile(.25);scale=scale.where(scale.abs()>1e-12,1.)
x=((x-center)/scale).clip(-12,12)
data={};selected_sha={}
for split in splits:
 keep=rows['__split']==split;sub=rows.loc[keep];labels=sub['label'].astype(str).map(group).map(mapping).to_numpy(np.int64)
 order=np.lexsort((sub['__id'].to_numpy(),labels));xx=x.loc[keep].to_numpy(np.float64)[order]
 data[split]=(xx,labels[order],sub['__id'].to_numpy(np.int64)[order])
 assert len(np.unique(labels))==len(classes),'external split missing class'
 selected_sha[split]=np.asarray([hashlib.sha256(r.tobytes()).hexdigest() for r in rows.loc[keep,features].to_numpy(np.float64)[order]])
for a,b in [('train','validation'),('train','test'),('validation','test')]:
 assert not(set(selected_sha[a])&set(selected_sha[b]))
np.savez_compressed(ROOT/'data/cic_primary.npz',**{f'{s}_{k}':a for s,v in data.items() for k,a in zip(['x','y','ids'],v)})
np.savez_compressed(ROOT/'data/cic_primary_split.npz',**splits)
np.savez_compressed(ROOT/'data/cic_primary_raw_feature_sha256.npz',**selected_sha)
prep={'medians':med.to_dict(),'winsor_low':lo.to_dict(),'winsor_high':hi.to_dict(),'robust_center':center.to_dict(),'robust_scale':scale.to_dict()}
(ROOT/'data/cic_primary_preprocessing.json').write_text(json.dumps(prep,indent=2))
manifest={'dataset':'CICIoT2023 public mirror cohort','files':frozen_files,'features':features,'classes':classes,
 'acquired_rows':len(meta),'raw_split_class_counts':{s:dict(zip(classes,np.bincount(meta.loc[meta['split']==s,'y'],minlength=len(classes)).tolist())) for s in splits},
 'conflicting_vector_label_rows_removed':int(conflicts.sum()),'exact_vector_duplicates_removed':int((~conflicts).sum()-len(clean)),
 'selected_split_class_counts':{s:dict(zip(classes,np.bincount(v[1],minlength=len(classes)).tolist())) for s,v in data.items()},
 'split':'public mirror train/validation/test membership; global exact numeric vector deduplication; class caps 10000/2500/10000',
 'chronological':False,'capture_independent':False,'candidate_budget':round(.6*len(features)),
 'source_authority':'https://www.unb.ca/cic/datasets/iotdataset-2023.html','publication_doi':'10.3390/s23135941',
 'raw_original_archive_byte_identity_established':False,'evidence_boundary':'hash-verified public-mirror derivative; not all official 46.7 million records'}
(ROOT/'data/cic_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)
