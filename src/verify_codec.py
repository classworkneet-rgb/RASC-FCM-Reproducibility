"""Measure the optional deployment codec on all saved masked models."""
from pathlib import Path
import json
import numpy as np
from experiment import ROOT,read_data
from model import ArchitectureSpec,FixedGatedDBNCNN
from parameter_codec import encode,decode

def main():
    datasets={};checks=[]
    for p in sorted((ROOT/'results/runs').glob('*.json')):
        r=json.loads(p.read_text());c=r['config']
        if c['mask']=='full':continue
        if c['dataset'] not in datasets:datasets[c['dataset']]=read_data(c['dataset'])
        data=datasets[c['dataset']];archive=np.load(p.with_suffix('.npz'))
        spec=ArchitectureSpec(data['train'][0].shape[1],int(data['train'][1].max())+1)
        model=FixedGatedDBNCNN(spec,seed=0);model.load_parameter_vector(archive['parameters']);mask=np.asarray(r['mask'],bool)
        payload=encode(model,mask);full=model.parameter_vector().nbytes
        before=model.predict_proba(data['test'][0],mask);model.load_parameter_vector(decode(payload,model,mask));after=model.predict_proba(data['test'][0],mask)
        error=float(np.max(np.abs(before-after)));assert error<2e-15
        checks.append({'run_id':r['run_id'],'dataset':c['dataset'],'dense_bytes':full,'packed_model_bytes':len(payload),'fraction_reduction':1-len(payload)/full,'max_probability_difference':error})
    (ROOT/'results/codec_verification.json').write_text(json.dumps({'checked_models':len(checks),'all_checks_passed':True,'scope':'inference-model serialization only, not network/training measurement','models':checks},indent=2))
    print('Verified inference-codec round trips for',len(checks),'models.')

if __name__=='__main__':main()
