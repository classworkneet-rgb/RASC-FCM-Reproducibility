"""Independently recompute every metric from archived probability matrices."""
from pathlib import Path
import json,hashlib
import numpy as np
from experiment import ROOT,read_data,metrics
from model import ArchitectureSpec,FixedGatedDBNCNN

def main():
    datasets={};checks=[]
    for path in sorted((ROOT/'results/runs').glob('*.json')):
        record=json.loads(path.read_text());cfg=record['config'];name=cfg['dataset']
        if name not in datasets:datasets[name]=read_data(name)
        d=datasets[name];a=np.load(path.with_suffix('.npz'));prob=a['probabilities']
        assert prob.shape==(len(d['test'][1]),int(d['train'][1].max())+1)
        assert np.isfinite(prob).all() and (prob>=0).all()
        np.testing.assert_allclose(prob.sum(axis=1),1.,atol=2e-7,rtol=0)
        model=FixedGatedDBNCNN(ArchitectureSpec(d['train'][0].shape[1],prob.shape[1]),seed=0)
        model.load_parameter_vector(a['parameters'])
        reconstructed=model.predict_proba(d['test'][0],np.asarray(record['mask'],bool))
        np.testing.assert_array_equal(reconstructed.astype(np.float32),prob)
        actual=metrics(d['test'][1],reconstructed)
        max_error=max(abs(actual[k]-record['test'][k]) for k in ['macro_f1','accuracy','balanced_accuracy','weighted_f1','mcc'])
        assert max_error<1e-12
        for k in ['macro_pr_auc','macro_roc_auc','ece']:assert abs(actual[k]-record['test'][k])<2e-5
        clients=[a[f'client_{k}'] for k in range(cfg['clients'])];joined=np.concatenate(clients)
        assert len(joined)==len(d['train'][1]) and np.array_equal(np.sort(joined),np.arange(len(joined)))
        assert hashlib.sha256(b''.join(i.tobytes() for i in clients)).hexdigest()==record['client_partition_sha256']
        checks.append({'run_id':record['run_id'],'predictions':'passed','metrics':'passed','partition_cover':'passed','max_discrete_metric_error':max_error})
    report={'checked_runs':len(checks),'all_checks_passed':True,'runs':checks}
    (ROOT/'results/prediction_verification.json').write_text(json.dumps(report,indent=2))
    print('Verified predictions, metrics and partition coverage for',len(checks),'runs.')

if __name__=='__main__':main()
