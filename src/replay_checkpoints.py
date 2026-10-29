"""Rebuild exact inference outputs from compact checkpoints and provider cohorts.

No training is performed. Existing run arrays are checked, never replaced.
"""
import argparse, hashlib, json
from datetime import datetime, timezone
import numpy as np
from experiment import ROOT, read_data, partition, metrics
from model import ArchitectureSpec, FixedGatedDBNCNN

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-run-arrays',action='store_true')
    args=parser.parse_args()
    hashes=json.loads((ROOT/'configs/prediction_array_checksums.json').read_text())
    datasets={};rows=[];earlier={}
    methods={('fedavg','full'):'FedAvg-full',('dca','prevalence'):'Prevalence-FCM-DCA',('fedavg','rasc'):'RASC-FedAvg',('dca','rasc'):'RASC-FCM-DCA'}
    for path in sorted((ROOT/'results/runs').glob('*.json')):
        r=json.loads(path.read_text());c=r['config'];rid=r['run_id']
        if c['dataset'] not in datasets:datasets[c['dataset']]=read_data(c['dataset'])
        d=datasets[c['dataset']]
        with np.load(ROOT/'checkpoints'/f'{rid}.npz') as a:parameters=a['parameters']
        assert digest(parameters)==hashes[rid]['parameters_sha256']
        model=FixedGatedDBNCNN(ArchitectureSpec(d['train'][0].shape[1],int(d['train'][1].max())+1),seed=0)
        model.load_parameter_vector(parameters)
        pred64=model.predict_proba(d['test'][0],np.asarray(r['mask'],bool));pred32=pred64.astype(np.float32)
        assert digest(pred32)==hashes[rid]['probabilities_sha256'],rid
        clients=partition(d['train'][1],c.get('partition_seed',c['seed']),c['clients'],c['alpha'])
        assert hashlib.sha256(b''.join(x.tobytes() for x in clients)).hexdigest()==r['client_partition_sha256']
        actual=metrics(d['test'][1],pred64)
        for k in ['macro_f1','accuracy','balanced_accuracy','weighted_f1','mcc']:assert abs(actual[k]-r['test'][k])<1e-12,(rid,k)
        for k in ['macro_pr_auc','macro_roc_auc','ece']:assert abs(actual[k]-r['test'][k])<2e-5,(rid,k)
        target=path.with_suffix('.npz')
        if target.exists():
            with np.load(target) as a:
                np.testing.assert_array_equal(a['parameters'],parameters)
                np.testing.assert_array_equal(a['probabilities'],pred32)
                for k,x in enumerate(clients):np.testing.assert_array_equal(a[f'client_{k}'],x)
        elif args.restore_run_arrays:
            temporary=target.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,parameters=parameters,probabilities=pred32,**{f'client_{k}':x for k,x in enumerate(clients)})
            with np.load(temporary) as check:
                np.testing.assert_array_equal(check['parameters'],parameters)
                np.testing.assert_array_equal(check['probabilities'],pred32)
            temporary.replace(target)
        if c['dataset']=='bot_primary' and c['rounds']==40 and c['clients']==5 and c['alpha']==.5 and (c['algorithm'],c['mask']) in methods:
            earlier[methods[c['algorithm'],c['mask']]+'_'+str(c['seed'])]=pred32
        rows.append({'run_id':rid,'parameter_and_probability_hashes':'passed','metrics':'passed','client_indices':'passed'})
    assert len(rows)==320 and len(earlier)==40
    if args.restore_run_arrays:
        p=ROOT/'results/reference_predictions.npz';p.parent.mkdir(exist_ok=True)
        if not p.exists():
            d=datasets['bot_primary'];np.savez_compressed(p,y_true=d['test'][1],test_global_index=d['test'][2],**earlier)
        else:
            with np.load(p) as a:
                for key,value in earlier.items():np.testing.assert_array_equal(a[key],value)
    report={'date':datetime.now(timezone.utc).date().isoformat(),'status':'passed','runs':len(rows),'earlier_matrices':len(earlier),'training_performed':False,'checks':rows}
    (ROOT/'results/checkpoint_replay_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified compact checkpoints, exact probabilities, metrics and client indices for',len(rows),'runs.')

if __name__=='__main__':main()
