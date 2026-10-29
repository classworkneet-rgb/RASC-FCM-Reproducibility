"""Verify the frozen experimental protocol and saved model outputs."""
from pathlib import Path
import collections, hashlib, json, sys
from datetime import datetime, timezone
import numpy as np
from experiment import ROOT, read_data, partition

def vector_set(x):
    a=np.ascontiguousarray(x)
    return set(a.view(np.dtype((np.void,a.dtype.itemsize*a.shape[1]))).ravel().tolist())

def main():
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'results/runs').glob('*.json'))]
    assert len(records)==320
    assert len(list((ROOT/'results/tuning').glob('*.json')))==36
    datasets={n:read_data(n) for n in {r['config']['dataset'] for r in records}}
    expected=json.loads((ROOT/'data/transformed_feature_overlap_audit.json').read_text())
    overlap={}
    for name,d in datasets.items():
        for s in ('train','validation','test'):
            x,y,ids=d[s]
            assert x.ndim==2 and len(x)==len(y)==len(ids) and np.isfinite(x).all()
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            assert not np.intersect1d(d[a][2],d[b][2]).size
        sets={s:vector_set(d[s][0]) for s in ('train','validation','test')}
        obs={a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
        assert obs==expected[name]['overlap'],(name,obs)
        overlap[name]=obs
    groups={};masks={};reference=np.load(ROOT/'results/reference_predictions.npz');reference_count=0
    methods={('fedavg','full'):'FedAvg-full',('dca','prevalence'):'Prevalence-FCM-DCA',('fedavg','rasc'):'RASC-FedAvg',('dca','rasc'):'RASC-FCM-DCA'}
    for r in records:
        c=r['config'];d=datasets[c['dataset']];mask=np.asarray(r['mask'],bool)
        budget=c.get('budget',22 if c['dataset'].startswith('bot') else round(.6*d['train'][0].shape[1]))
        assert mask.sum()==(len(mask) if c['mask']=='full' else budget)
        best=-np.inf;best_round=0
        for entry in r['round_log']:
            if entry['validation_macro_f1']>best+1e-12:best=entry['validation_macro_f1'];best_round=entry['round']
        assert best_round==r['best_round'] and best==r['validation_macro_f1']
        ids=partition(d['train'][1],c.get('partition_seed',c['seed']),c.get('clients',5),c.get('alpha',.5))
        assert hashlib.sha256(b''.join(x.tobytes() for x in ids)).hexdigest()==r['client_partition_sha256']
        with np.load(ROOT/'results/runs'/f"{r['run_id']}.npz") as a:
            for k,x in enumerate(ids):np.testing.assert_array_equal(x,a[f'client_{k}'])
            key=(c['dataset'],c['seed'],c.get('alpha',.5),c.get('clients',5),budget)
            value=r['client_partition_sha256']
            assert groups.setdefault(key,value)==value
            mkey=key+(c['mask'],)
            assert masks.setdefault(mkey,r['mask'])==r['mask']
            if c['dataset']=='bot_primary' and c.get('rounds',40)==40 and c.get('alpha',.5)==.5 and c.get('clients',5)==5 and (c['algorithm'],c['mask']) in methods:
                name=methods[c['algorithm'],c['mask']]+'_'+str(c['seed'])
                np.testing.assert_array_equal(a['probabilities'],reference[name])
                np.testing.assert_array_equal(d['test'][1],reference['y_true'])
                np.testing.assert_array_equal(d['test'][2],reference['test_global_index'])
                reference_count+=1
    assert reference_count==40
    report={'status':'passed','date_utc':datetime.now(timezone.utc).date().isoformat(),'test_runs':len(records),'tuning_runs':36,'unique_partition_conditions':len(groups),'all_partitions_rebuilt_exactly':True,'all_masks_paired_across_optimizers':True,'all_validation_checkpoints_obey_earliest_maximum_rule':True,'reference_matrices_exact':reference_count,'cohort_overlap_recomputed':overlap,'per_dataset_runs':dict(collections.Counter(r['config']['dataset'] for r in records))}
    (ROOT/'results/final_contract_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
