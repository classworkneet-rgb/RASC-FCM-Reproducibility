"""Replay new checkpoints and check selector inputs, masks, metrics and pairing."""
from __future__ import annotations
import argparse, hashlib, json
import numpy as np
from experiment import ROOT, SEEDS, read_data, partition, metrics, run, key
from model import ArchitectureSpec, FixedGatedDBNCNN
from selector_extension import configurations, digest, frozen_masks, selector_weights

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrain-interface',action='store_true',help='Also reproduce the first reference RASC trajectory for both optimizers')
    args=parser.parse_args()
    data=read_data('bot_primary');x,y,_=data['train']
    masks_path=ROOT/'configs/selector_extension_masks.json'
    obj=json.loads(masks_path.read_text())
    protocol_path=ROOT/'configs/selector_extension_protocol.json'
    assert obj['protocol_sha256']==hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    protocol=json.loads(protocol_path.read_text())
    assert hashlib.sha256((ROOT/'data/bot_primary.npz').read_bytes()).hexdigest()==protocol['input_sha256']['bot_primary.npz']
    for record in obj['records']:
        clients=partition(y,record['seed'],5,.5)
        assert hashlib.sha256(b''.join(v.tobytes() for v in clients)).hexdigest()==record['partition_sha256']
        assert all(sum(m)==22 and len(m)==37 for m in record['masks'].values())
    records=[]
    for cfg in configurations():
        rid=key(cfg);r=json.loads((ROOT/'results/selector_runs'/f'{rid}.json').read_text())
        assert r['config']==cfg and r['run_id']==rid
        assert r['selector_protocol_sha256']==hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        assert r['selector_masks_sha256']==hashlib.sha256(masks_path.read_bytes()).hexdigest()
        record=next(a for a in obj['records'] if a['seed']==cfg['seed'])
        assert r['mask']==record['masks'][cfg['mask']]
        assert r['selector_class_weights']==selector_weights(cfg['mask'],record)
        assert r['client_partition_sha256']==record['partition_sha256']
        assert r['best_round']==max(r['round_log'],key=lambda a:a['validation_macro_f1'])['round']
        with np.load(ROOT/'selector_checkpoints'/f'{rid}.npz') as a:parameters=a['parameters']
        assert digest(parameters)==r['parameters_sha256']
        if r['mask']==record['masks']['rasc']:
            reference_id=record['reference_run_ids'][cfg['algorithm']]
            with np.load(ROOT/'checkpoints'/f'{reference_id}.npz') as a:
                np.testing.assert_array_equal(parameters,a['parameters'])
            reference=json.loads((ROOT/'results/runs'/f'{reference_id}.json').read_text())
            assert r['round_log']==reference['round_log']
        model=FixedGatedDBNCNN(ArchitectureSpec(37,5),seed=0)
        model.load_parameter_vector(parameters)
        p=model.predict_proba(data['test'][0],np.asarray(r['mask'],bool))
        assert digest(p.astype(np.float32))==r['probabilities_sha256']
        actual=metrics(data['test'][1],p)
        for name in ['macro_f1','accuracy','balanced_accuracy','weighted_f1','mcc','ece','macro_pr_auc','macro_roc_auc']:
            assert abs(actual[name]-r['test'][name])<1e-12,(rid,name)
        assert r['test']['support']==[10000,10000,10000,96,16]
        records.append(rid)
    report={'status':'passed','verified_new_runs':len(records),
        'all_parameter_and_probability_hashes_match':True,'all_metrics_recomputed':True,
        'all_frozen_masks_and_paired_partitions_match':True,
        'all_validation_checkpoints_obey_earliest_maximum_rule':True,
        'test_support':[10000,10000,10000,96,16],'run_ids':records}
    (ROOT/'results/selector_extension_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified all 100 selector checkpoints, probability hashes, metrics, masks and partitions.')
    if args.retrain_interface:
        original=[json.loads(p.read_text()) for p in (ROOT/'results/runs').glob('*.json')]
        checks=[]
        for algorithm in ['fedavg','fedadam']:
            ref=next(r for r in original if r['config']['dataset']=='bot_primary'
                and r['config']['seed']==SEEDS[0] and r['config']['algorithm']==algorithm
                and r['config']['mask']=='rasc' and r['config']['rounds']==40
                and r['config']['clients']==5 and r['config']['alpha']==.5)
            current,_,_,vector=run(ref['config'],mask_builder=frozen_masks)
            with np.load(ROOT/'checkpoints'/f"{ref['run_id']}.npz") as checkpoint:
                np.testing.assert_array_equal(vector,checkpoint['parameters'])
            assert current['round_log']==ref['round_log']
            assert current['test']==ref['test']
            checks.append({'algorithm':algorithm,'run_id':ref['run_id'],
                'selected_parameters_sha256':digest(vector),'selected_parameters_exact':True,
                'validation_trajectory_exact':True,'test_metrics_exact':True})
        (ROOT/'results/selector_interface_verification.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2)+'\n')
        print('Exact reference retraining passed for both optimizers.')

if __name__=='__main__':main()
