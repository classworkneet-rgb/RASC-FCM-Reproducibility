"""Fixed-budget selector controls and component omissions for the BoT-IoT study.

The design in configs/selector_extension_protocol.json was written before the
additional outcomes were computed. Original benchmark outcomes were already
known, so every new inference is exploratory.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from experiment import ROOT, SEEDS, run, key, partition, read_data
from rare_class_consensus import client_summary, aggregate_rare_class_mask

NAMES = ['no_stability','uniform_class_weights','no_reservation',
         'mutual_information','random']

def selector_weights(mask_name,record):
    if mask_name in ['mutual_information','random']:return None
    if mask_name=='uniform_class_weights':
        support=np.asarray(record['diagnostics']['class_support'])
        eligible=(support>=8).astype(float)
        return (eligible/eligible.sum()).tolist()
    return record['diagnostics']['class_weights']

def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def top_mask(scores, budget):
    mask=np.zeros(len(scores),dtype=bool)
    mask[np.argsort(-scores,kind='stable')[:budget]]=True
    return mask

def build_masks(x,y,clients,seed,budget):
    summaries=[client_summary(x[ids],y[ids],classes=int(y.max())+1,
        candidate_budget=budget,bootstraps=8,seed=seed+1009*k,support_cap=256)
        for k,ids in enumerate(clients)]
    common=dict(feature_budget=budget,reserve_per_class=1,
                rare_weight_cap=4.,minimum_global_support=8)
    original,diag=aggregate_rare_class_mask(summaries,beta=.75,**common)
    no_stability,_=aggregate_rare_class_mask(summaries,beta=1.,**common)
    uniform,_=aggregate_rare_class_mask(summaries,feature_budget=budget,
        reserve_per_class=1,beta=.75,rare_weight_cap=1.,minimum_global_support=8)
    no_reservation=top_mask(diag['composite'],budget)
    weighted_mi=np.zeros(x.shape[1],float)
    local_mi=[]
    for k,ids in enumerate(clients):
        score=np.zeros(x.shape[1],float)
        if len(np.unique(y[ids]))>1:
            score=mutual_info_classif(x[ids],y[ids],discrete_features=False,
                n_neighbors=3,random_state=seed+2003*k)
            score[np.std(x[ids],axis=0)<=1e-12]=0.
        local_mi.append(score)
        weighted_mi+=len(ids)/len(y)*score
    random=np.zeros(x.shape[1],bool)
    random[np.random.default_rng(seed+1618033).choice(x.shape[1],budget,replace=False)]=True
    masks=dict(rasc=original,no_stability=no_stability,
        uniform_class_weights=uniform,no_reservation=no_reservation,
        mutual_information=top_mask(weighted_mi,budget),random=random)
    assert all(int(v.sum())==budget for v in masks.values())
    return masks,diag,{'mutual_information_scores':weighted_mi.tolist(),
        'client_mutual_information_scores':[v.tolist() for v in local_mi],
        'class_support':diag['class_support'].tolist(),
        'class_weights':diag['class_weights'].tolist(),
        'stability':diag['stability'].tolist(),
        'class_scores':diag['class_scores'].tolist(),
        'composite':diag['composite'].tolist()}

def freeze_masks():
    protocol_path=ROOT/'configs/selector_extension_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    x,y,_=read_data('bot_primary')['train']
    expected=protocol['input_sha256']['bot_primary.npz']
    assert hashlib.sha256((ROOT/'data/bot_primary.npz').read_bytes()).hexdigest()==expected
    prior=[json.loads(p.read_text()) for p in (ROOT/'results/runs').glob('*.json')]
    rows=[]
    for seed in SEEDS:
        clients=partition(y,seed,5,.5)
        masks,diag,details=build_masks(x,y,clients,seed,22)
        references=[r for r in prior if r['config']['dataset']=='bot_primary'
            and r['config']['seed']==seed and r['config']['mask']=='rasc'
            and r['config']['algorithm'] in ['fedavg','fedadam']
            and r['config']['rounds']==40 and r['config']['clients']==5
            and r['config']['alpha']==.5]
        assert len(references)==2
        partition_hash=hashlib.sha256(b''.join(i.tobytes() for i in clients)).hexdigest()
        for reference in references:
            np.testing.assert_array_equal(masks['rasc'].astype(int),reference['mask'])
            assert reference['client_partition_sha256']==partition_hash
        rows.append({'seed':seed,'partition_sha256':partition_hash,
            'masks':{name:mask.astype(int).tolist() for name,mask in masks.items()},
            'diagnostics':details,
            'reference_run_ids':{r['config']['algorithm']:r['run_id'] for r in references}})
    out=ROOT/'configs/selector_extension_masks.json'
    encoded=json.dumps({'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'records':rows},indent=2)+'\n'
    if out.exists():
        assert out.read_text()==encoded, 'Frozen selector masks changed'
    else:out.write_text(encoded)
    print('All 10 RASC masks and paired partition digests match the original records.',flush=True)

def frozen_masks(x,y,clients,seed,budget):
    assert x.shape==(30388,37) and budget==22
    obj=json.loads((ROOT/'configs/selector_extension_masks.json').read_text())
    record=next(r for r in obj['records'] if r['seed']==seed)
    assert hashlib.sha256(b''.join(i.tobytes() for i in clients)).hexdigest()==record['partition_sha256']
    masks={k:np.asarray(v,bool) for k,v in record['masks'].items()}
    return masks,{'class_weights':np.asarray(record['diagnostics']['class_weights'])}

def execute_one(cfg):
    folder=ROOT/'results/selector_runs';folder.mkdir(parents=True,exist_ok=True)
    checkpoint_folder=ROOT/'selector_checkpoints';checkpoint_folder.mkdir(exist_ok=True)
    rid=key(cfg);out=folder/f'{rid}.json'
    if out.exists():
        assert (checkpoint_folder/f'{rid}.npz').exists()
        return json.loads(out.read_text())
    result,prob,clients,vector=run(cfg,test=True,mask_builder=frozen_masks)
    result['run_id']=rid
    result['probabilities_sha256']=digest(prob.astype(np.float32))
    result['parameters_sha256']=digest(vector)
    result['selector_protocol_sha256']=hashlib.sha256((ROOT/'configs/selector_extension_protocol.json').read_bytes()).hexdigest()
    result['selector_masks_sha256']=hashlib.sha256((ROOT/'configs/selector_extension_masks.json').read_bytes()).hexdigest()
    result['selector_diagnostics_note']='Reported class weights describe the original consensus summary; local training weights and balanced sampling are unchanged for every selector.'
    mask_record=next(r for r in json.loads((ROOT/'configs/selector_extension_masks.json').read_text())['records'] if r['seed']==cfg['seed'])
    result['selector_class_weights']=selector_weights(cfg['mask'],mask_record)
    np.savez_compressed(checkpoint_folder/f'{rid}.npz',parameters=vector)
    temp=out.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out)
    print(rid,cfg['algorithm'],cfg['mask'],cfg['seed'],
          'macro_f1',round(result['test']['macro_f1'],6),
          'seconds',round(result['runtime_seconds'],1),flush=True)
    return result

def configurations():
    protocol=json.loads((ROOT/'configs/selector_extension_protocol.json').read_text())
    assert protocol['planned_new_test_runs']==100
    return [dict(dataset='bot_primary',algorithm=algorithm,mask=mask,seed=seed,
                 alpha=.5,clients=5,rounds=40,study='selector_components',**options)
            for seed in SEEDS for algorithm,options in protocol['algorithms'].items()
            for mask in NAMES]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-masks',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.freeze_masks:freeze_masks();return
    assert (ROOT/'configs/selector_extension_masks.json').exists(), 'Freeze masks before training'
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(execute_one,configurations()))
    print('Completed all 100 fixed-design selector runs.',flush=True)

if __name__=='__main__':main()
