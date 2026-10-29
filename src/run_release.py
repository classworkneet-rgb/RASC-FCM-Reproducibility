"""Run validation-only optimizer tuning, freeze its winners, then evaluate.

Each method-seed result is atomically checkpointed. Resume by rerunning the
same command; no completed configuration is silently overwritten.
"""
from pathlib import Path
import argparse,itertools,json,os
from concurrent.futures import ProcessPoolExecutor
from experiment import ROOT,SEEDS,execute,read_data,partition

def baseline(algorithm,mask,seed,dataset='bot_primary',alpha=.5,clients=5,rounds=40):
    return dict(dataset=dataset,algorithm=algorithm,mask=mask,seed=seed,alpha=alpha,clients=clients,rounds=rounds)

def job(item):
    cfg,validation=item;return execute(cfg,test=not validation)

def batch(jobs,workers):
    with ProcessPoolExecutor(workers) as pool:return list(pool.map(job,jobs))

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tune','primary','sensitivity','external','expanded','extended','all'],default='all');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    protocol=json.loads((ROOT/'configs/protocol_freeze.json').read_text())
    phases=['tune','primary','sensitivity','external','expanded','extended'] if a.phase=='all' else [a.phase]
    selected_path=ROOT/'configs/selected_optimizer_settings.json'
    for phase in phases:
        if phase=='tune':
            candidates=[]
            for algo,field,values in [('fedprox','mu',[.01,.1,1.]),('scaffold','server_lr',[.3,1.,3.]),('fedadam','server_lr',[.001,.01,.1]),('fedyogi','server_lr',[.001,.01,.1])]:
                for val in values:
                    for seed in protocol['tuning_seeds']:
                        cfg=baseline(algo,'full',seed);cfg[field]=val;candidates.append((cfg,True))
            results=batch(candidates,a.workers);selected={};audit=[]
            for algo,field,values in [('fedprox','mu',[.01,.1,1.]),('scaffold','server_lr',[.3,1.,3.]),('fedadam','server_lr',[.001,.01,.1]),('fedyogi','server_lr',[.001,.01,.1])]:
                means=[]
                for value in values:
                    rows=[r for r in results if r['config']['algorithm']==algo and r['config'][field]==value]
                    score=sum(r['validation_macro_f1'] for r in rows)/len(rows);means.append(score)
                    audit.append({'algorithm':algo,'parameter':field,'value':value,'mean_validation_macro_f1':score,'runs':[r['run_id'] for r in rows]})
                winner=max(range(len(values)),key=lambda i:means[i]);selected[algo]={field:values[winner]}
            selected_path.write_text(json.dumps({'selected':selected,'validation_grid':audit},indent=2));print('TUNING FROZEN',selected,flush=True);continue
        selected=json.loads(selected_path.read_text())['selected']
        jobs=[]
        if phase=='primary':
            for seed in SEEDS:
                for algo in ['fedavg','dca','fedprox','scaffold','fedadam','fedyogi']:
                    masks=['full','rasc'] if algo!='dca' else ['prevalence','rasc']
                    if algo=='fedavg':masks+=['prevalence']
                    for mask in masks:
                        cfg=baseline(algo,mask,seed);cfg.update(selected.get(algo,{}));jobs.append((cfg,False))
        if phase=='sensitivity':
            settings=[(alpha,5) for alpha in [.1,.3,1.]]+[(.5,k) for k in [10,20]]
            for alpha,k in settings:
                for seed in SEEDS[:3]:
                    for algo,mask in [('fedavg','full'),('fedavg','prevalence'),('fedavg','rasc'),('fedprox','full'),('scaffold','full'),('fedadam','full')]:
                        cfg=baseline(algo,mask,seed,alpha=alpha,clients=k);cfg.update(selected.get(algo,{}))
                        y=read_data('bot_primary')['train'][1]
                        for attempt in range(101):
                            partition_seed=seed+1000003*attempt
                            try:partition(y,partition_seed,k,alpha);break
                            except RuntimeError:continue
                        else:raise RuntimeError('no nonempty partition in 101 attempts')
                        if attempt:cfg.update(partition_seed=partition_seed,partition_rejections=attempt)
                        jobs.append((cfg,False))
        if phase=='external':
            for seed in SEEDS[:5]:
                for algo,mask in [('fedavg','full'),('fedavg','prevalence'),('fedavg','rasc'),('fedprox','full'),('scaffold','full'),('fedadam','full'),('fedprox','rasc'),('scaffold','rasc'),('fedadam','rasc')]:
                    cfg=baseline(algo,mask,seed,dataset='cic_primary');cfg.update(selected.get(algo,{}));jobs.append((cfg,False))
        if phase=='expanded':
            for seed in SEEDS[:3]:
                for algo,mask in [('fedavg','full'),('fedavg','prevalence'),('fedavg','rasc'),('scaffold','full'),('fedadam','full')]:
                    cfg=baseline(algo,mask,seed,dataset='bot_expanded');cfg.update(selected.get(algo,{}));jobs.append((cfg,False))
        if phase=='extended':
            for seed in SEEDS:
                for algo,mask in [('fedavg','full'),('fedavg','rasc'),('fedadam','full'),('fedadam','rasc')]:
                    cfg=baseline(algo,mask,seed,rounds=200);cfg.update(selected.get(algo,{}));jobs.append((cfg,False))
        print('PHASE',phase,'jobs',len(jobs),flush=True);batch(jobs,a.workers)

if __name__=='__main__':main()
