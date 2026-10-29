"""Matched federated optimizers and frozen evaluation; no network services.

The NumPy classifier and common-mask selector are shared across methods. FedProx uses
the proximal gradient evaluated BEFORE each SGD step. SCAFFOLD implements
Option II with persistent client controls and the canonical uniform server
mean. FedAdam/FedYogi use uncorrected moments of model deltas, as in FedOpt.
"""
from __future__ import annotations
import argparse, json, time, hashlib, platform, sys, os
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, precision_recall_fscore_support, confusion_matrix,
    average_precision_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from model import ArchitectureSpec, FixedGatedDBNCNN
from aggregation import ClientUpdate, DriftConfig, aggregate_updates, sample_weighted_fedavg_delta
from rare_class_consensus import client_summary, aggregate_prevalence_mask, aggregate_rare_class_mask

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[104729,130363,155921,196613,216091,262147,300007,350377,399989,448451]
BOT_CLASSES=['DDoS','DoS','Reconnaissance','Normal','Theft']

def read_data(name):
    d=np.load(ROOT/'data'/f'{name}.npz')
    return {s:tuple(d[f'{s}_{k}'] for k in ['x','y','ids']) for s in ['train','validation','test']}

def partition(y,seed,clients=5,alpha=.5):
    rng=np.random.default_rng(seed);buckets=[[] for _ in range(clients)]
    for c in range(int(y.max())+1):
        ids=np.flatnonzero(y==c);rng.shuffle(ids)
        counts=rng.multinomial(len(ids),rng.dirichlet(np.full(clients,alpha)))
        cursor=0
        for k,n in enumerate(counts):buckets[k].extend(ids[cursor:cursor+n].tolist());cursor+=n
    result=[]
    for ids in buckets:
        if not ids:raise RuntimeError('empty client; no silent redraw')
        a=np.asarray(ids,dtype=np.int64);rng.shuffle(a);result.append(a)
    assert len(np.unique(np.concatenate(result)))==len(y)
    return result

def prepare_masks(x,y,client_ids,seed,budget):
    summaries=[client_summary(x[ids],y[ids],classes=int(y.max())+1,
        candidate_budget=budget,bootstraps=8,seed=seed+1009*k,support_cap=256)
        for k,ids in enumerate(client_ids)]
    prev=aggregate_prevalence_mask(summaries,feature_budget=budget)
    rasc,diag=aggregate_rare_class_mask(summaries,feature_budget=budget,reserve_per_class=1,
        beta=.75,rare_weight_cap=4.,minimum_global_support=8)
    return {'full':np.ones(x.shape[1],bool),'prevalence':prev,'rasc':rasc},diag

def clone(model):
    m=FixedGatedDBNCNN(model.spec,seed=0);m.load_parameter_vector(model.parameter_vector());return m

def pretrain(model,x,client_ids,mask,seed):
    initial=model.parameter_vector().copy();updates=[]
    for k,ids in enumerate(client_ids):
        m=clone(model);chosen=ids if len(ids)<=2500 else ids[np.linspace(0,len(ids)-1,2500,dtype=int)]
        m.rbm_pretrain_first_layer(x[chosen],mask,epochs=1,learning_rate=.003,seed=seed+k)
        updates.append(ClientUpdate(m.parameter_vector()-initial,len(ids)))
    model.load_parameter_vector(initial+sample_weighted_fedavg_delta(updates))

def metrics(y,prob):
    c=prob.shape[1];pred=prob.argmax(1)
    p,r,f,n=precision_recall_fscore_support(y,pred,labels=np.arange(c),zero_division=0)
    bi=(np.arange(c)[None,:]==y[:,None]).astype(int)
    confidence=prob.max(1);ece=0.
    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        keep=(confidence>lo)&(confidence<=hi)
        if keep.any():ece+=keep.mean()*abs((pred[keep]==y[keep]).mean()-confidence[keep].mean())
    return {'accuracy':float(accuracy_score(y,pred)),
        'balanced_accuracy':float(r.mean()),'macro_f1':float(f.mean()),
        'weighted_f1':float(f1_score(y,pred,average='weighted',zero_division=0)),
        'macro_pr_auc':float(average_precision_score(bi,prob,average='macro')),
        'macro_roc_auc':float(roc_auc_score(bi,prob,average='macro',multi_class='ovr')),
        'mcc':float(matthews_corrcoef(y,pred)),'ece':float(ece),
        'per_class_recall':r.tolist(),'per_class_f1':f.tolist(),
        'support':n.tolist(),'confusion_matrix':confusion_matrix(y,pred,labels=np.arange(c)).tolist()}

def adaptive_update(delta,m,v,kind,server_lr,beta1=.9,beta2=.99,tau=.001):
    m=beta1*m+(1-beta1)*delta
    if kind=='fedadam':v=beta2*v+(1-beta2)*delta**2
    elif kind=='fedyogi':v=v-(1-beta2)*delta**2*np.sign(v-delta**2)
    else:raise ValueError(kind)
    return server_lr*m/(np.sqrt(v)+tau),m,v

def local_step(model,x,y,mask,eta,current,algo,mu,c_client,c_server):
    before=model.parameter_vector().copy() if algo=='fedprox' else None
    model.train_batch(x,y,mask,learning_rate=eta,class_weights=np.ones(model.spec.class_count),weight_decay=1e-5)
    if algo=='fedprox':model.load_parameter_vector(model.parameter_vector()-eta*mu*(before-current))
    if algo=='scaffold':model.load_parameter_vector(model.parameter_vector()+eta*(c_client-c_server))

def run(cfg,test=True,*,mask_builder=None):
    data=read_data(cfg['dataset']);x,y,_=data['train'];xv,yv,_=data['validation']
    seed=cfg['seed'];clients=partition(y,cfg.get('partition_seed',seed),cfg.get('clients',5),cfg.get('alpha',.5))
    budget=cfg.get('budget',22 if cfg['dataset'].startswith('bot') else round(.6*x.shape[1]))
    ms,diag=(mask_builder or prepare_masks)(x,y,clients,seed,budget);mask=ms[cfg['mask']]
    c=int(y.max())+1;model=FixedGatedDBNCNN(ArchitectureSpec(x.shape[1],c),seed=seed)
    start=time.perf_counter();pretrain(model,x,clients,mask,seed)
    npar=model.parameter_count();previous=np.zeros(npar);controls=np.zeros((len(clients),npar));control=np.zeros(npar)
    m=np.zeros(npar);v=np.zeros(npar);eta=cfg.get('local_lr',.003);steps=cfg.get('steps',6);batch=cfg.get('batch',768)
    algo=cfg['algorithm'];best=-np.inf;best_round=0;best_vector=model.parameter_vector().copy();logs=[]
    weights=np.asarray([len(i) for i in clients],float);weights/=weights.sum()
    ps=[]
    for ids in clients:
        counts=np.bincount(y[ids],minlength=c).astype(float);s=1/np.maximum(counts[y[ids]],1);ps.append(s/s.sum())
    for t in range(1,cfg.get('rounds',40)+1):
        current=model.parameter_vector().copy();updates=[];next_controls=[];loss=[]
        for k,ids in enumerate(clients):
            local=clone(model);rng=np.random.default_rng(seed+100003*t+1009*k)
            for _ in range(steps):
                sampled=rng.choice(ids,size=batch,replace=True,p=ps[k])
                local_step(local,x[sampled],y[sampled],mask,eta,current,algo,cfg.get('mu',.1),controls[k],control)
            updates.append(ClientUpdate(local.parameter_vector()-current,len(ids)))
            if algo=='scaffold':next_controls.append(controls[k]-control+(current-local.parameter_vector())/(steps*eta))
        reliability=1.;deviation=0.
        if algo=='dca':
            agg=aggregate_updates(current,updates,previous,DriftConfig(update_clip_norm=3.,norm_drift_scale=.25,
                drift_cap=4.,drift_decay=.75,fairness_floor=.25,sample_cap=10000))
            new=agg.new_parameters;reliability=float(np.mean([d.reliability for d in agg.diagnostics]))
            deviation=float(np.max(np.abs(agg.normalized_weights-weights)))
        elif algo=='scaffold':
            # Canonical SCAFFOLD uses uniform client weighting, not n_k/n.
            new=current+cfg.get('server_lr',1.)*np.mean([u.delta for u in updates],axis=0)
            controls=np.stack(next_controls);control=controls.mean(axis=0)
        else:
            delta=sample_weighted_fedavg_delta(updates)
            if algo in ['fedadam','fedyogi']:
                update,m,v=adaptive_update(delta,m,v,algo,cfg.get('server_lr',.01));new=current+update
            else:new=current+delta
        if not np.isfinite(new).all():raise FloatingPointError('nonfinite model')
        model.load_parameter_vector(new);previous=new-current
        prob=model.predict_proba(xv,mask);val=float(f1_score(yv,prob.argmax(1),labels=np.arange(c),average='macro',zero_division=0))
        logs.append({'round':t,'validation_macro_f1':val,'mean_reliability':reliability,'max_weight_deviation':deviation})
        if val>best+1e-12:best=val;best_round=t;best_vector=new.copy()
    model.load_parameter_vector(best_vector)
    result={'config':cfg,'best_round':best_round,'validation_macro_f1':best,'parameter_count':npar,
        'runtime_seconds':time.perf_counter()-start,'mask':mask.astype(int).tolist(),'round_log':logs,
        'client_class_counts':[np.bincount(y[ids],minlength=c).tolist() for ids in clients],
        'class_weights':diag['class_weights'].tolist(),
        'client_partition_sha256':hashlib.sha256(b''.join(i.tobytes() for i in clients)).hexdigest(),
        'training_rows':len(y),'selected_features':int(mask.sum())}
    prob=None
    if test:
        prob=model.predict_proba(data['test'][0],mask)
        result['test']=metrics(data['test'][1],prob)
    return result,prob,clients,best_vector

def key(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:16]

def execute(cfg,test=True):
    folder=ROOT/'results'/('runs' if test else 'tuning');folder.mkdir(parents=True,exist_ok=True)
    rid=key(cfg);out=folder/f'{rid}.json'
    if out.exists() and (not test or out.with_suffix('.npz').exists()):return json.loads(out.read_text())
    result,prob,clients,vector=run(cfg,test=test);result['run_id']=rid
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(out)
    if test:
        np.savez_compressed(folder/f'{rid}.npz',probabilities=prob.astype(np.float32),parameters=vector,
            **{f'client_{k}':ids for k,ids in enumerate(clients)})
    print(rid,cfg['dataset'],cfg['algorithm'],cfg['mask'],cfg['seed'],cfg.get('alpha',.5),
        'val',round(result['validation_macro_f1'],4),'test',round(result.get('test',{}).get('macro_f1',0),4),
        round(result['runtime_seconds'],1),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--validation-only',action='store_true');a=p.parse_args()
    execute(json.loads(Path(a.config).read_text()),not a.validation_only)
