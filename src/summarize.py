"""Derive tables and exact paired-seed statistics directly from run records."""
from pathlib import Path
import itertools,json,csv
import numpy as np,pandas as pd
from experiment import ROOT,SEEDS

def exact_sign_flip(diff):
    d=np.asarray(diff,float);signs=np.asarray(list(itertools.product([-1.,1.],repeat=len(d))))
    return float(np.mean(np.abs(signs@d/len(d))>=abs(d.mean())-1e-15))

def contrast(a,b):
    d=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(97031)
    means=np.mean(d[rng.integers(0,len(d),size=(20000,len(d)))],axis=1)
    return {'n':len(d),'mean_difference':float(d.mean()),'sample_sd':float(d.std(ddof=1)),'ci95_low':float(np.quantile(means,.025)),
      'ci95_high':float(np.quantile(means,.975)),'exact_sign_flip_p':exact_sign_flip(d),'paired_differences':d.tolist()}

def holm(values):
    order=np.argsort(values,kind='stable');out=np.ones(len(values));maximum=0.
    for i,j in enumerate(order):maximum=max(maximum,(len(values)-i)*values[j]);out[j]=min(1.,maximum)
    return out.tolist()

def collect():
    runs=[json.loads(p.read_text()) for p in (ROOT/'results/runs').glob('*.json')]
    for r in runs:
        if 'test' not in r:raise RuntimeError('run missing test results')
    return runs

def main():
    runs=collect();rows=[]
    keys=['accuracy','balanced_accuracy','macro_f1','weighted_f1','macro_pr_auc','macro_roc_auc','mcc','ece']
    for r in runs:
        c=r['config'];row={**c,'run_id':r['run_id'],'runtime_seconds':r['runtime_seconds'],'best_round':r['best_round'],
          'parameter_count':r['parameter_count'],'selected_features':r['selected_features'],**{k:r['test'][k] for k in keys}}
        rows.append(row)
    frame=pd.DataFrame(rows).sort_values(['dataset','rounds','clients','alpha','algorithm','mask','seed'])
    frame.to_csv(ROOT/'results/all_runs.csv',index=False)
    groups=['dataset','rounds','clients','alpha','algorithm','mask'];summary=[]
    for values,part in frame.groupby(groups):
        row=dict(zip(groups,values));row['n']=len(part)
        for k in keys+['runtime_seconds','best_round']:
            row[k+'_mean']=float(part[k].mean());row[k+'_sd']=float(part[k].std(ddof=1))
        summary.append(row)
    pd.DataFrame(summary).to_csv(ROOT/'results/summary.csv',index=False)
    primary=[r for r in runs if r['config']['dataset']=='bot_primary' and r['config']['clients']==5 and r['config']['alpha']==.5 and r['config']['rounds']==40]
    def values(algo,mask,metric='macro_f1'):
        rr={r['config']['seed']:r['test'][metric] for r in primary if r['config']['algorithm']==algo and r['config']['mask']==mask}
        return [rr[s] for s in SEEDS]
    comparisons={}
    for algo in ['fedavg','fedprox','scaffold','fedadam','fedyogi']:
        comparisons['rasc_minus_full_'+algo]=contrast(values(algo,'rasc'),values(algo,'full'))
    comparisons['rasc_minus_prevalence_fedavg']=contrast(values('fedavg','rasc'),values('fedavg','prevalence'))
    comparisons['rasc_minus_prevalence_dca']=contrast(values('dca','rasc'),values('dca','prevalence'))
    comparisons['dca_minus_fedavg_rasc']=contrast(values('dca','rasc'),values('fedavg','rasc'))
    comps=['rasc_minus_full_'+a for a in ['fedprox','scaffold','fedadam','fedyogi']]
    adj=holm([comparisons[k]['exact_sign_flip_p'] for k in comps])
    for k,p in zip(comps,adj):comparisons[k]['holm_p_extension_family']=p
    (ROOT/'results/paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
    # Sensitivities and the smaller external confirmation are descriptive.
    report={'completed_test_runs':len(runs),'completed_tuning_runs':len(list((ROOT/'results/tuning').glob('*.json'))),
      'primary':{a+'_'+m:{'mean':float(np.mean(values(a,m))),'sd':float(np.std(values(a,m),ddof=1))} for a,m in [('fedavg','full'),('fedavg','prevalence'),('fedavg','rasc'),('fedprox','full'),('fedprox','rasc'),('scaffold','full'),('scaffold','rasc'),('fedadam','full'),('fedadam','rasc'),('fedyogi','full'),('fedyogi','rasc')]},
      'paired_comparisons':comparisons}
    duration=[]
    for algo in ['fedavg','fedadam']:
        long={mask:{r['config']['seed']:r['test']['macro_f1'] for r in runs if
            r['config']['dataset']=='bot_primary' and r['config']['rounds']==200 and
            r['config']['algorithm']==algo and r['config']['mask']==mask} for mask in ['full','rasc']}
        if all(len(long[m])==10 for m in long):
            difference=np.asarray([long['rasc'][s]-long['full'][s] for s in SEEDS])
            short_difference=np.asarray(values(algo,'rasc'))-np.asarray(values(algo,'full'))
            duration.append({'algorithm':algo,'scope':'exploratory follow-up after the initial three duration outcomes',
                'long_rasc_minus_full':contrast([long['rasc'][s] for s in SEEDS],[long['full'][s] for s in SEEDS]),
                'mask_by_duration_interaction':contrast(difference,short_difference),
                'initial_three_mean_difference':float(difference[:3].mean()),
                'remaining_seven_mean_difference':float(difference[3:].mean()),
                'full_mean':float(np.mean(list(long['full'].values()))),
                'rasc_mean':float(np.mean(list(long['rasc'].values())))})
    report['duration_followup']=duration
    (ROOT/'results/duration_followup.json').write_text(json.dumps(duration,indent=2))
    (ROOT/'results/release_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
