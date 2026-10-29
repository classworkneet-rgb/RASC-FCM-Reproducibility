"""Recompute secondary inference from frozen paired outcomes, without training."""
import json
import numpy as np
from experiment import ROOT,SEEDS
from summarize import collect,contrast,holm

def main():
 rr=collect();primary=[r for r in rr if r['config']['dataset']=='bot_primary' and r['config']['rounds']==40 and r['config']['clients']==5 and r['config']['alpha']==.5]
 def values(a,m):
  d={r['config']['seed']:r['test']['macro_f1'] for r in primary if r['config']['algorithm']==a and r['config']['mask']==m}
  return [d[s] for s in SEEDS]
 positive=contrast(values('fedavg','rasc'),values('fedavg','prevalence'))
 frozen=json.loads((ROOT/'results/paired_comparisons.json').read_text())
 for k in positive:assert positive[k]==frozen['rasc_minus_prevalence_fedavg'][k],k
 names=['rasc_minus_prevalence_fedavg']+['rasc_minus_full_'+a for a in ['fedprox','scaffold','fedadam','fedyogi']]
 adjusted=holm([frozen[k]['exact_sign_flip_p'] for k in names])
 proximal={}
 for m in ['full','rasc']:
  delta=np.array(values('fedprox',m))-np.array(values('fedavg',m))
  proximal[m]={'mean_difference':float(delta.mean()),'max_absolute_seed_difference':float(np.abs(delta).max()),'identical_macro_f1_pairs':int((delta==0).sum()),'paired_differences':delta.tolist()}
 report={'analysis_date':'2026-10-27','scope':'secondary selector inference and exploratory five-comparison multiplicity sensitivity; no change to the original four-comparison family','positive_selector_contrast':positive,'exploratory_five_contrast_holm':dict(zip(names,adjusted)),'fedprox_vs_fedavg':proximal,'bootstrap_replicates':20000,'bootstrap_seed':97031,'exact_sign_assignments':1024,'primary_seeds':SEEDS}
 (ROOT/'results/additional_inference.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
