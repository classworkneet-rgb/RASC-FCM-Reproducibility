"""Recompute the ten exploratory selector contrasts and their common Holm family."""
from __future__ import annotations
import csv, hashlib, json
import numpy as np
from experiment import ROOT, SEEDS
from selector_extension import NAMES, configurations, digest
from summarize import contrast, holm

LABELS={
 'full':'All features','prevalence':'Prevalence consensus','rasc':'RASC-FCM',
 'no_stability':'Without stability','uniform_class_weights':'Uniform class weights',
 'no_reservation':'Without reserved slots','mutual_information':'Mutual information',
 'random':'Random mask'}

def selected_reference():
    records=[json.loads(p.read_text()) for p in (ROOT/'results/runs').glob('*.json')]
    return [r for r in records if r['config']['dataset']=='bot_primary'
        and r['config']['rounds']==40 and r['config']['clients']==5
        and r['config']['alpha']==.5 and r['config']['algorithm'] in ['fedavg','fedadam']]

def main():
    root=ROOT/'results'
    new=[json.loads(p.read_text()) for p in sorted((root/'selector_runs').glob('*.json'))]
    assert len(new)==100
    assert {json.dumps(r['config'],sort_keys=True) for r in new}=={json.dumps(c,sort_keys=True) for c in configurations()}
    original=selected_reference()
    masks=json.loads((ROOT/'configs/selector_extension_masks.json').read_text())
    masks={r['seed']:r for r in masks['records']}
    by={(r['config']['algorithm'],r['config']['mask'],r['config']['seed']):r for r in original+new}
    def rows(algo,mask):return [by[algo,mask,s] for s in SEEDS]
    comparisons=[];summary=[]
    for algo in ['fedavg','fedadam']:
        reference=rows(algo,'rasc')
        for mask in ['full','prevalence','rasc']+NAMES:
            if (algo,mask,SEEDS[0]) not in by:continue
            rr=rows(algo,mask);values=np.array([r['test']['macro_f1'] for r in rr])
            identical=0;jaccards=[]
            for r,ref in zip(rr,reference):
                a=np.asarray(r['mask'],bool);b=np.asarray(ref['mask'],bool)
                identical+=int(np.array_equal(a,b));jaccards.append(float((a&b).sum()/(a|b).sum()))
            summary.append(dict(algorithm=algo,mask=mask,label=LABELS[mask],n=len(rr),
                macro_f1_mean=float(values.mean()),macro_f1_sd=float(values.std(ddof=1)),
                accuracy_mean=float(np.mean([r['test']['accuracy'] for r in rr])),
                identical_rasc_masks=identical,jaccard_mean=float(np.mean(jaccards)),
                selected_features=rr[0]['selected_features'],
                mean_per_class_recall=np.mean([r['test']['per_class_recall'] for r in rr],axis=0).tolist()))
            if mask in NAMES:
                inference=contrast([r['test']['macro_f1'] for r in reference],values)
                comparisons.append(dict(algorithm=algo,comparison='rasc_minus_'+mask,
                    mask=mask,label=LABELS[mask],**inference))
    adjusted=holm([x['exact_sign_flip_p'] for x in comparisons])
    for row,p in zip(comparisons,adjusted):row['holm_p_ten_contrasts']=p
    report={'status':'complete','new_test_runs':len(new),'original_test_runs':320,
        'total_neural_test_runs':420,'validation_only_tuning_runs':36,
        'interpretation':'exploratory, conditional on one fixed BoT-IoT cohort; original benchmark outcomes known before this extension',
        'contrast_direction':'original RASC minus each alternative at the same 22-feature budget',
        'multiplicity':'Holm over all ten new contrasts; confidence intervals are pointwise',
        'protocol_sha256':hashlib.sha256((ROOT/'configs/selector_extension_protocol.json').read_bytes()).hexdigest(),
        'summary':summary,'contrasts':comparisons}
    (root/'selector_extension_inference.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,data in [('selector_extension_summary.csv',summary),('selector_extension_contrasts.csv',comparisons)]:
        with (root/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
