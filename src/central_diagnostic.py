from pathlib import Path
import json,numpy as np,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiment import read_data,metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
D=read_data('bot_primary');x,y,_=D['train'];xv,yv,_=D['validation'];xt,yt,_=D['test']
results=[]
for alg in ['logistic','histgradient']:
 for seed in [104729,130363,155921]:
  start=time.perf_counter()
  m=LogisticRegression(max_iter=1000,C=1.,class_weight='balanced',random_state=seed) if alg=='logistic' else HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=31,l2_regularization=1.,class_weight='balanced',random_state=seed,early_stopping=False)
  m.fit(x,y);r={'method':'central_'+alg,'seed':seed,'validation':metrics(yv,m.predict_proba(xv)),'test':metrics(yt,m.predict_proba(xt)),'runtime_seconds':time.perf_counter()-start}
  results.append(r);print(alg,seed,r['test']['macro_f1'],flush=True)
(Path(__file__).resolve().parents[1]/'results/central_diagnostics.json').write_text(json.dumps(results,indent=2))
