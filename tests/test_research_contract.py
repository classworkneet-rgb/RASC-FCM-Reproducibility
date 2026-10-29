from pathlib import Path
import unittest,sys,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from model import ArchitectureSpec,FixedGatedDBNCNN
from parameter_codec import encode,decode
from experiment import local_step,adaptive_update,partition
from rare_class_consensus import client_summary,aggregate_rare_class_mask

class ResearchContract(unittest.TestCase):
 def test_gradient_finite_difference(self):
  rng=np.random.default_rng(67);x=rng.normal(size=(9,7));y=rng.integers(0,3,9);mask=np.array([1,1,0,1,1,0,1],bool)
  m=FixedGatedDBNCNN(ArchitectureSpec(7,3),seed=43);initial=m.parameter_vector().copy();eta=1e-4
  m.train_batch(x,y,mask,learning_rate=eta,weight_decay=1e-5);analytic=(initial-m.parameter_vector())/eta
  for i in np.linspace(0,len(initial)-1,30,dtype=int):
   vp=initial.copy();vm=initial.copy();vp[i]+=1e-5;vm[i]-=1e-5
   m.load_parameter_vector(vp);lp=m.loss(x,y,mask,weight_decay=1e-5)
   m.load_parameter_vector(vm);lm=m.loss(x,y,mask,weight_decay=1e-5)
   self.assertAlmostEqual(analytic[i],(lp-lm)/2e-5,places=6)
 def test_fedprox_gradient_uses_pre_step_vector(self):
  rng=np.random.default_rng(72);x=rng.normal(size=(7,7));y=rng.integers(0,3,7);mask=np.ones(7,bool)
  m=FixedGatedDBNCNN(ArchitectureSpec(7,3),seed=3);baseline=FixedGatedDBNCNN(m.spec,seed=3)
  initial=m.parameter_vector().copy();current=initial+.2;eta=.01;mu=.3
  baseline.train_batch(x,y,mask,learning_rate=eta,class_weights=np.ones(3),weight_decay=1e-5)
  local_step(m,x,y,mask,eta,current,'fedprox',mu,np.zeros(len(initial)),np.zeros(len(initial)))
  np.testing.assert_allclose(m.parameter_vector(),baseline.parameter_vector()-eta*mu*(initial-current),atol=1e-14)
 def test_scaffold_control_correction(self):
  rng=np.random.default_rng(73);x=rng.normal(size=(7,7));y=rng.integers(0,3,7);mask=np.ones(7,bool)
  m=FixedGatedDBNCNN(ArchitectureSpec(7,3),seed=4);baseline=FixedGatedDBNCNN(m.spec,seed=4)
  initial=m.parameter_vector().copy();eta=.01;ci=np.full(len(initial),.2);c=np.full(len(initial),.1)
  baseline.train_batch(x,y,mask,learning_rate=eta,class_weights=np.ones(3),weight_decay=1e-5)
  local_step(m,x,y,mask,eta,initial,'scaffold',0,ci,c)
  np.testing.assert_allclose(m.parameter_vector(),baseline.parameter_vector()+eta*(ci-c),atol=1e-14)
  cnew=ci-c+(initial-m.parameter_vector())/(eta*1)
  # Option II recovers the uncorrected one-step local gradient.
  np.testing.assert_allclose(cnew,(initial-baseline.parameter_vector())/eta,atol=1e-12)
 def test_fedopt_moments(self):
  d=np.array([.2,-.3]);z=np.zeros(2)
  step,m,v=adaptive_update(d,z,z,'fedadam',.01)
  np.testing.assert_allclose(m,.1*d);np.testing.assert_allclose(v,.01*d*d)
  np.testing.assert_allclose(step,.01*m/(np.sqrt(v)+.001))
  step,m,v=adaptive_update(d,z,z,'fedyogi',.01)
  self.assertTrue(np.all(v>=0));np.testing.assert_allclose(v,.01*d*d)
 def test_codec_preserves_predictions_and_active_updates(self):
  rng=np.random.default_rng(83);x=rng.normal(size=(19,7));y=rng.integers(0,3,19);mask=np.array([1,0,1,1,0,1,1],bool)
  m=FixedGatedDBNCNN(ArchitectureSpec(7,3),seed=7);copy=FixedGatedDBNCNN(m.spec,seed=7)
  payload=encode(m,mask);copy.load_parameter_vector(decode(payload,copy,mask))
  self.assertEqual(len(payload),8*(m.parameter_count()-2*24))
  np.testing.assert_allclose(m.predict_proba(x,mask),copy.predict_proba(x,mask),atol=1e-15)
  for _ in range(3):
   m.train_batch(x,y,mask,learning_rate=.003,weight_decay=1e-5)
   copy.train_batch(x,y,mask,learning_rate=.003,weight_decay=1e-5)
   np.testing.assert_allclose(m.predict_proba(x,mask),copy.predict_proba(x,mask),atol=1e-15)
 def test_partition_cover_no_overlap(self):
  y=np.repeat(np.arange(5),1000)
  for alpha in [.1,.3,.5,1.]:
   for n in [5,10,20]:
    try:clients=partition(y,104729,n,alpha)
    except RuntimeError:
     # A sparse Dirichlet draw may produce an empty client. It must fail
     # explicitly instead of silently changing the random partition.
     self.assertEqual(alpha,.1);continue
    joined=np.concatenate(clients)
    self.assertEqual(len(joined),len(y));self.assertEqual(len(np.unique(joined)),len(y))
 def test_selector_budget_and_reproducibility(self):
  rng=np.random.default_rng(14);x=rng.normal(size=(400,15));y=np.repeat(np.arange(4),100)
  summaries=[client_summary(x,y,classes=4,candidate_budget=9,bootstraps=8,seed=1)]
  mask,_=aggregate_rare_class_mask(summaries,feature_budget=9)
  mask2,_=aggregate_rare_class_mask(summaries,feature_budget=9)
  self.assertEqual(int(mask.sum()),9);np.testing.assert_array_equal(mask,mask2)
if __name__=='__main__':unittest.main()
