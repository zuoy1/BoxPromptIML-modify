import numpy as np
from fecd.pseudo.sam_ensemble import SamEnsembleGenerator,summarize_prompt_probabilities

class FakePredictor:
 model=None
 def __init__(self):self.set_count=0;self.calls=0
 def set_image(self,image):self.set_count+=1;self.shape=image.shape[:2]
 def predict(self,box,**kwargs):
  assert kwargs=={'point_coords':None,'multimask_output':False,'return_logits':True};self.calls+=1
  h,w=self.shape;mask=np.full((1,h,w),-2 if self.calls%2 else 2,np.float32);return mask,np.array([.8],np.float32),np.zeros((1,256,256),np.float32)
def test_multi_box_max_and_empty():
 p=FakePredictor();g=SamEnsembleGenerator('/does/not/exist',device='cpu',predictor=p)
 image=np.zeros((17,23,3),np.uint8);r=g.predict(image,[[0,0,2,2],[20,15,23,17]],1)
 assert p.set_count==1 and p.calls==2 and r['prompt_probs'].shape==(1,17,23)
 # two component probs are max-fused; one is sigmoid(+2)
 assert r['native']['mu'].min()>.8
 e=g.predict(image,[],8);assert e['empty_boxes'] and not e['native']['mu'].any() and e['prompt_probs'].shape==(8,17,23)

def test_population_stats_and_threshold_votes():
 p=np.array([[[.2,.5]],[[.8,.5]]],np.float32);mu,var,a=summarize_prompt_probabilities(p)
 np.testing.assert_allclose(mu,[[.5,.5]]);np.testing.assert_allclose(var,[[.09,0]])
 np.testing.assert_allclose(a,[[.5,2/3]])
