import numpy as np
from fecd.pseudo.box_ops import PROMPT_VARIANTS,prompt_boxes

def test_base_k1_equals_k8_and_edge_clipping():
 b=np.array([[0,0,1,1],[95,95,100,100],[1,1,3,3]],np.float32)
 one=prompt_boxes(b,(100,100),1)[0];eight=prompt_boxes(b,(100,100),8)[0]
 for a,c in zip(one,eight):np.testing.assert_array_equal(a,c)
 for group in prompt_boxes(b,(100,100),8):
  for x1,y1,x2,y2 in group: assert 0<=x1<x2<=100 and 0<=y1<y2<=100

def test_prompt_variants_are_fixed():
 assert [v.name for v in PROMPT_VARIANTS]==['Base','Left','Right','Up','Down','Small','Large','Mixed']
