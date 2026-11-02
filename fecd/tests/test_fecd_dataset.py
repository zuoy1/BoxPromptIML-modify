import numpy as np
from fecd.datasets.transforms import apply_spatial_transform

def test_synchronized_flip_rotation():
 m=np.arange(12).reshape(3,4);image=np.repeat(m[:,:,None],3,axis=2);bm=np.stack([m,m+20],axis=-1)
 for op in ('hflip','vflip','rot90:1','rot90:3'):
  im,mm,bb=apply_spatial_transform(image,m,bm,operation=op)
  assert np.array_equal(im[:,:,0],mm)
  assert np.array_equal(bb[:,:,0],mm)

def test_box_coordinates_follow_discrete_geometry():
 from fecd.datasets.transforms import transform_boxes_xyxy
 box=np.array([[2,3,8,9]],np.float32)
 np.testing.assert_array_equal(transform_boxes_xyxy(box,(12,10),'hflip'),[[2,3,8,9]])
 # Non-symmetric to catch coordinate swaps.
 box=np.array([[1,2,4,6]],np.float32)
 np.testing.assert_array_equal(transform_boxes_xyxy(box,(10,12),'hflip'),[[8,2,11,6]])
 np.testing.assert_array_equal(transform_boxes_xyxy(box,(10,12),'vflip'),[[1,4,4,8]])
 np.testing.assert_array_equal(transform_boxes_xyxy(box,(10,12),'rot90:1'),[[2,8,6,11]])
