import unittest,numpy as np
from behavior_xr1.adapters import eef_local_to_base
class T(unittest.TestCase):
 def test_identity(self): np.testing.assert_allclose(eef_local_to_base([1,2,3],np.eye(3)),[1,2,3])
 def test_yaw(self): R=np.array([[0,-1,0],[1,0,0],[0,0,1]]); np.testing.assert_allclose(eef_local_to_base([1,0,0],R),[0,1,0])
if __name__=='__main__': unittest.main()

class RotationT(unittest.TestCase):
 def test_local_rotation_basis(self):
  from behavior_xr1.adapters.frame_transform import eef_local_rotation_to_base
  R=np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
  np.testing.assert_allclose(eef_local_rotation_to_base([.01,0,0],R),[0,.01,0])
 def test_local_rotation_all_axes(self):
  from behavior_xr1.adapters.frame_transform import eef_local_rotation_to_base
  R=np.array([[0,0,1],[0,1,0],[-1,0,0]],dtype=float)
  for v in np.eye(3): np.testing.assert_allclose(eef_local_rotation_to_base(.01*v,R),R@(.01*v))
