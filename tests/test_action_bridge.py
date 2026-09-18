import unittest,numpy as np
from behavior_xr1.adapters import build_controller_slices,XR1EEFActionAdapter
class T(unittest.TestCase):
 def test_dynamic(self):
  s=build_controller_slices(['base','trunk','arm_left','gripper_left','arm_right','gripper_right'],[3,4,6,1,6,1],21); self.assertEqual((s[2].start,s[2].stop),(7,13))
 def test_bridge(self):
  s=build_controller_slices(['base','trunk','arm_left','gripper_left','arm_right','gripper_right'],[3,4,6,1,6,1],21); b=XR1EEFActionAdapter(s); x=b.convert(np.zeros((2,60)),np.eye(3),np.eye(3),trunk=np.zeros((2,4)),gripper=np.zeros((2,2))); self.assertEqual(x.shape,(2,21))
 def test_strict(self):
  s=build_controller_slices(['base','trunk','arm_left','gripper_left','arm_right','gripper_right'],[3,4,6,1,6,1],21); 
  with self.assertRaises(ValueError): XR1EEFActionAdapter(s).convert(np.zeros(60),np.eye(3),np.eye(3),trunk=np.zeros(4))
if __name__=='__main__': unittest.main()
def test_neutral_guard_rejects_unvalidated_base_and_grippers():
 import numpy as np
 from behavior_xr1.adapters.r1pro_action_schema import build_controller_slices
 from behavior_xr1.adapters.action_bridge import XR1EEFActionAdapter
 s=build_controller_slices(['base','trunk','arm_left','gripper_left','arm_right','gripper_right'],[3,4,6,1,6,1],21); b=XR1EEFActionAdapter(s); kw=dict(rotation_left=np.eye(3),rotation_right=np.eye(3),trunk=np.zeros(4),gripper=np.zeros(2))
 x=np.zeros(60); x[17]=1
 import pytest
 with pytest.raises(ValueError,match='UNVALIDATED_BASE_NONZERO'): b.convert(x,**kw)
 x=np.zeros(60); x[6]=1
 with pytest.raises(ValueError,match='UNVALIDATED_GRIPPER_NONZERO'): b.convert(x,**kw)
