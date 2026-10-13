import unittest,numpy as np
from behavior_xr1.adapters import build_controller_slices,build_hold_action
class T(unittest.TestCase):
 def test_hold(self):
  s=build_controller_slices(['base','trunk','arm_left','gripper_left','arm_right','gripper_right'],[3,4,6,1,6,1],21); a=build_hold_action(s,trunk_target=np.arange(4),gripper_target=(.1,.2)); self.assertEqual(a.shape,(21,)); np.testing.assert_allclose(a[3:7],np.arange(4))
if __name__=='__main__':unittest.main()
