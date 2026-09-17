import unittest, numpy as np
from behavior_xr1.adapters import RuntimeObservationAdapter
from behavior_xr1.training.noise import sample_noise
from behavior_xr1.data.tasks import TaskMap
class T(unittest.TestCase):
 def test_obs_rgb_rgba(self):
  o={'head':{'rgb':np.zeros((2,2,4),np.uint8)},'proprio':np.zeros(61),'instruction':'x','task_id':3}
  c=RuntimeObservationAdapter(rgb_keys=('head.rgb',))(o); self.assertEqual(c.rgb['head.rgb'].shape,(2,2,3)); self.assertEqual(c.task_id,3)
 def test_obs_depth_gate(self):
  o={'proprio':np.zeros(61)}; c=RuntimeObservationAdapter()(o); self.assertFalse(c.metadata['use_depth'])
 def test_noise(self):
  a=sample_noise((2,5,3),correlated=True,seed=4); b=sample_noise((2,5,3),correlated=True,seed=4); np.testing.assert_array_equal(a,b)
 def test_task(self):
  m=TaskMap([f't{i}' for i in range(100)]); self.assertEqual(m.id_for('t7'),7); self.assertEqual(m.task_for(7),'t7')
if __name__=='__main__': unittest.main()
