import unittest,numpy as np
from behavior_xr1.models import XR1PolicyWrapper
class T(unittest.TestCase):
 def test_mock_wrapper(self):
  w=XR1PolicyWrapper(lambda x: np.zeros((30,60))); self.assertEqual(w({}).shape,(30,60))
 def test_missing_backend(self):
  with self.assertRaises(RuntimeError): XR1PolicyWrapper()({})
if __name__=='__main__': unittest.main()
