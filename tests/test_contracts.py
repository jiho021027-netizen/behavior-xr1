import unittest

from behavior_xr1.data.contracts import (
    BEHAVIOR_R1PRO_ACTION_DIM,
    BEHAVIOR_R1PRO_PROPRIO_DIM,
    XR1_ACTION_DIM,
    XR1_STATE_DIM,
    assert_known_dimensions,
)


class ContractTests(unittest.TestCase):
    def test_audited_dimensions(self):
        assert_known_dimensions()

    def test_adapters_are_required(self):
        self.assertNotEqual(BEHAVIOR_R1PRO_PROPRIO_DIM, XR1_STATE_DIM)
        self.assertNotEqual(BEHAVIOR_R1PRO_ACTION_DIM, XR1_ACTION_DIM)


if __name__ == "__main__":
    unittest.main()
