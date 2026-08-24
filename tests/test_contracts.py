import unittest

from behavior_xr1.adapters import BehaviorStatePacker, XR1ActionToBehaviorAction
from behavior_xr1.data.contracts import (
    BEHAVIOR_R1PRO_ACTION_DIM,
    BEHAVIOR_R1PRO_PROPRIO_DIM,
    XR1_ACTION_DIM,
    XR1_STATE_DIM,
    assert_known_dimensions,
)
from behavior_xr1.data.schema import (
    BEHAVIOR_DEFAULT_ACTION_COMPONENTS,
    BEHAVIOR_DEFAULT_ACTION_DIM,
    Component,
    XR1_ACTION_COMPONENTS,
    XR1_ACTION_DIM,
    XR1_STATE_COMPONENTS,
    XR1_STATE_DIM,
    validate_layout,
)


class ContractTests(unittest.TestCase):
    def test_audited_dimensions(self):
        assert_known_dimensions()

    def test_adapters_are_required(self):
        self.assertNotEqual(BEHAVIOR_R1PRO_PROPRIO_DIM, XR1_STATE_DIM)
        self.assertNotEqual(BEHAVIOR_R1PRO_ACTION_DIM, XR1_ACTION_DIM)

    def test_complete_audited_layouts(self):
        validate_layout(BEHAVIOR_DEFAULT_ACTION_COMPONENTS, BEHAVIOR_DEFAULT_ACTION_DIM)
        validate_layout(XR1_STATE_COMPONENTS, XR1_STATE_DIM)
        validate_layout(XR1_ACTION_COMPONENTS, XR1_ACTION_DIM)

    def test_overlapping_or_gapped_slices_fail(self):
        with self.assertRaises(ValueError):
            validate_layout((Component("a", 0, 2, "", ""), Component("b", 1, 3, "", "")), 3)

    def test_semantic_conversion_cannot_run_silently(self):
        with self.assertRaises(NotImplementedError):
            BehaviorStatePacker().pack([0.0] * 61)
        with self.assertRaises(NotImplementedError):
            XR1ActionToBehaviorAction().convert([[0.0] * 60] * 30)

    def test_invalid_adapter_shapes_are_explicit(self):
        with self.assertRaises(ValueError):
            BehaviorStatePacker().pack([0.0] * 60)
        with self.assertRaises(ValueError):
            XR1ActionToBehaviorAction().convert([[0.0] * 60] * 29)
        with self.assertRaises(ValueError):
            XR1ActionToBehaviorAction().convert([[0.0] * 59] * 30)


if __name__ == "__main__":
    unittest.main()
