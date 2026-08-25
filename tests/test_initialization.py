import unittest

import numpy as np

from behavior_xr1.models.initialization import (
    NEW_ACTION_DIMS,
    NEW_STATE_DIMS,
    choice_output_rows,
    zero_new_behavior_parameters,
)


class InitializationPolicyTests(unittest.TestCase):
    def test_only_new_dimension_weights_are_zeroed(self):
        parameters = {
            "state_projector.layers.0.weight": np.ones((4, 60)),
            "state_projector_choice.layers.0.weight": np.ones((5, 60)),
            "action_projector.layers.0.weight": np.ones((6, 60)),
            "action_output_layer.layers.2.weight": np.ones((60, 7)),
            "action_projector_choice.1.layers.0.weight": np.ones((300, 8)),
        }
        zero_new_behavior_parameters(parameters)
        for name in ("state_projector.layers.0.weight", "state_projector_choice.layers.0.weight"):
            self.assertTrue((parameters[name][:, NEW_STATE_DIMS] == 0).all())
            self.assertTrue((parameters[name][:, 0:16] == 1).all())
        self.assertTrue((parameters["action_projector.layers.0.weight"][:, NEW_ACTION_DIMS] == 0).all())
        self.assertTrue((parameters["action_projector.layers.0.weight"][:, 17:20] == 1).all())
        self.assertTrue((parameters["action_output_layer.layers.2.weight"][NEW_ACTION_DIMS] == 0).all())
        self.assertTrue((parameters["action_output_layer.layers.2.weight"][0:20] == 1).all())
        self.assertTrue((parameters["action_projector_choice.1.layers.0.weight"][choice_output_rows()] == 0).all())

    def test_missing_or_changed_checkpoint_tensor_fails(self):
        with self.assertRaises(KeyError):
            zero_new_behavior_parameters({})


if __name__ == "__main__":
    unittest.main()
