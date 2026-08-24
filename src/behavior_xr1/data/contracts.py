"""Static interface facts confirmed from pinned upstream source.

These constants do not define a packing policy. In particular, silently
truncating BEHAVIOR proprioception or padding actions is forbidden.
"""

XR1_STATE_LENGTH = 1
XR1_STATE_DIM = 60
XR1_ACTION_HORIZON = 30
XR1_ACTION_DIM = 60

BEHAVIOR_R1PRO_PROPRIO_DIM = 61
BEHAVIOR_R1PRO_ACTION_DIM = 23
BEHAVIOR_TASK_COUNT = 100
BEHAVIOR_SKILL_COUNT = 31


def assert_known_dimensions() -> None:
    """Catch accidental edits to the audited interface constants."""
    assert (XR1_STATE_LENGTH, XR1_STATE_DIM) == (1, 60)
    assert (XR1_ACTION_HORIZON, XR1_ACTION_DIM) == (30, 60)
    assert BEHAVIOR_R1PRO_PROPRIO_DIM == 61
    assert BEHAVIOR_R1PRO_ACTION_DIM == 23
