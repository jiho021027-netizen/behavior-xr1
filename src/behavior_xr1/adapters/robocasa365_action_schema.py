"""Canonical action layout for the Xiaomi RoboCasa365 checkpoint."""
from __future__ import annotations
class RoboCasa365ActionSchema:
    model_dim: int = 60
    active_dim: int = 12
    end_effector_position: slice = slice(0, 3)
    end_effector_rotation: slice = slice(3, 6)
    gripper_close: slice = slice(6, 7)
    base_motion: slice = slice(7, 11)
    control_mode: slice = slice(11, 12)
    @property
    def active_indices(self): return tuple(range(self.active_dim))
    @property
    def masked_indices(self): return tuple(range(self.active_dim, self.model_dim))
    def validate(self):
        fields=(self.end_effector_position,self.end_effector_rotation,self.gripper_close,self.base_motion,self.control_mode)
        used=[i for field in fields for i in range(field.start,field.stop)]
        if sorted(used)!=list(self.active_indices): raise ValueError("RoboCasa365 fields must cover model indices 0:12 exactly")
        if self.active_dim>self.model_dim: raise ValueError("active dimensions exceed model dimensions")
XR1_ROBOCASA365_ACTION_SCHEMA=RoboCasa365ActionSchema()
XR1_ROBOCASA365_ACTION_SCHEMA.validate()
