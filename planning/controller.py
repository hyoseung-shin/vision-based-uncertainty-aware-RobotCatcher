from __future__ import annotations
import numpy as np


GRIPPER_OPEN  = 255.0
GRIPPER_CLOSE = 0.0


class JointTracker:
    def __init__(self,
                 rise_time_s: float = 0.25,
                 v_max: float = 2.5):
        self.q_des = None 
        self.q_cmd = None  
        self.rise_time_s = rise_time_s
        self.v_max = v_max
        self.t_set = None
        self.gripper_cmd = GRIPPER_OPEN

    def set_target(self, q_target: np.ndarray, t_now: float) -> None:
        if self.q_cmd is None:
            self.q_cmd = q_target.copy()
        self.q_des = q_target.copy()
        self.t_set = t_now

    def close_gripper(self) -> None:
        self.gripper_cmd = GRIPPER_CLOSE

    def open_gripper(self) -> None:
        self.gripper_cmd = GRIPPER_OPEN

    def step(self, dt: float) -> np.ndarray | None:
        if self.q_des is None or self.q_cmd is None:
            return None
        delta = self.q_des - self.q_cmd
        max_step = self.v_max * dt
        norm = np.linalg.norm(delta)
        if norm > max_step:
            delta = delta * (max_step / norm)
        self.q_cmd = self.q_cmd + delta
        return np.concatenate([self.q_cmd, [self.gripper_cmd]])
