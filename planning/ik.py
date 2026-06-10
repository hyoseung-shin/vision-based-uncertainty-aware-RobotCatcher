from __future__ import annotations
import numpy as np
import mujoco


ARM_DOF = 7


class FrankaIK:
    def __init__(self, model: mujoco.MjModel, ee_body: str = "hand", damping: float = 0.08, step_max: float = 0.4, max_iter: int = 25, tol: float = 5e-4):
        self.model = model
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if self.ee_id < 0:
            raise KeyError(f"Body '{ee_body}' not found.")
        self.damping = damping
        self.step_max = step_max
        self.max_iter = max_iter
        self.tol = tol
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def solve(self, data: mujoco.MjData, target_pos: np.ndarray, q_init: np.ndarray | None = None) -> tuple[np.ndarray, float, bool]:
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0.0
        if q_init is not None:
            scratch.qpos[:ARM_DOF] = q_init

        for _ in range(self.max_iter):
            mujoco.mj_forward(self.model, scratch)
            ee_pos = scratch.xpos[self.ee_id].copy()
            err = target_pos - ee_pos
            err_norm = float(np.linalg.norm(err))
            if err_norm < self.tol:
                return scratch.qpos[:ARM_DOF].copy(), err_norm, True

            if err_norm > self.step_max:
                err = err * (self.step_max / err_norm)

            mujoco.mj_jacBody(self.model, scratch, self._jacp, self._jacr,
                              self.ee_id)
            J = self._jacp[:, :ARM_DOF]                 # 3 x 7

            JJt = J @ J.T + (self.damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, err)

            dq = np.clip(dq, -0.2, 0.2)
            scratch.qpos[:ARM_DOF] += dq

        return scratch.qpos[:ARM_DOF].copy(), err_norm, False
