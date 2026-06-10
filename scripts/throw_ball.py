from __future__ import annotations
import numpy as np
import mujoco


def throw_ball(model, data, init_pos, init_vel, spin=None):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    qpos_a = int(model.jnt_qposadr[jid])
    qvel_a = int(model.jnt_dofadr[jid])
    data.qpos[qpos_a:qpos_a + 3] = init_pos
    data.qpos[qpos_a + 3:qpos_a + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[qvel_a:qvel_a + 3] = init_vel
    if spin is None:
        spin = np.zeros(3)
    data.qvel[qvel_a + 3:qvel_a + 6] = spin
