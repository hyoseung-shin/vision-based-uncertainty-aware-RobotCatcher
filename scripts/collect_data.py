from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.throw_ball import throw_ball


SCENE = str(ROOT / "assets" / "franka_emika_panda" / "catch_scene.xml")
PHYSICS_HZ = 500
SAMPLE_HZ = 60
SAMPLE_EVERY = PHYSICS_HZ // SAMPLE_HZ


def sample_one_throw(model, data, rng) -> np.ndarray:
    init_pos = np.array([
        rng.uniform(-0.3, 0.3),
        rng.uniform(1.3, 1.7),
        rng.uniform(1.5, 1.9),
    ])
    init_vel = np.array([
        rng.uniform(-0.3, 0.3),
        rng.uniform(-3.2, -1.8),
        rng.uniform(0.5, 2.2),
    ])
    spin = rng.normal(scale=2.0, size=3)

    mujoco.mj_resetData(model, data)
    throw_ball(model, data, init_pos, init_vel, spin)
    mujoco.mj_forward(model, data)

    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    qvel_a = int(model.jnt_dofadr[jid])

    samples = []
    for step in range(int(1.5 * PHYSICS_HZ)):
        if step % SAMPLE_EVERY == 0:
            p = data.xpos[ball_id].copy()
            v = data.qvel[qvel_a:qvel_a + 3].copy()
            samples.append(np.concatenate([p, v]))
            if p[2] < 0.06:
                break
        mujoco.mj_step(model, data)
    return np.array(samples)


def windowize(traj: np.ndarray, window: int,
              horizon: int) -> tuple[np.ndarray, np.ndarray]:
    T = traj.shape[0]
    if T < window + horizon:
        return np.zeros((0, window, 6)), np.zeros((0, horizon, 3))
    X = np.stack([traj[i:i + window]
                  for i in range(T - window - horizon + 1)])
    Y = np.stack([traj[i + window:i + window + horizon, :3]
                  for i in range(T - window - horizon + 1)])
    return X, Y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-throws", type=int, default=2000)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--horizon", type=int, default=36)
    p.add_argument("--out", default="data/trajectories.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)

    Xs, Ys = [], []
    for k in range(args.n_throws):
        traj = sample_one_throw(model, data, rng)
        Xk, Yk = windowize(traj, args.window, args.horizon)
        if Xk.shape[0] > 0:
            Xs.append(Xk); Ys.append(Yk)
        if (k + 1) % 200 == 0:
            n = sum(x.shape[0] for x in Xs)
            print(f"  [{k+1}/{args.n_throws}]  pairs so far: {n}", flush=True)

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    Y = np.concatenate(Ys, axis=0).astype(np.float32)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             X=X, Y=Y,
             window=args.window, horizon=args.horizon)
    print(f"Saved {X.shape[0]} pairs to {out_path}")
    print(f"  X shape = {X.shape}")
    print(f"  Y shape = {Y.shape}")


if __name__ == "__main__":
    main()
