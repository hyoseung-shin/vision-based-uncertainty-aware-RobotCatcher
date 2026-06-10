from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import mujoco
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catch_robot import CatchRunner, RunConfig, ThrowConfig


def render_overview_video(runner: CatchRunner, out_path: str, fps: int = 50) -> None:
    model = runner.model
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "scene_ready"))
    from scripts.throw_ball import throw_ball
    cfg = runner.cfg
    throw_ball(model, data, np.array(cfg.throw.init_pos), np.array(cfg.throw.init_vel), np.array(cfg.throw.spin))
    mujoco.mj_forward(model, data)

    width, height = 960, 540
    renderer = mujoco.Renderer(model, height=height, width=width)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 3.0
    cam.azimuth = 130
    cam.elevation = -20
    cam.lookat[:] = [0.3, 0.3, 0.7]

    from planning.ik import FrankaIK
    from planning.controller import JointTracker
    ik = FrankaIK(model)
    tracker = JointTracker(rise_time_s=0.20, v_max=8.0)
    tracker.set_target(data.qpos[:7].copy(), 0.0)
    data.ctrl[:] = model.key_ctrl[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "scene_ready")]

    physics_steps = int(cfg.duration_s * 500)
    frame_every = int(500 / fps)
    committed = False
    catch_t = None
    catch_p = None

    from vision.predictor import AnalyticalPredictor
    from vision.intercept import InterceptSelector, WorkspaceApprox

    predictor = AnalyticalPredictor(horizon_s=cfg.horizon_s)
    selector  = InterceptSelector(workspace=WorkspaceApprox())

    for step in range(physics_steps):
        t = data.time
        if step % 8 == 0:    # 60 Hz vision
            ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
            qvel_a = int(model.jnt_dofadr[jid])
            p_hat = data.xpos[ball_id].copy()
            v_hat = data.qvel[qvel_a:qvel_a+3].copy()
            pred = predictor.predict(t, p_hat, v_hat,
                                     np.eye(3) * 1e-4, np.eye(3) * 1e-2)
            sel = selector.choose(pred, t)
            if sel is not None and sel.reachable and not committed:
                q_sol, ik_err, ok = ik.solve(data, sel.p_catch,
                                             q_init=data.qpos[:7])
                if ok and ik_err < 0.02:
                    tracker.set_target(q_sol, t)
                    committed = True
                    catch_t = sel.t_catch
                    catch_p = sel.p_catch
        cmd = tracker.step(1.0 / 500)
        if cmd is not None:
            data.ctrl[:8] = cmd
        if committed and catch_t is not None and t >= catch_t - 0.05:
            tracker.close_gripper()
        mujoco.mj_step(model, data)

        if step % frame_every == 0:
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            cv2.putText(frame, f"t={t:.2f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            if catch_p is not None:
                cv2.putText(frame,
                            f"catch -> ({catch_p[0]:+.2f}, "
                            f"{catch_p[1]:+.2f}, {catch_p[2]:+.2f})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    renderer.close()
    print(f"  Overview video: {out_path}")


def render_stats(runner: CatchRunner, out_path: str) -> None:
    t = np.array(runner.log_t)
    gt = np.array(runner.log_gt_pos)
    ee = np.array(runner.log_ee_pos)
    d  = np.array(runner.log_dist) * 100

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axs[0].plot(t, gt[:, 1], "k--", label="ball y")
    axs[0].plot(t, gt[:, 2], "k:",  label="ball z")
    axs[0].plot(t, ee[:, 1], "C0--", label="EE y")
    axs[0].plot(t, ee[:, 2], "C0:",  label="EE z")
    if runner.log_intercept:
        ti = np.array([row[0] for row in runner.log_intercept])
        pc = np.array([row[2] for row in runner.log_intercept])
        axs[0].scatter(ti, pc[:, 1], marker="x", c="C3", s=30, label="catch pt y")
        axs[0].scatter(ti, pc[:, 2], marker="x", c="C1", s=30, label="catch pt z")
    axs[0].set_ylabel("position (m)")
    axs[0].legend(ncol=3, fontsize=8); axs[0].grid(alpha=.3)
    axs[0].set_title("Ball, EE, and committed catch-point timeline")

    axs[1].plot(t, d, "C0-", lw=1.5)
    axs[1].axhline(10.0, ls="--", color="r", alpha=.5, label="catch threshold (10 cm)")
    axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("ball–EE distance (cm)")
    axs[1].legend(); axs[1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Stats plot:    {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gt", action="store_true")
    parser.add_argument("--vy", type=float, default=-2.5)
    parser.add_argument("--vz", type=float, default=1.5)
    parser.add_argument("--out-dir", default="/tmp")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        duration_s=1.4,
        use_gt_perception=args.use_gt,
        throw=ThrowConfig(init_pos=[0.0, 1.5, 1.7],
                          init_vel=[0.0, args.vy, args.vz]),
    )
    runner = CatchRunner(cfg)
    res = runner.run()
    print(f"\nCatch result: caught={res.caught}  "
          f"min_dist={res.min_dist_m*100:.1f}cm @ t={res.min_dist_t:.2f}s\n")

    render_stats(runner, str(out_dir / "demo_stats.png"))
    render_overview_video(runner, str(out_dir / "demo_overview.mp4"))
    runner.close()


if __name__ == "__main__":
    main()
