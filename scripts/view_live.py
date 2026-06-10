from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.camera   import StereoRig, camera_model_from_mujoco
from vision.detector import RedBallDetector
from vision.stereo   import triangulate
from vision.tracker  import BallTracker
from vision.ekf      import BallEKF
from vision.predictor import AnalyticalPredictor
from vision.intercept import InterceptSelector, WorkspaceApprox
from planning.ik     import FrankaIK
from planning.controller import JointTracker
from scripts.throw_ball import throw_ball


SCENE = str(ROOT / "assets" / "franka_emika_panda" / "catch_scene.xml")
PHYSICS_HZ = 500
VISION_EVERY = 8                        # 60 Hz vision


class LiveRunner:
    def __init__(self, use_gt: bool = False, vy: float = -2.5, vz: float = 1.5):
        self.use_gt = use_gt
        self.vy = vy
        self.vz = vz
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.data  = mujoco.MjData(self.model)

        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "scene_ready")

        self.rig = None if use_gt else StereoRig(self.model, self.data)
        self.detector = RedBallDetector(pixel_sigma_base=1.0)
        self.tracker = BallTracker()
        self.ekf = BallEKF(sigma_a=6.0)
        self.predictor = AnalyticalPredictor(horizon_s=0.6)
        self.intercept = InterceptSelector(workspace=WorkspaceApprox())

        self.ik = FrankaIK(self.model)
        self.joint_tracker = JointTracker(rise_time_s=0.20, v_max=8.0)

        self.committed_catch_t = None
        self.committed_catch_p = None
        self.catch_executed = False
        self.t_first_intercept = None
        self.n_detections = 0
        self._reset_throw()

    def _reset_throw(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        throw_ball(self.model, self.data, init_pos=np.array([0.0, 1.5, 1.7]), init_vel=np.array([0.0, self.vy, self.vz]))
        self.data.ctrl[:] = self.model.key_ctrl[self.key_id]
        mujoco.mj_forward(self.model, self.data)
        self.joint_tracker = JointTracker(rise_time_s=0.20, v_max=8.0)
        self.joint_tracker.set_target(self.data.qpos[:7].copy(), 0.0)
        self.tracker = BallTracker()
        self.ekf = BallEKF(sigma_a=6.0)
        self.committed_catch_t = None
        self.committed_catch_p = None
        self.catch_executed = False
        self.t_first_intercept = None
        self.n_detections = 0
        print(f"\n[reset] throw launched at t={self.data.time:.3f}  "
              f"vy={self.vy:+.1f}  vz={self.vz:+.1f}")

    def step_once(self, step_idx: int):
        t = self.data.time

        if step_idx % VISION_EVERY == 0:
            self._vision_tick(t)

        cmd = self.joint_tracker.step(1.0 / PHYSICS_HZ)
        if cmd is not None:
            self.data.ctrl[:8] = cmd

        if (self.committed_catch_t is not None and
                not self.catch_executed and
                t >= self.committed_catch_t - 0.05):
            self.joint_tracker.close_gripper()
            self.catch_executed = True

        mujoco.mj_step(self.model, self.data)

    def _vision_tick(self, t: float):
        X = None; S = None
        if self.use_gt:
            p_gt = self.data.xpos[self.ball_id].copy()
            noise = np.random.randn(3) * 0.01
            X = p_gt + noise
            S = np.diag([1e-4, 9e-4, 1e-4])
            if p_gt[2] < 0.06 or p_gt[1] < -0.5 or p_gt[1] > 2.0:
                X = None
        else:
            img_L, img_R = self.rig.render_pair()
            cam_L, cam_R = self.rig.models()
            dL = self.detector.detect(img_L); dR = self.detector.detect(img_R)
            if dL is not None and dR is not None:
                X, S = triangulate(dL.center, dR.center, cam_L, cam_R, dL.pixel_sigma, dR.pixel_sigma)

        if X is not None:
            tr = self.tracker.step(t, X, S)
            if tr is not None:
                self.n_detections += 1
                if not self.ekf.initialized():
                    self.ekf.initialize(t, X)
                else:
                    self.ekf.update(t, X, S)
        else:
            self.tracker.step(t, None, None)

        if self.ekf.initialized() and self.ekf.n_updates >= 6:
            p_hat = self.ekf.position
            v_hat = self.ekf.velocity
            Pp = self.ekf.P[:3, :3]; Pv = self.ekf.P[3:6, 3:6]
            pred = self.predictor.predict(t, p_hat, v_hat, Pp, Pv)
            sel = self.intercept.choose(pred, t)
            if sel is not None and sel.reachable:
                accept = self.committed_catch_p is None or (
                    np.linalg.norm(sel.p_catch - self.committed_catch_p) > 0.05
                    and (self.committed_catch_t - t) > 0.10
                )
                if accept:
                    q_sol, ik_err, ok = self.ik.solve(
                        self.data, sel.p_catch, q_init=self.data.qpos[:7])
                    if ok and ik_err < 0.02:
                        if self.t_first_intercept is None:
                            self.t_first_intercept = t
                        self.joint_tracker.set_target(q_sol, t)
                        self.committed_catch_t = sel.t_catch
                        self.committed_catch_p = sel.p_catch.copy()
                        print(f"  t={t:.3f}  commit catch @ t={sel.t_catch:.3f}  "
                              f"p={sel.p_catch.round(3)}")


def main():
    parser = argparse.ArgumentParser(
        description="Live MuJoCo viewer for the catching pipeline.")
    parser.add_argument("--use-gt", action="store_true", help="Bypass stereo rendering (faster, smoother).")
    parser.add_argument("--vy", type=float, default=-2.5)
    parser.add_argument("--vz", type=float, default=1.5)
    parser.add_argument("--auto-reset", action="store_true", help="Automatically re-throw 1 s after each catch attempt.")
    parser.add_argument("--experiment", action="store_true",
                        help="Experiment mode: randomized throws + running "
                             "catch-rate statistics printed to the console. "
                             "Implies --auto-reset.")
    parser.add_argument("--n-throws", type=int, default=0,
                        help="Stop after this many throws in experiment mode "
                             "(0 = run forever until you close the window).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for randomized throws (experiment mode).")
    args = parser.parse_args()

    runner = LiveRunner(use_gt=args.use_gt, vy=args.vy, vz=args.vz)
    print("Opening MuJoCo viewer.")
    print("  • SPACE      pause/resume")
    print("  • BACKSPACE  reset to keyframe & re-throw")
    print("  • mouse      orbit / pan / zoom")
    if args.experiment:
        print(f"  • EXPERIMENT MODE: throws are randomized; statistics printed "
              f"as we go. Seed = {args.seed}, target = "
              f"{args.n_throws or 'infinite'} throws.")
        args.auto_reset = True
    if args.auto_reset:
        print("  • auto-reset enabled (new throw 1 s after each catch attempt)")

    rng = np.random.default_rng(args.seed)

    throw_count = 0
    caught_count = 0
    min_dist_log = []
    intercept_t_log = []

    def randomize_throw():
        vy = float(rng.uniform(-3.0, -2.0))
        vz = float(rng.uniform(0.8, 2.0))
        runner.vy = vy
        runner.vz = vz

    def evaluate_and_log_throw():
        nonlocal throw_count, caught_count
        ee_id = runner.ee_id; ball_id = runner.ball_id
        ball_pos = runner.data.xpos[ball_id]
        ee_pos = runner.data.xpos[ee_id]
        dist = float(np.linalg.norm(ball_pos - ee_pos))
        caught = (dist < 0.10 and runner.committed_catch_t is not None)
        throw_count += 1
        if caught: caught_count += 1
        min_dist_log.append(dist)
        if runner.t_first_intercept is not None:
            intercept_t_log.append(runner.t_first_intercept)
        rate = 100 * caught_count / max(throw_count, 1)
        mean_dist = 100 * np.mean(min_dist_log) if min_dist_log else 0
        print(f"  [throw {throw_count:3d}]  caught={caught}  "
              f"final_dist={dist*100:.1f}cm  vy={runner.vy:+.2f} vz={runner.vz:+.2f}"
              f"  →  running catch_rate={rate:.0f}%  "
              f"mean_dist={mean_dist:.1f}cm")

    step_idx = 0

    with mujoco.viewer.launch_passive(runner.model, runner.data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0.3, 0.3, 0.7]

        while viewer.is_running():
            step_start = time.perf_counter()
            runner.step_once(step_idx)
            step_idx += 1

            if args.auto_reset and runner.committed_catch_t is not None:
                if runner.data.time - runner.committed_catch_t > 1.0:
                    if args.experiment:
                        evaluate_and_log_throw()
                        if args.n_throws > 0 and throw_count >= args.n_throws:
                            print(f"\nReached target of {args.n_throws} throws.")
                            print(f"Final catch rate: {caught_count}/{throw_count} "
                                  f"= {100*caught_count/throw_count:.0f}%")
                            print(f"Mean min distance: "
                                  f"{100*np.mean(min_dist_log):.1f} cm")
                            if intercept_t_log:
                                print(f"Mean first-intercept time: "
                                      f"{1000*np.mean(intercept_t_log):.0f} ms")
                            break
                        randomize_throw()
                    runner._reset_throw()
                    step_idx = 0

            viewer.sync()
            elapsed = time.perf_counter() - step_start
            sleep_for = (1.0 / PHYSICS_HZ) - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()
