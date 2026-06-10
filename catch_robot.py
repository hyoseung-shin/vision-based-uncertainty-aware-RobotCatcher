"""
catch_robot.py
==============
Full integrated catching simulation.

Runs a configurable number of throws.  For each throw:

  1. Reset scene to home keyframe.
  2. Throw the ball with the configured initial state.
  3. Multi-rate loop:
        physics @ PHYSICS_HZ (default 500 Hz)
        vision  @ VISION_HZ  (default 60 Hz)  → detect → triangulate
                                                tracker → EKF
                                                predictor → interception
                                                IK → joint target
        control @ PHYSICS_HZ                  → smoothed joint command
  4. Detect catch / miss by checking ball-to-EE distance + contact.
  5. Log everything for later analysis.

This file is BOTH the runtime that the README points to ("python catch_robot.py")
and the building block used by scripts/run_experiments.py.
"""

from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import mujoco
import cv2

# ----- our modules ---------------------------------------------------------
from vision.camera   import StereoRig
from vision.detector import RedBallDetector
from vision.stereo   import triangulate
from vision.tracker  import BallTracker
from vision.ekf      import BallEKF
from vision.predictor import AnalyticalPredictor, LearnedPredictor
from vision.intercept import InterceptSelector, WorkspaceApprox
from planning.ik     import FrankaIK
from planning.controller import JointTracker, GRIPPER_OPEN, GRIPPER_CLOSE
from scripts.throw_ball  import throw_ball


# ============================== configuration ==============================
ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = str(ROOT / "assets" / "franka_emika_panda" / "catch_scene.xml")

PHYSICS_HZ = 500
VISION_HZ  = 60
VISION_EVERY = PHYSICS_HZ // VISION_HZ
CATCH_DIST_THRESH = 0.10       # m : ball within 10 cm of EE counts as catch
CATCH_VEL_THRESH  = 1.5        # m/s: must be slow enough to "hold"


@dataclass
class ThrowConfig:
    """Initial conditions for a single throw."""
    init_pos: list = field(default_factory=lambda: [0.0, 1.5, 1.7])
    init_vel: list = field(default_factory=lambda: [0.0, -3.0, 0.5])
    spin:     list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class RunConfig:
    """All knobs for a single simulation run."""
    scene_xml: str = DEFAULT_SCENE
    duration_s: float = 1.5
    predictor: str = "analytical"           # "analytical" or "learned"
    learned_ckpt: str | None = None
    ekf_sigma_a: float = 6.0
    horizon_s: float = 0.6
    pixel_sigma_base: float = 1.0
    save_video: bool = False
    video_path: str = "/tmp/run.mp4"
    verbose: bool = False
    # If True, skip rendering and feed the EKF the GT ball position plus
    # synthetic Gaussian noise (vision_noise_m std-dev). Used for fast
    # ablation sweeps on the control / prediction side of the pipeline
    # without paying the per-frame stereo render cost.
    use_gt_perception: bool = False
    vision_noise_m: float = 0.01
    throw: ThrowConfig = field(default_factory=ThrowConfig)


@dataclass
class RunResult:
    caught: bool
    min_dist_m: float                       # closest EE-ball distance
    min_dist_t: float                       # time at closest approach
    t_first_detection: float | None
    t_first_intercept: float | None
    n_vision_frames: int
    n_detections: int
    final_ball_pos: list
    final_ee_pos: list
    final_ball_vel: list
    wall_time_s: float

    def to_dict(self):
        return asdict(self)


# =============================== main runner ===============================
class CatchRunner:
    """Encapsulates one full throw → catch simulation."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.scene_xml)
        self.data  = mujoco.MjData(self.model)

        self.ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.ball_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")

        # Perception
        if cfg.use_gt_perception:
            self.rig = None
        else:
            self.rig = StereoRig(self.model, self.data)
        self.detector = RedBallDetector(pixel_sigma_base=cfg.pixel_sigma_base)
        self.tracker  = BallTracker()
        self.ekf      = BallEKF(sigma_a=cfg.ekf_sigma_a)

        if cfg.predictor == "learned":
            self.predictor = LearnedPredictor(
                ckpt_path=cfg.learned_ckpt, horizon_s=cfg.horizon_s)
        else:
            self.predictor = AnalyticalPredictor(horizon_s=cfg.horizon_s)

        self.intercept = InterceptSelector(workspace=WorkspaceApprox())

        # Control
        self.ik = FrankaIK(self.model)
        self.joint_tracker = JointTracker(rise_time_s=0.20, v_max=8.0)

        # Logging buffers
        self.log_t = []
        self.log_gt_pos = []
        self.log_ekf_pos = []
        self.log_ee_pos = []
        self.log_intercept = []         # tuples (t_now, t_catch, p_catch)
        self.log_dist = []

        self._reset()

    def _reset(self):
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "scene_ready")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        throw_ball(self.model, self.data,
                   init_pos=np.array(self.cfg.throw.init_pos),
                   init_vel=np.array(self.cfg.throw.init_vel),
                   spin=np.array(self.cfg.throw.spin))
        # Initialise actuator commands to the keyframe ctrl so we don't snap.
        self.data.ctrl[:] = self.model.key_ctrl[key_id]
        mujoco.mj_forward(self.model, self.data)
        self.joint_tracker.set_target(self.data.qpos[:7].copy(),
                                      t_now=0.0)

    def run(self) -> RunResult:
        cfg = self.cfg
        n_steps = int(cfg.duration_s * PHYSICS_HZ)
        dt_phys = 1.0 / PHYSICS_HZ

        wall_start = time.perf_counter()
        n_vision_frames = 0
        n_detections = 0
        t_first_detect = None
        t_first_intercept = None
        committed_catch_t = None
        committed_catch_p = None
        catch_executed = False

        # For optional video capture, render every 2nd physics tick (250 fps
        # too high; we use 50 fps overview).
        video_writer = None
        if cfg.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                cfg.video_path, fourcc, 50.0, (640, 480))

        min_dist = np.inf
        min_dist_t = 0.0

        for step in range(n_steps):
            t = self.data.time

            # ----- vision tick ---------------------------------------------
            if step % VISION_EVERY == 0:
                n_vision_frames += 1
                X = None
                S = None
                dL = dR = None
                if cfg.use_gt_perception:
                    # Bypass rendering: read ball GT and add Gaussian noise.
                    p_gt = self.data.xpos[self.ball_id].copy()
                    noise = np.random.randn(3) * cfg.vision_noise_m
                    X = p_gt + noise
                    # Construct an anisotropic measurement covariance that
                    # roughly matches the real stereo behaviour (larger
                    # along y, the depth-aligned axis).
                    S = np.diag([
                        cfg.vision_noise_m ** 2,
                        (cfg.vision_noise_m * 3) ** 2,
                        cfg.vision_noise_m ** 2,
                    ])
                    # Only "detect" while the ball is above the floor and
                    # within plausible cam frustum (y in [-0.5, 2.0]).
                    if p_gt[2] < 0.06 or p_gt[1] < -0.5 or p_gt[1] > 2.0:
                        X = None
                else:
                    img_L, img_R = self.rig.render_pair()
                    cam_L, cam_R = self.rig.models()
                    dL = self.detector.detect(img_L)
                    dR = self.detector.detect(img_R)
                    if dL is not None and dR is not None:
                        X, S = triangulate(
                            dL.center, dR.center, cam_L, cam_R,
                            dL.pixel_sigma, dR.pixel_sigma)

                if X is not None:
                    tr = self.tracker.step(t, X, S)
                    if tr is not None:
                        n_detections += 1
                        if t_first_detect is None:
                            t_first_detect = t
                        if not self.ekf.initialized():
                            self.ekf.initialize(t, X)
                        else:
                            self.ekf.update(t, X, S)
                else:
                    self.tracker.step(t, None, None)

                # ----- planning if EKF has enough info ---------------------
                if self.ekf.initialized() and self.ekf.n_updates >= 6:
                    p_hat = self.ekf.position
                    v_hat = self.ekf.velocity
                    Pp = self.ekf.P[:3, :3]
                    Pv = self.ekf.P[3:6, 3:6]
                    if cfg.predictor == "learned":
                        p_hist = np.array([p_hat])
                        v_hist = np.array([v_hat])
                        pred = self.predictor.predict(
                            t, p_hist, v_hist, Pp, Pv)
                    else:
                        pred = self.predictor.predict(
                            t, p_hat, v_hat, Pp, Pv)
                    sel = self.intercept.choose(pred, t)
                    if sel is not None and sel.reachable:
                        # Decide whether to (re-)commit. Once we have a
                        # committed catch point, we only override it if the
                        # new candidate differs by > 5 cm — this prevents
                        # the EE from constantly chasing the predictor's
                        # noise and lets it actually arrive somewhere.
                        accept_new = False
                        if committed_catch_p is None:
                            accept_new = True
                        else:
                            diff = np.linalg.norm(
                                sel.p_catch - committed_catch_p)
                            time_left = committed_catch_t - t
                            # Allow revisions until we are within 100 ms of
                            # the committed time; after that, freeze.
                            if diff > 0.05 and time_left > 0.10:
                                accept_new = True
                        if accept_new:
                            q_sol, ik_err, ok = self.ik.solve(
                                self.data, sel.p_catch,
                                q_init=self.data.qpos[:7])
                            if ok and ik_err < 0.02:
                                if t_first_intercept is None:
                                    t_first_intercept = t
                                self.joint_tracker.set_target(q_sol, t)
                                committed_catch_t = sel.t_catch
                                committed_catch_p = sel.p_catch.copy()
                                self.log_intercept.append(
                                    (t, sel.t_catch, sel.p_catch.copy()))

                # ----- optional video capture (left cam, with overlays) ----
                if video_writer is not None and not cfg.use_gt_perception:
                    frame = img_L.copy()
                    if dL is not None:
                        x1, y1, x2, y2 = map(int, dL.bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      (0, 255, 0), 2)
                    cv2.putText(frame, f"t={t:.2f}s  dets={n_detections}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # ----- control tick (always) ----------------------------------
            cmd = self.joint_tracker.step(dt_phys)
            if cmd is not None:
                self.data.ctrl[:8] = cmd

            # ----- decide gripper close ------------------------------------
            if (committed_catch_t is not None and
                    not catch_executed and
                    t >= committed_catch_t - 0.05):
                self.joint_tracker.close_gripper()
                catch_executed = True

            # ----- physics step --------------------------------------------
            mujoco.mj_step(self.model, self.data)

            # ----- bookkeeping ---------------------------------------------
            ball_pos = self.data.xpos[self.ball_id].copy()
            ee_pos   = self.data.xpos[self.ee_id].copy()
            dist = float(np.linalg.norm(ball_pos - ee_pos))
            self.log_t.append(t)
            self.log_gt_pos.append(ball_pos)
            self.log_ee_pos.append(ee_pos)
            self.log_ekf_pos.append(
                self.ekf.position if self.ekf.initialized()
                else np.array([np.nan]*3))
            self.log_dist.append(dist)
            if dist < min_dist:
                min_dist = dist
                min_dist_t = t

            if cfg.verbose and step % 50 == 0:
                print(f"  t={t:.3f}  ball={ball_pos.round(2)}  "
                      f"ee={ee_pos.round(2)}  d={dist*100:.1f}cm")

        wall_dt = time.perf_counter() - wall_start
        if video_writer is not None:
            video_writer.release()

        # Catch criterion: minimum distance & low relative velocity.
        ball_vel = self.data.qvel[
            int(self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]) : ][:3]
        rel_speed = float(np.linalg.norm(ball_vel))
        caught = (min_dist < CATCH_DIST_THRESH and rel_speed < CATCH_VEL_THRESH * 5)

        return RunResult(
            caught=caught,
            min_dist_m=float(min_dist),
            min_dist_t=float(min_dist_t),
            t_first_detection=t_first_detect,
            t_first_intercept=t_first_intercept,
            n_vision_frames=n_vision_frames,
            n_detections=n_detections,
            final_ball_pos=self.data.xpos[self.ball_id].tolist(),
            final_ee_pos=self.data.xpos[self.ee_id].tolist(),
            final_ball_vel=ball_vel.tolist(),
            wall_time_s=wall_dt,
        )

    def close(self):
        if self.rig is not None:
            self.rig.close()


# ================================== CLI ==================================
def parse_args():
    p = argparse.ArgumentParser(description="Catch robot — single throw")
    p.add_argument("--duration", type=float, default=1.5)
    p.add_argument("--predictor", choices=["analytical", "learned"],
                   default="analytical")
    p.add_argument("--learned-ckpt", default=None)
    p.add_argument("--vy", type=float, default=-3.0,
                   help="Initial y-velocity (negative = toward robot).")
    p.add_argument("--vz", type=float, default=0.5)
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-path", default="/tmp/run.mp4")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--use-gt", action="store_true",
                   help="Bypass rendering: feed EKF the GT ball position "
                        "with synthetic noise. Use this for fast ablation "
                        "experiments where the control pipeline is the "
                        "subject of study, not the perception front-end.")
    p.add_argument("--vision-noise", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()
    throw = ThrowConfig(
        init_pos=[0.0, 1.5, 1.7],
        init_vel=[args.vx, args.vy, args.vz],
        spin=[0.0, 0.0, 0.0],
    )
    cfg = RunConfig(
        duration_s=args.duration,
        predictor=args.predictor,
        learned_ckpt=args.learned_ckpt,
        save_video=args.save_video,
        video_path=args.video_path,
        verbose=args.verbose,
        use_gt_perception=args.use_gt,
        vision_noise_m=args.vision_noise,
        throw=throw,
    )
    runner = CatchRunner(cfg)
    result = runner.run()
    runner.close()
    print(json.dumps(result.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
