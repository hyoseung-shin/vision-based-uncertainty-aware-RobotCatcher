from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import mujoco

from vision.ekf import BallEKF
from vision.predictor import AnalyticalPredictor
from vision.tracker import BallTracker
from vision.intercept import InterceptSelector, WorkspaceApprox
from planning.ik import FrankaIK


SCENE = str(ROOT / "assets" / "franka_emika_panda" / "catch_scene.xml")


def test_ekf_recovers_gravity():
    ekf = BallEKF(sigma_a=4.0)
    rng = np.random.default_rng(0)
    p0 = np.array([0.0, 1.5, 1.7])
    v0 = np.array([0.0, -3.0, 0.5])
    R = np.diag([1e-4, 5e-3, 1e-4])
    for k, t in enumerate(np.linspace(0, 0.7, 43)):
        p = p0 + v0 * t + 0.5 * np.array([0, 0, -9.81]) * t ** 2
        z = p + rng.normal(scale=0.01, size=3)
        if not ekf.initialized():
            ekf.initialize(t, z)
        else:
            ekf.update(t, z, R)
    az = ekf.acceleration[2]
    ok = -11.0 < az < -8.5
    print(f"  EKF gravity estimate: a_z = {az:.2f}  (expected ≈ -9.81)  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_predictor_matches_truth():
    pred = AnalyticalPredictor(horizon_s=0.5)
    p0 = np.array([0.0, 1.0, 1.5])
    v0 = np.array([0.0, -2.5, 0.3])
    res = pred.predict(0.0, p0, v0,
                       np.eye(3) * 1e-6, np.eye(3) * 1e-4)
    t_end = res.times[-1]
    p_truth = p0 + v0 * t_end + 0.5 * np.array([0, 0, -9.81]) * t_end ** 2
    err = float(np.linalg.norm(res.means[-1] - p_truth)) * 1000
    ok = err < 30.0
    print(f"  Predictor end-of-horizon error: {err:.1f} mm   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_tracker_associates():
    tr = BallTracker(gate_chi2=27.0)
    rng = np.random.default_rng(0)
    sigma = np.eye(3) * 1e-4
    p = np.array([0.0, 1.5, 1.7]); v = np.array([0.0, -3.0, 0.5])
    for k in range(5):
        t = 0.02 * k
        p_k = p + v * t + rng.normal(scale=0.005, size=3)
        tr.step(t, p_k, sigma)
    track = tr.step(0.5, p + np.array([2.0, 2.0, 2.0]), sigma)
    inliers_only = (track is not None and track.misses > 0)
    print(f"  Tracker rejects outlier: misses={track.misses if track else 'n/a'}  "
          f"{'PASS' if inliers_only else 'FAIL'}")
    return inliers_only


def test_ik_reaches_known_targets():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data,
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "scene_ready"))
    mujoco.mj_forward(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    ik = FrankaIK(model)
    targets = [np.array(t) for t in [[0.4, 0.0, 0.6], [0.5, 0.2, 0.9], [0.3, -0.2, 0.8], [0.55, 0.3, 0.5], [0.0, 0.5, 0.8]]]
    failed = []
    for tgt in targets:
        q, err, ok = ik.solve(data, tgt)
        d2 = mujoco.MjData(model); d2.qpos[:] = data.qpos; d2.qpos[:7] = q
        mujoco.mj_forward(model, d2)
        real_err = float(np.linalg.norm(d2.xpos[ee_id] - tgt))
        if not ok or real_err > 0.005:
            failed.append((tgt, real_err))
    ok = len(failed) == 0
    print(f"  IK reaches 5 targets sub-cm: "
          f"{5 - len(failed)}/5  {'PASS' if ok else 'FAIL'}")
    return ok


def test_intercept_picks_feasible():
    pred = AnalyticalPredictor(horizon_s=0.6).predict(
        0.1, np.array([0.0, 1.2, 1.7]), np.array([0.0, -3.0, -0.48]),
        np.eye(3) * 1e-4, np.eye(3) * 1e-2)
    sel = InterceptSelector(workspace=WorkspaceApprox())
    res = sel.choose(pred, 0.1)
    ok = res is not None and res.reachable
    print(f"  Intercept finds a reachable catch point: "
          f"{'PASS' if ok else 'FAIL'}")
    if ok:
        print(f"    chosen p = {res.p_catch.round(3)}  t = {res.t_catch:.3f}")
    return ok


def main():
    tests = [test_ekf_recovers_gravity,
             test_predictor_matches_truth,
             test_tracker_associates,
             test_ik_reaches_known_targets,
             test_intercept_picks_feasible]
    print(f"Running {len(tests)} module tests:")
    results = [t() for t in tests]
    n_pass = sum(results)
    print(f"\n{n_pass}/{len(tests)} tests passed.")
    sys.exit(0 if n_pass == len(tests) else 1)


if __name__ == "__main__":
    main()
