from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import mujoco
import cv2

from vision.camera import StereoRig
from vision.detector import RedBallDetector
from vision.stereo import triangulate, reprojection_error
from scripts.throw_ball import throw_ball


SCENE = str(ROOT / "assets" / "franka_emika_panda" / "catch_scene.xml")
PHYSICS_HZ = 500
VISION_EVERY = 8
DURATION_S = 1.0


def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data,
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "scene_ready"),
    )
    throw_ball(model, data,
               init_pos=np.array([0.0, 1.5, 1.7]),
               init_vel=np.array([0.0, -3.0, 0.5]))
    mujoco.mj_forward(model, data)
    rig = StereoRig(model, data)
    detector = RedBallDetector(pixel_sigma_base=1.0)

    n_steps = int(DURATION_S * PHYSICS_HZ)
    errs = []
    for step in range(n_steps):
        mujoco.mj_step(model, data)
        if step % VISION_EVERY != 0:
            continue
        img_L, img_R = rig.render_pair()
        cam_L, cam_R = rig.models()
        dL = detector.detect(img_L); dR = detector.detect(img_R)
        gt = data.sensor("ball_pos_gt").data.copy()
        if dL is None or dR is None:
            continue
        X, _ = triangulate(dL.center, dR.center, cam_L, cam_R, dL.pixel_sigma, dR.pixel_sigma)
        errs.append(float(np.linalg.norm(X - gt)))

    errs = np.array(errs)
    print(f"Frames with valid stereo detection: {len(errs)}")
    print(f"3D position error:")
    print(f"  mean   = {errs.mean() * 1000:.1f} mm")
    print(f"  median = {np.median(errs) * 1000:.1f} mm")
    print(f"  max    = {errs.max()   * 1000:.1f} mm")
    print("PASS" if errs.mean() < 0.020 else "FAIL")
    rig.close()


if __name__ == "__main__":
    main()
