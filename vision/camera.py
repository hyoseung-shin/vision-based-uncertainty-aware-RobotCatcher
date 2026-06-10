from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mujoco


_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


@dataclass
class CameraModel:
    name: str
    width: int
    height: int
    K: np.ndarray
    R_wc: np.ndarray
    t_wc: np.ndarray

    @property
    def P(self) -> np.ndarray:
        Rt = np.hstack([self.R_wc, self.t_wc.reshape(3, 1)])
        return self.K @ Rt

    @property
    def cam_center_world(self) -> np.ndarray:
        return -self.R_wc.T @ self.t_wc

    def project(self, X_world: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X_world).astype(np.float64)
        X_cam = (self.R_wc @ X.T).T + self.t_wc
        z = X_cam[:, 2]
        valid = z > 1e-6
        uv = np.full((X.shape[0], 2), np.nan)
        uv[valid, 0] = self.K[0, 0] * X_cam[valid, 0] / z[valid] + self.K[0, 2]
        uv[valid, 1] = self.K[1, 1] * X_cam[valid, 1] / z[valid] + self.K[1, 2]
        return uv[0] if X_world.ndim == 1 else uv


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> np.ndarray:
    fovy_rad = np.deg2rad(fovy_deg)
    fy = 0.5 * height / np.tan(0.5 * fovy_rad)
    fx = fy
    cx = 0.5 * width
    cy = 0.5 * height
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]])


def camera_model_from_mujoco(model, data, cam_name: str,
                             width: int = 640, height: int = 480) -> CameraModel:
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise KeyError(f"Camera '{cam_name}' not found in model.")
    mujoco.mj_forward(model, data)
    cam_xpos = data.cam_xpos[cid].copy()
    R_cw_gl  = data.cam_xmat[cid].reshape(3, 3).copy()
    R_wc_gl  = R_cw_gl.T
    t_wc_gl  = -R_wc_gl @ cam_xpos
    R_wc_cv = _GL_TO_CV @ R_wc_gl
    t_wc_cv = _GL_TO_CV @ t_wc_gl
    fovy = float(model.cam_fovy[cid])
    K = intrinsics_from_fovy(fovy, width, height)
    return CameraModel(cam_name, width, height, K, R_wc_cv, t_wc_cv)


class StereoRig:
    def __init__(self, model, data,
                 left_name: str = "cam_left",
                 right_name: str = "cam_right",
                 width: int = 640, height: int = 480):
        self.model = model
        self.data = data
        self.left_name = left_name
        self.right_name = right_name
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def models(self):
        L = camera_model_from_mujoco(self.model, self.data, self.left_name,
                                     self.width, self.height)
        R = camera_model_from_mujoco(self.model, self.data, self.right_name,
                                     self.width, self.height)
        return L, R

    def render_pair(self):
        self._renderer.update_scene(self.data, camera=self.left_name)
        img_L = self._renderer.render().copy()
        self._renderer.update_scene(self.data, camera=self.right_name)
        img_R = self._renderer.render().copy()
        return img_L, img_R

    def render_overview(self, camera: int = -1):
        """Render the free 'tracking' camera for visualization."""
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self):
        try:
            self._renderer.close()
        except Exception:
            pass
