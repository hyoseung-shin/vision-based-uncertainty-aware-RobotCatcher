from __future__ import annotations
import numpy as np
from .camera import CameraModel


def _dlt_solve(uv_L, uv_R, P_L, P_R):
    uL, vL = uv_L
    uR, vR = uv_R
    A = np.vstack([
        uL * P_L[2, :] - P_L[0, :],
        vL * P_L[2, :] - P_L[1, :],
        uR * P_R[2, :] - P_R[0, :],
        vR * P_R[2, :] - P_R[1, :],
    ])
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if abs(X_h[3]) < 1e-12:
        raise ValueError("Degenerate triangulation.")
    return X_h[:3] / X_h[3]


def triangulate(uv_L, uv_R, cam_L: CameraModel, cam_R: CameraModel, sigma_L: float = 1.0, sigma_R: float = 1.0):
    P_L = cam_L.P
    P_R = cam_R.P
    uv_L_arr = np.asarray(uv_L, dtype=np.float64)
    uv_R_arr = np.asarray(uv_R, dtype=np.float64)
    X = _dlt_solve(uv_L_arr, uv_R_arr, P_L, P_R)

    h = 1e-3
    J = np.zeros((3, 4))
    for i, base in enumerate([(uv_L_arr, 0), (uv_L_arr, 1), (uv_R_arr, 0), (uv_R_arr, 1)]):
        arr, idx = base
        plus  = arr.copy(); plus[idx]  += h
        minus = arr.copy(); minus[idx] -= h
        if i < 2:
            X_plus  = _dlt_solve(plus,  uv_R_arr, P_L, P_R)
            X_minus = _dlt_solve(minus, uv_R_arr, P_L, P_R)
        else:
            X_plus  = _dlt_solve(uv_L_arr, plus,  P_L, P_R)
            X_minus = _dlt_solve(uv_L_arr, minus, P_L, P_R)
        J[:, i] = (X_plus - X_minus) / (2 * h)

    Sigma_pix = np.diag([sigma_L**2, sigma_L**2, sigma_R**2, sigma_R**2])
    Sigma_X = J @ Sigma_pix @ J.T
    Sigma_X = 0.5 * (Sigma_X + Sigma_X.T)
    return X, Sigma_X


def reprojection_error(X_world, uv_L, uv_R, cam_L: CameraModel, cam_R: CameraModel):
    uv_L_pred = cam_L.project(X_world)
    uv_R_pred = cam_R.project(X_world)
    eL = float(np.linalg.norm(uv_L_pred - np.asarray(uv_L)))
    eR = float(np.linalg.norm(uv_R_pred - np.asarray(uv_R)))
    return eL, eR
