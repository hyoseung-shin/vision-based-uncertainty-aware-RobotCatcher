from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

G_VEC = np.array([0.0, 0.0, -9.81])
RHO_AIR = 1.204                         
BALL_R = 0.035                         
BALL_M = 0.058                       
BALL_AREA = np.pi * BALL_R ** 2     
CD_DEFAULT = 0.47                    


@dataclass
class Prediction:
    times: np.ndarray  
    means: np.ndarray  
    covs:  np.ndarray

class AnalyticalPredictor:
    def __init__(self, drag_cd: float = CD_DEFAULT, mass: float = BALL_M, area: float = BALL_AREA, horizon_s: float = 0.6, step_s: float = 1.0 / 60.0):
        self.k_drag = 0.5 * RHO_AIR * drag_cd * area / mass   # 1/m coefficient
        self.horizon_s = horizon_s
        self.step_s = step_s

    def _accel(self, v: np.ndarray) -> np.ndarray:
        return G_VEC - self.k_drag * np.linalg.norm(v) * v

    def _rk4_step(self, p: np.ndarray, v: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
        k1v = self._accel(v);            k1p = v
        k2v = self._accel(v + 0.5*h*k1v); k2p = v + 0.5*h*k1v
        k3v = self._accel(v + 0.5*h*k2v); k3p = v + 0.5*h*k2v
        k4v = self._accel(v + h*k3v);     k4p = v + h*k3v
        p_next = p + (h / 6.0) * (k1p + 2*k2p + 2*k3p + k4p)
        v_next = v + (h / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        return p_next, v_next

    def predict(self, t0: float, p0: np.ndarray, v0: np.ndarray, P0_pos: np.ndarray, P0_vel: np.ndarray,
                horizon_s: float | None = None, step_s: float | None = None) -> Prediction:
        H_s = self.horizon_s if horizon_s is None else horizon_s
        dt = self.step_s if step_s is None else step_s
        n_steps = int(H_s / dt)
        times = t0 + np.arange(1, n_steps + 1) * dt
        means = np.zeros((n_steps, 3))
        covs  = np.zeros((n_steps, 3, 3))

        p, v = p0.copy(), v0.copy()
        S = np.block([
            [P0_pos,          np.zeros((3, 3))],
            [np.zeros((3, 3)), P0_vel         ],
        ])

        for i in range(n_steps):
            p, v = self._rk4_step(p, v, dt)
            F = np.eye(6)
            F[:3, 3:6] = np.eye(3) * dt
            S = F @ S @ F.T
            S[:3, :3] += np.eye(3) * (0.05 * dt) ** 2
            means[i] = p
            covs[i]  = S[:3, :3]
        return Prediction(times=times, means=means, covs=covs)


class LearnedPredictor:
    def __init__(self, ckpt_path: str | Path | None = None, horizon_s: float = 0.6, step_s: float = 1.0 / 60.0, window: int = 5):
        self.horizon_s = horizon_s
        self.step_s = step_s
        self.window = window
        self.n_horizons = int(horizon_s / step_s)
        self._fallback = AnalyticalPredictor(horizon_s=horizon_s, step_s=step_s)
        self._model = None
        self._ckpt_path = Path(ckpt_path) if ckpt_path else None
        if self._ckpt_path and self._ckpt_path.exists():
            self._try_load()

    def _try_load(self) -> None:
        try:
            import torch  # noqa
            from .predictor_model import LSTMRegressor
            ckpt = torch.load(self._ckpt_path, map_location="cpu", weights_only=False)
            model = LSTMRegressor(input_dim=ckpt["input_dim"], hidden=ckpt["hidden"], n_horizons=ckpt["n_horizons"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self._model = model
            self._mean_z = ckpt.get("mean_z")
            self._std_z  = ckpt.get("std_z")
        except Exception:
            self._model = None

    def predict(self, t0: float, p_history: np.ndarray, v_history: np.ndarray, P0_pos: np.ndarray, P0_vel: np.ndarray) -> Prediction:
        if self._model is None:
            return self._fallback.predict(
                t0, p_history[-1], v_history[-1], P0_pos, P0_vel,
                horizon_s=self.horizon_s, step_s=self.step_s,
            )

        import torch
        seq = np.concatenate(
            [p_history[-self.window:], v_history[-self.window:]], axis=-1
        )
        if seq.shape[0] < self.window:              
            pad = np.repeat(seq[:1], self.window - seq.shape[0], axis=0)
            seq = np.concatenate([pad, seq], axis=0)
        z = ((seq - self._mean_z) / self._std_z).astype(np.float32)
        with torch.no_grad():
            mu, log_sigma = self._model(torch.from_numpy(z).unsqueeze(0))
        mu = mu.squeeze(0).numpy()                      
        sigma = np.exp(log_sigma.squeeze(0).numpy())   
        times = t0 + np.arange(1, self.n_horizons + 1) * self.step_s
        covs = np.stack([np.diag(s ** 2) for s in sigma], axis=0)
        return Prediction(times=times, means=mu, covs=covs)
