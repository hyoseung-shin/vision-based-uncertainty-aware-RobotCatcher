from __future__ import annotations
import numpy as np

_H = np.zeros((3, 9))
_H[:3, :3] = np.eye(3)


class BallEKF:
    def __init__(self, sigma_a: float = 6.0, sigma_init_p: float = 0.10, sigma_init_v: float = 1.0, sigma_init_a: float = 10.0):
        self.sigma_a = sigma_a
        self.x = np.zeros(9)
        self.P = np.diag([sigma_init_p**2] * 3 + [sigma_init_v**2] * 3 + [sigma_init_a**2] * 3)
        self.t_last: float | None = None
        self.n_updates = 0

    def initialized(self) -> bool:
        return self.t_last is not None

    def initialize(self, t: float, p_meas: np.ndarray,  v_init: np.ndarray | None = None) -> None:
        self.x[:3] = p_meas
        self.x[3:6] = np.zeros(3) if v_init is None else v_init
        self.x[6:9] = 0.0
        self.t_last = t
        self.n_updates = 0

    def predict(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if self.t_last is None:
            raise RuntimeError("EKF not initialized.")
        dt = t - self.t_last
        if dt < 0:
            raise ValueError(f"Negative dt = {dt}")
        F, Q = self._F_and_Q(dt)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        return x_pred, P_pred

    def update(self, t: float, p_meas: np.ndarray, cov_meas: np.ndarray) -> None:
        x_pred, P_pred = self.predict(t)
        innov = p_meas - _H @ x_pred
        S = _H @ P_pred @ _H.T + cov_meas
        K = P_pred @ _H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ innov
        self.P = (np.eye(9) - K @ _H) @ P_pred
        self.P = 0.5 * (self.P + self.P.T)
        self.t_last = t
        self.n_updates += 1

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def acceleration(self) -> np.ndarray:
        return self.x[6:9].copy()

    def _F_and_Q(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        F = np.block([
            [I3, I3 * dt, I3 * 0.5 * dt * dt],
            [Z3, I3,      I3 * dt           ],
            [Z3, Z3,      I3                ],
        ])
        q = self.sigma_a ** 2
        dt2, dt3, dt4, dt5 = dt**2, dt**3, dt**4, dt**5
        Q = q * np.block([
            [I3 * dt5/20, I3 * dt4/8, I3 * dt3/6],
            [I3 * dt4/8,  I3 * dt3/3, I3 * dt2/2],
            [I3 * dt3/6,  I3 * dt2/2, I3 * dt   ],
        ])
        return F, Q
