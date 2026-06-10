from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .predictor import Prediction


@dataclass
class InterceptResult:
    p_catch: np.ndarray
    t_catch: float
    horizon_index: int  
    cost: float
    reachable: bool


@dataclass
class WorkspaceApprox:
    base_xy: np.ndarray = None
    r_min: float = 0.25
    r_max: float = 0.65
    z_min: float = 0.25
    z_max: float = 0.85

    def __post_init__(self):
        if self.base_xy is None:
            self.base_xy = np.array([0.0, 0.0])

    def reachable(self, p: np.ndarray) -> bool:
        dxy = np.linalg.norm(p[:2] - self.base_xy)
        d_full = np.sqrt(dxy ** 2 + (p[2] - 0.4) ** 2)
        return (self.r_min <= dxy <= self.r_max
                and self.z_min <= p[2] <= self.z_max
                and d_full <= 0.75)


class InterceptSelector:
    def __init__(self, workspace: WorkspaceApprox | None = None, alpha_time: float = 1.0, beta_unc:   float = 50.0,
                 unreachable_cost: float = 1e3, floor_z: float = 0.05, ball_radius: float = 0.035, min_lead_time: float = 0.10):
        self.workspace = workspace or WorkspaceApprox()
        self.alpha_time = alpha_time
        self.beta_unc = beta_unc
        self.unreachable_cost = unreachable_cost
        self.floor_z = floor_z
        self.ball_radius = ball_radius
        self.min_lead_time = min_lead_time

    def choose(self, pred: Prediction, t_now: float) -> InterceptResult | None:
        best = None
        best_cost = np.inf
        for i in range(len(pred.times)):
            t_h = pred.times[i]
            p_h = pred.means[i]
            Σ_h = pred.covs[i]
            dt  = t_h - t_now
            if dt < self.min_lead_time:
                continue
            if p_h[2] < self.floor_z + self.ball_radius:
                continue
            reachable = self.workspace.reachable(p_h)
            unc_term = float(np.trace(Σ_h))
            cost = (self.alpha_time * dt
                    + self.beta_unc * unc_term
                    + (0.0 if reachable else self.unreachable_cost))
            if cost < best_cost:
                best_cost = cost
                best = InterceptResult(
                    p_catch=p_h.copy(), t_catch=float(t_h),
                    horizon_index=i, cost=float(cost),
                    reachable=reachable,
                )
        return best
