from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Track:
    track_id: int
    t_history: list[float] = field(default_factory=list)
    p_history: list[np.ndarray] = field(default_factory=list)
    cov_history: list[np.ndarray] = field(default_factory=list)
    last_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    misses: int = 0

    @property
    def length(self) -> int:
        return len(self.p_history)

    @property
    def last_pos(self) -> np.ndarray:
        return self.p_history[-1]

    @property
    def last_time(self) -> float:
        return self.t_history[-1]


class BallTracker:
    def __init__(self, gate_chi2: float = 27.0, max_misses: int = 6):
        self.gate_chi2 = gate_chi2
        self.max_misses = max_misses
        self._track: Track | None = None
        self._next_id = 0

    @property
    def active(self) -> Track | None:
        return self._track

    def step(self, t: float, p_meas: np.ndarray | None, cov_meas: np.ndarray | None) -> Track | None:
        if p_meas is None:
            if self._track is not None:
                self._track.misses += 1
                if self._track.misses > self.max_misses:
                    self._track = None
            return self._track

        if self._track is None:
            self._track = self._seed(t, p_meas, cov_meas)
            return self._track

        dt = t - self._track.last_time
        if dt <= 0:
            return self._associate_and_update(t, p_meas, cov_meas)

        p_pred = self._track.last_pos + self._track.last_velocity * dt
        cov_pred = cov_meas + np.eye(3) * (0.01 * dt) ** 2

        innov = p_meas - p_pred
        try:
            md2 = float(innov @ np.linalg.solve(cov_pred, innov))
        except np.linalg.LinAlgError:
            md2 = np.inf

        if md2 > self.gate_chi2:
            self._track.misses += 1
            if self._track.misses > self.max_misses:
                self._track = None
            return self._track

        return self._associate_and_update(t, p_meas, cov_meas)

    def _seed(self, t, p_meas, cov_meas):
        tid = self._next_id
        self._next_id += 1
        tr = Track(track_id=tid)
        tr.t_history.append(t)
        tr.p_history.append(p_meas.copy())
        tr.cov_history.append(cov_meas.copy())
        return tr

    def _associate_and_update(self, t, p_meas, cov_meas):
        tr = self._track
        if len(tr.p_history) >= 1:
            dt = max(t - tr.last_time, 1e-6)
            tr.last_velocity = (p_meas - tr.last_pos) / dt
        tr.t_history.append(t)
        tr.p_history.append(p_meas.copy())
        tr.cov_history.append(cov_meas.copy())
        tr.misses = 0
        return tr
