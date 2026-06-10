from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    score: float
    area_px: int
    pixel_sigma: float


class RedBallDetector:
    def __init__(self, hsv_ranges=None, min_area_px: int = 10, pixel_sigma_base: float = 1.0):
        if hsv_ranges is None:
            hsv_ranges = [
                (np.array([0, 100, 60]),   np.array([10, 255, 255])),
                (np.array([170, 100, 60]), np.array([180, 255, 255])),
            ]
        self.hsv_ranges = hsv_ranges
        self.min_area_px = min_area_px
        self.pixel_sigma_base = pixel_sigma_base

    def _mask(self, img_rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.hsv_ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def detect(self, img_rgb: np.ndarray):
        mask = self._mask(img_rgb)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.max() < self.min_area_px:
            return None
        best = 1 + int(np.argmax(areas))
        x, y, w, h, area = stats[best]
        cu, cv_ = centroids[best]
        sigma = self.pixel_sigma_base * (8.0 / max(np.sqrt(area), 4.0))
        sigma = float(np.clip(sigma, 0.5, 4.0))
        return Detection(
            bbox=(float(x), float(y), float(x + w), float(y + h)),
            center=(float(cu), float(cv_)),
            score=float(area) / float(mask.size),
            area_px=int(area),
            pixel_sigma=sigma,
        )
