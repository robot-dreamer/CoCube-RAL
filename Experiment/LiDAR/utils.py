from dataclasses import dataclass
import math

import numpy as np


MAP_W, MAP_H = 300, 200
SCALE = 5
FPS = 20


@dataclass
class RobotPose:
    position: np.ndarray
    yaw: float


@dataclass
class TrajectorySample:
    position: np.ndarray
    velocity: np.ndarray
    done: bool


@dataclass
class TimedTrajectory:
    path: np.ndarray
    times: np.ndarray

    @classmethod
    def empty(cls):
        return cls(np.empty((0, 2), dtype=float), np.array([], dtype=float))

    @classmethod
    def from_path(cls, path, speed):
        path = np.array(path, dtype=float)
        if len(path) < 2:
            return cls.empty()

        distances = cumulative_lengths(path)
        speed = max(float(speed), 1.0)
        return cls(path, distances / speed)

    @property
    def total_time(self):
        if len(self.times) == 0:
            return 0.0
        return float(self.times[-1])

    def sample(self, elapsed):
        if len(self.path) == 0:
            return TrajectorySample(np.zeros(2), np.zeros(2), True)
        if len(self.path) == 1 or elapsed >= self.total_time:
            return TrajectorySample(
                self.path[-1],
                self._segment_velocity(len(self.path) - 2),
                True,
            )

        elapsed = max(float(elapsed), 0.0)
        idx = int(np.searchsorted(self.times, elapsed, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.path) - 2))
        t0, t1 = self.times[idx], self.times[idx + 1]
        alpha = 0.0 if t1 <= t0 else (elapsed - t0) / (t1 - t0)
        position = self.path[idx] + alpha * (self.path[idx + 1] - self.path[idx])
        return TrajectorySample(position, self._segment_velocity(idx), False)

    def _segment_velocity(self, idx):
        if len(self.path) < 2:
            return np.zeros(2)
        idx = int(np.clip(idx, 0, len(self.path) - 2))
        dt = max(self.times[idx + 1] - self.times[idx], 1e-6)
        return (self.path[idx + 1] - self.path[idx]) / dt


def screen_to_algo(sx, sy):
    return sx, MAP_H - sy


def algo_to_screen(ax, ay):
    return ax, MAP_H - ay


def algo_to_pygame(ax, ay, scale=SCALE):
    sx, sy = algo_to_screen(ax, ay)
    return int(sx * scale), int(sy * scale)


def pygame_to_algo(px, py, scale=SCALE):
    sx = np.clip(px / scale, 0, MAP_W)
    sy = np.clip(py / scale, 0, MAP_H)
    return np.array(screen_to_algo(sx, sy), dtype=float)


def lidar_to_cocube(theta):
    return -theta + np.pi / 2


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def clamp_speed(speed):
    return int(np.clip(round(speed), -50, 50))


def smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def cumulative_lengths(path):
    path = np.array(path, dtype=float)
    if len(path) == 0:
        return np.array([], dtype=float)

    lengths = np.zeros(len(path), dtype=float)
    if len(path) > 1:
        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        lengths[1:] = np.cumsum(segment_lengths)
    return lengths


def bresenham_cells(start, end):
    x0, y0 = map(int, start)
    x1, y1 = map(int, end)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    cells = []

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return cells


def point_in_map(point):
    return 0 <= point[0] <= MAP_W and 0 <= point[1] <= MAP_H


def yaw_to_vector(yaw):
    return np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
