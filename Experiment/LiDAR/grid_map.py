import math
from collections import defaultdict

import numpy as np

from utils import MAP_H, MAP_W, bresenham_cells, point_in_map


class OccupancyGridMap:
    def __init__(
        self,
        width=MAP_W,
        height=MAP_H,
        resolution=2.0,
        p_hit=0.70,
        p_miss=0.35,
        p_min=0.12,
        p_max=0.97,
        p_occ=0.80,
        raycast_max_range=120.0,
        inflation_radius=4.0,
    ):
        self.width = float(width)
        self.height = float(height)
        self.resolution = float(resolution)
        self.cols = int(math.ceil(self.width / self.resolution))
        self.rows = int(math.ceil(self.height / self.resolution))
        self.raycast_max_range = float(raycast_max_range)
        self.inflation_radius = float(inflation_radius)

        self.p_hit_log = self.logit(p_hit)
        self.p_miss_log = self.logit(p_miss)
        self.p_min_log = self.logit(p_min)
        self.p_max_log = self.logit(p_max)
        self.p_occ_log = self.logit(p_occ)
        self.log_odds = np.zeros((self.rows, self.cols), dtype=float)

    @staticmethod
    def logit(probability):
        probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
        return math.log(probability / (1.0 - probability))

    @staticmethod
    def sigmoid(log_odds):
        return 1.0 / (1.0 + np.exp(-log_odds))

    def reset(self):
        self.log_odds.fill(0.0)

    def update_from_scan(self, origin, points_world):
        if points_world is None or len(points_world) == 0:
            return
        if not point_in_map(origin):
            return

        origin_idx = self.world_to_grid(origin)
        hit_count = defaultdict(int)
        hit_miss_count = defaultdict(int)

        for point in np.array(points_world, dtype=float):
            endpoint, is_hit = self._bounded_endpoint(np.array(origin), point)
            end_idx = self.world_to_grid(endpoint)
            if origin_idx is None or end_idx is None:
                continue

            ray_cells = bresenham_cells(origin_idx, end_idx)
            for cell in ray_cells[:-1]:
                if not self.in_bounds(cell):
                    continue
                hit_miss_count[cell] += 1

            if self.in_bounds(end_idx):
                hit_miss_count[end_idx] += 1
                if is_hit:
                    hit_count[end_idx] += 1

        self._apply_cached_updates(hit_count, hit_miss_count)

    def is_occupied_world(self, point, inflated=True):
        idx = self.world_to_grid(point)
        if idx is None:
            return True
        mask = self.inflated_obstacle_mask() if inflated else self.obstacle_mask()
        x, y = idx
        return bool(mask[y, x])

    def path_is_occupied(self, path, inflated=True, ignore_center=None, ignore_radius=0.0):
        if path is None or len(path) == 0:
            return False

        mask = self.inflated_obstacle_mask() if inflated else self.obstacle_mask()
        points = np.array(path, dtype=float)
        for start, end in zip(points[:-1], points[1:]):
            start_idx = self.world_to_grid(start)
            end_idx = self.world_to_grid(end)
            if start_idx is None or end_idx is None:
                return True
            for cell in bresenham_cells(start_idx, end_idx):
                if not self.in_bounds(cell):
                    return True
                point = self.grid_to_world(cell)
                if ignore_center is not None:
                    if np.linalg.norm(point - ignore_center) <= ignore_radius:
                        continue
                x, y = cell
                if mask[y, x]:
                    return True
        return False

    def obstacle_mask(self):
        return self.log_odds >= self.p_occ_log

    def inflated_obstacle_mask(self):
        mask = self.obstacle_mask()
        radius_cells = int(math.ceil(self.inflation_radius / self.resolution))
        if radius_cells <= 0 or not mask.any():
            return mask

        inflated = mask.copy()
        occupied = np.argwhere(mask)
        for y, x in occupied:
            y0 = max(0, y - radius_cells)
            y1 = min(self.rows, y + radius_cells + 1)
            x0 = max(0, x - radius_cells)
            x1 = min(self.cols, x + radius_cells + 1)
            inflated[y0:y1, x0:x1] = True
        return inflated

    def probability_grid(self):
        return self.sigmoid(self.log_odds)

    def world_to_grid(self, point):
        x, y = float(point[0]), float(point[1])
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return None
        gx = min(int(x / self.resolution), self.cols - 1)
        gy = min(int(y / self.resolution), self.rows - 1)
        return gx, gy

    def grid_to_world(self, cell):
        gx, gy = cell
        return np.array(
            [(gx + 0.5) * self.resolution, (gy + 0.5) * self.resolution],
            dtype=float,
        )

    def in_bounds(self, cell):
        x, y = cell
        return 0 <= x < self.cols and 0 <= y < self.rows

    def _bounded_endpoint(self, origin, point):
        delta = point - origin
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            return origin.copy(), False

        adjusted = False
        if distance > self.raycast_max_range:
            point = origin + delta / distance * self.raycast_max_range
            distance = self.raycast_max_range
            adjusted = True

        if not point_in_map(point):
            point = self._clip_to_map(origin, point)
            adjusted = True

        return point, not adjusted

    def _clip_to_map(self, origin, point):
        direction = point - origin
        candidates = []
        for axis, limit in ((0, 0.0), (0, self.width), (1, 0.0), (1, self.height)):
            if abs(direction[axis]) < 1e-6:
                continue
            t = (limit - origin[axis]) / direction[axis]
            if 0.0 <= t <= 1.0:
                candidate = origin + t * direction
                if point_in_map(candidate):
                    candidates.append(candidate)

        if not candidates:
            return np.clip(point, [0.0, 0.0], [self.width, self.height])

        distances = [np.linalg.norm(candidate - origin) for candidate in candidates]
        return candidates[int(np.argmin(distances))]

    def _apply_cached_updates(self, hit_count, hit_miss_count):
        for cell, total in hit_miss_count.items():
            hits = hit_count[cell]
            misses = total - hits
            update = self.p_hit_log if hits >= misses and hits != 0 else self.p_miss_log
            x, y = cell
            current = self.log_odds[y, x]
            if update >= 0 and current >= self.p_max_log:
                continue
            if update <= 0 and current <= self.p_min_log:
                continue
            self.log_odds[y, x] = float(
                np.clip(current + update, self.p_min_log, self.p_max_log)
            )
