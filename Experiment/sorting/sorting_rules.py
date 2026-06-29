import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from sorting_config import SortingConfig


Position = Tuple[float, float]


@dataclass(frozen=True)
class SoccerTarget:
    logical_id: int
    device_id: int
    position: Position


def soccer_device_id(logical_id: int, soccer_start_id: int = 21) -> int:
    return soccer_start_id + logical_id - 1


def owner_robot_id(logical_soccer_id: int, robot_ids: Sequence[int]) -> int:
    owner_index = (logical_soccer_id - 1) % len(robot_ids)
    return robot_ids[owner_index]


def build_robot_soccer_plan(
    robot_ids: Sequence[int],
    soccer_count: int,
) -> Dict[int, List[int]]:
    plan = {robot_id: [] for robot_id in robot_ids}
    for logical_id in range(1, soccer_count + 1):
        plan[owner_robot_id(logical_id, robot_ids)].append(logical_id)
    return plan


def team_id_for_robot_order(robot_order_index: int, config: SortingConfig) -> int:
    robots_per_team = config.robot_count // config.team_count
    return (robot_order_index // robots_per_team) + 1


def team_theta(team_id: int, team_count: int) -> float:
    return math.pi - 2 * math.pi * (team_id - 1) / team_count


def team_dropoff_anchor(team_id: int, config: SortingConfig) -> Position:
    width, height = config.map_size
    center_x = width / 2
    center_y = height / 2
    theta = team_theta(team_id, config.team_count)
    direction_x = math.cos(theta)
    direction_y = math.sin(theta)

    if abs(direction_x) < 1e-6:
        scale_x = float("inf")
    elif direction_x > 0:
        scale_x = (width - config.dropoff_margin - center_x) / direction_x
    else:
        scale_x = (config.dropoff_margin - center_x) / direction_x

    if abs(direction_y) < 1e-6:
        scale_y = float("inf")
    elif direction_y > 0:
        scale_y = (height - config.dropoff_margin - center_y) / direction_y
    else:
        scale_y = (config.dropoff_margin - center_y) / direction_y

    scale = min(scale_x, scale_y)
    return center_x + direction_x * scale, center_y + direction_y * scale


def team_dropoff_angle(team_id: int, team_count: int) -> int:
    theta = team_theta(team_id, team_count)
    return round((90 - math.degrees(theta)) % 360)


def is_sorted_position(position: Iterable[float], config: SortingConfig) -> bool:
    x, y = position
    width, height = config.map_size
    inside_work_area = (
        config.sorted_margin <= x <= width - config.sorted_margin
        and config.sorted_margin <= y <= height - config.sorted_margin
    )
    inside_map = 0 <= x <= width and 0 <= y <= height
    return (not inside_work_area) or (not inside_map)
    


def clamp_position(position: Position, config: SortingConfig) -> Position:
    width, height = config.map_size
    x = min(max(position[0], config.dropoff_margin), width - config.dropoff_margin)
    y = min(max(position[1], config.dropoff_margin), height - config.dropoff_margin)
    return round(x), round(y)


def distance(a: Position, b: Position) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pickup_approach_position(
    robot_position: Position,
    soccer_position: Position,
    gripper_offset: float,
) -> Position:
    dx = soccer_position[0] - robot_position[0]
    dy = soccer_position[1] - robot_position[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return soccer_position

    unit_x = dx / length
    unit_y = dy / length
    return (
        soccer_position[0] - unit_x * gripper_offset,
        soccer_position[1] - unit_y * gripper_offset,
    )


def build_dropoff_positions(
    robot_ids: Sequence[int],
    team_by_robot: Dict[int, int],
    config: SortingConfig,
) -> Dict[int, Position]:
    rng = random.Random(config.random_seed)
    positions: Dict[int, Position] = {}

    for team_id in range(1, config.team_count + 1):
        team_robot_ids = [robot_id for robot_id in robot_ids if team_by_robot[robot_id] == team_id]
        anchor_x, anchor_y = team_dropoff_anchor(team_id, config)
        theta = team_theta(team_id, config.team_count)
        tangent_x = -math.sin(theta)
        tangent_y = math.cos(theta)

        accepted: List[Position] = []
        for index, robot_id in enumerate(team_robot_ids):
            base_offset = (index - (len(team_robot_ids) - 1) / 2) * config.dropoff_spacing
            base = [anchor_x + tangent_x * base_offset, anchor_y + tangent_y * base_offset]
            base[0] += 50
            base[1] += 50
            candidate = _sample_dropoff_position(base, accepted, rng, config)
            positions[robot_id] = candidate
            accepted.append(candidate)

    return positions


def _sample_dropoff_position(
    base: Position,
    accepted: Sequence[Position],
    rng: random.Random,
    config: SortingConfig,
) -> Position:
    for _ in range(100):
        candidate = clamp_position(
            (
                base[0] + rng.uniform(-config.dropoff_jitter, config.dropoff_jitter),
                base[1] + rng.uniform(-config.dropoff_jitter, config.dropoff_jitter),
            ),
            config,
        )
        if all(distance(candidate, other) >= config.dropoff_min_distance for other in accepted):
            return candidate

    return clamp_position(base, config)
