from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class NetworkConfig:
    gateway: str = "192.168.3.1"
    local_ip: str = "192.168.3.118"
    ip_prefix: int = 100


@dataclass(frozen=True)
class MotionConfig:
    gripper_offset: float = 39.0
    pickup_arrival_tolerance: float = 8.0
    dropoff_distance: float = 10.0
    waypoint_distance: float = 10.0
    command_lookahead_distance: float = 35.0
    command_interval_seconds: float = 0.03
    command_speed: int = 30
    pre_grasp_wait_seconds: float = 0.5
    wheel_max_linear_m_s: float = 0.05
    wheel_base_m: float = 0.0255
    pid_k: float = 1.2
    pid_kp: float = 1.8
    pid_ki: float = 0.0
    pid_kd: float = 0.5
    pickup_wait_seconds: float = 1.0
    dropoff_wait_seconds: float = 0.8


@dataclass(frozen=True)
class RvoConfig:
    # Matches cocube_rvo2.py PyRVOSimulator (world in meters, swapped x/y vs pos_p).
    time_step: float = 0.5
    neighbor_distance: float = 0.07
    max_neighbors: int = 3
    time_horizon: float = 10.0
    obstacle_time_horizon: float = 4.0
    robot_radius: float = 0.06
    max_speed: float = 0.1
    # setAgentPrefVelocity scale v0, same role as cocube_rvo2.Game.setAgentVelocity(speed=v0)
    pref_velocity_gain: float = 5.0
    # doStep repetitions per control cycle (cocube_rvo2 uses 5).
    substeps: int = 5


@dataclass(frozen=True)
class SortingConfig:
    team_count: int = 4
    robot_count: int = 8
    soccer_count: int = 8
    map_size: Tuple[int, int] = (500, 500)
    soccer_start_id: int = 21
    sorted_margin: int = 30
    dropoff_margin: int = 50
    dropoff_spacing: int = 60
    dropoff_jitter: int = 35
    dropoff_min_distance: int = 40
    max_idle_cycles_before_finish: int = 1000
    random_seed: Optional[int] = 7

    def validate(self) -> None:
        if self.team_count <= 0:
            raise ValueError("team_count must be greater than 0")
        if self.robot_count <= 0:
            raise ValueError("robot_count must be greater than 0")
        if self.soccer_count <= 0:
            raise ValueError("soccer_count must be greater than 0")
        if self.robot_count % self.team_count != 0:
            raise ValueError("robot_count must be divisible by team_count")
        if len(self.map_size) != 2 or self.map_size[0] <= 0 or self.map_size[1] <= 0:
            raise ValueError("map_size must be a positive (width, height) tuple")
