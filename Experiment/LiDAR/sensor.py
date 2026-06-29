from collections import deque
from dataclasses import dataclass
import json
import math
import time

from cocube_udp import CoCube
import numpy as np

from utils import RobotPose, lidar_to_cocube, screen_to_algo, wrap_to_pi


@dataclass
class SensorFrame:
    pose: RobotPose
    local_lidar_points: list
    world_lidar_points: np.ndarray


class LidarSensor:
    def __init__(
        self,
        robot_id=1,
        gateway="192.168.3.1",
        local_ip="192.168.3.118",
        ip_prefix=100,
        udp_port=5000,
        lidar_offset=30.1,
        range_calibration_file=None,
        keep_cloud_frames=8,
        simulated=False,
    ):
        self.simulated = simulated
        self.lidar_offset = lidar_offset
        self.range_calibration = self._load_range_calibration(range_calibration_file)
        self.recent_clouds = deque(maxlen=keep_cloud_frames)
        self.sim_pose = RobotPose(np.array([30.0, 30.0], dtype=float), 0.0)
        self.last_frame = SensorFrame(self.sim_pose, [], np.empty((0, 2), dtype=float))

        self.agent = None
        if not self.simulated:
            self.agent = CoCube(
                robot_id,
                gateway=gateway,
                local_ip=local_ip,
                ip_prefix=ip_prefix,
                udp_port=udp_port,
            )

    def read(self):
        pose = self.get_pose()
        local_points = self._read_local_lidar_points()
        world_points = self.local_lidar_to_world(local_points, pose)
        if len(world_points) > 0:
            self.recent_clouds.append(world_points)

        self.last_frame = SensorFrame(pose, local_points, world_points)
        return self.last_frame

    def get_pose(self):
        if self.simulated or self.agent is None:
            return self.sim_pose

        pos_screen = self.agent.get_pos()
        ax, ay = screen_to_algo(pos_screen[0], pos_screen[1])
        yaw = wrap_to_pi(self.agent.get_yaw() - np.pi / 2)
        return RobotPose(np.array([ax, ay], dtype=float), yaw)

    def set_wheel_speed(self, left_speed, right_speed):
        if self.agent is not None:
            self.agent.set_wheel_speed(left_speed, right_speed)
        else:
            self._integrate_simulated_motion(left_speed, right_speed)

    def wheels_stop(self):
        if self.agent is not None:
            self.agent.wheels_stop()

    def stop(self):
        if self.agent is not None:
            self.agent.wheels_stop()
            self.agent.stop()

    def _read_local_lidar_points(self):
        if self.agent is None:
            return []
        points = self.agent.get_lidar_points()
        return [] if points is None else points

    def local_lidar_to_world(self, local_points, pose):
        if not local_points:
            return np.empty((0, 2), dtype=float)

        world_points = []
        cos_yaw = math.cos(pose.yaw)
        sin_yaw = math.sin(pose.yaw)
        for theta, radius in local_points:
            theta = lidar_to_cocube(theta)
            radius = self.correct_range(radius)
            lx = (radius + self.lidar_offset) * math.cos(theta)
            ly = (radius + self.lidar_offset) * math.sin(theta)
            gx = pose.position[0] + lx * cos_yaw - ly * sin_yaw
            gy = pose.position[1] + lx * sin_yaw + ly * cos_yaw
            world_points.append((gx, gy))

        return np.array(world_points, dtype=float)

    def _integrate_simulated_motion(self, left_speed, right_speed):
        dt = 0.05
        linear = 0.5 * (left_speed + right_speed) * 0.2
        angular = (right_speed - left_speed) * 0.015
        self.sim_pose.yaw = wrap_to_pi(self.sim_pose.yaw + angular * dt)
        self.sim_pose.position += linear * dt * np.array(
            [math.cos(self.sim_pose.yaw), math.sin(self.sim_pose.yaw)]
        )
        time.sleep(dt)

    def correct_range(self, radius):
        if self.range_calibration is None:
            return radius
        corrected = float(np.polyval(self.range_calibration, radius))
        return max(corrected, 0.0)

    def _load_range_calibration(self, calibration_file):
        if not calibration_file:
            return None
        with open(calibration_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        coeffs = data.get("coefficients_high_to_low")
        if not coeffs:
            raise ValueError(
                f"{calibration_file} does not contain coefficients_high_to_low"
            )
        return np.array(coeffs, dtype=float)
