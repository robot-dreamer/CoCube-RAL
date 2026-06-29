import math
import threading
import time
from enum import Enum
from typing import List, Optional, Tuple

from cocube_udp import CoCube
from sorting_config import MotionConfig, NetworkConfig


Position = Tuple[float, float]

# Same scale as cocube_udp CoCube.pos_m and cocube_rvo2.calculate_wheel_speeds.
_MAP_TO_WHEEL_CMD = 1.35 * 0.001


def _limit_speed(speed: float, inf: float, sup: float) -> float:
    if speed > sup:
        return sup
    if speed < inf:
        return inf
    return speed


class RobotMode(str, Enum):
    IDLE = "idle"
    MOVING_TO_SOCCER = "moving_to_soccer"
    PICKING_UP = "picking_up"
    CARRYING = "carrying"
    MOVING_TO_DROPOFF = "moving_to_dropoff"
    DROPPING_OFF = "dropping_off"
    FINISHED = "finished"


class SortingRobot:
    def __init__(
        self,
        robot_id: int,
        team_id: int,
        network: NetworkConfig,
    ):
        self.robot_id = robot_id
        self.team_id = team_id
        self.node = CoCube(
            robot_id,
            gateway=network.gateway,
            local_ip=network.local_ip,
            ip_prefix=network.ip_prefix,
        )
        self.mode = RobotMode.IDLE
        self.target_position: List[float] = [0, 0]
        self.soccer_position: Optional[Position] = None
        self.dropoff_position: List[float] = [0, 0]
        self.dropoff_angle = 0
        self.active_soccer_device_id: Optional[int] = None
        self.last_completed_soccer_device_id: Optional[int] = None
        self.idle_cycles = 0
        self.action_thread: Optional[threading.Thread] = None
        self.last_motion_command_time = 0.0
        self._lock = threading.RLock()
        self._pid_integral = 0.0
        self._pid_prev_error = 0.0

    def get_position(self) -> Position:
        return self.node.pos_p[0], self.node.pos_p[1]

    def reset_wheel_controller(self) -> None:
        self._pid_integral = 0.0
        self._pid_prev_error = 0.0

    def prepare_for_sorting(self) -> None:
        print(f"[CoCube {self.robot_id}]: preparing for sorting")
        self.node.gripper_open()
        with self._lock:
            self.mode = RobotMode.IDLE
        self.reset_wheel_controller()

    def assign_soccer(
        self,
        soccer_device_id: int,
        soccer_position: Position,
        pickup_position: Position,
    ) -> None:
        with self._lock:
            self.active_soccer_device_id = soccer_device_id
            self.soccer_position = soccer_position
            self.target_position = [pickup_position[0], pickup_position[1]]
            self.last_completed_soccer_device_id = None
            self.idle_cycles = 0
            self.mode = RobotMode.MOVING_TO_SOCCER
        # self.reset_wheel_controller()

    def send_to_dropoff(self) -> None:
        with self._lock:
            self.target_position = list(self.dropoff_position)
            self.mode = RobotMode.MOVING_TO_DROPOFF
        self.reset_wheel_controller()

    def mark_finished_if_idle_too_long(self, max_idle_cycles: int) -> None:
        with self._lock:
            if self.mode != RobotMode.IDLE:
                return
            self.idle_cycles += 1
            if self.idle_cycles > max_idle_cycles:
                self.mode = RobotMode.FINISHED

    def consume_completed_soccer(self) -> Optional[int]:
        with self._lock:
            completed = self.last_completed_soccer_device_id
            self.last_completed_soccer_device_id = None
            return completed

    def rectify_angle(self, angle: float) -> float:
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _yaw_pid_controller(self, target_heading: float, motion: MotionConfig) -> float:
        error = self.rectify_angle(target_heading - self.node.yaw)
        self._pid_integral += error
        derivative = error - self._pid_prev_error
        self._pid_prev_error = error
        k = motion.pid_k
        kp = motion.pid_kp
        ki = motion.pid_ki
        kd = motion.pid_kd
        return k * (kp * error + ki * self._pid_integral + kd * derivative)

    def _calculate_vel(self, target_pos: Position, motion: MotionConfig) -> float:
        gx, gy = target_pos[0], target_pos[1]
        px_m, py_m = self.node.pos_m[0], self.node.pos_m[1]
        vx = gx - py_m
        vy = target_pos[1] - px_m
        norm = math.hypot(vx, vy)
        if norm < 1e-9:
            return 0.0
        nx, ny = vx / norm, vy / norm
        max_vel = motion.wheel_max_linear_m_s
        return (nx * math.cos(self.node.yaw) + ny * math.sin(self.node.yaw)) * max_vel

    def _set_velocity(self, linear: float, angular: float, motion: MotionConfig) -> None:
        left, right = self._calculate_wheel_speeds(linear, angular, motion)
        left_i = int(round(left))
        right_i = int(round(right))
        if left_i == 0 and right_i == 0:
            self.node.wheels_break()
        else:
            self.node.set_wheel_speed(left_i, right_i)

    def _calculate_wheel_speeds(
        self,
        linear_velocity: float,
        angular_velocity: float,
        motion: MotionConfig,
    ) -> Tuple[float, float]:
        wb = motion.wheel_base_m
        right_wheel_speed = linear_velocity + (wb * angular_velocity) / 2
        left_wheel_speed = linear_velocity - (wb * angular_velocity) / 2
        scale = _MAP_TO_WHEEL_CMD
        right_wheel_speed = _limit_speed(right_wheel_speed / scale, -50, 50)
        left_wheel_speed = _limit_speed(left_wheel_speed / scale, -50, 50)
        return left_wheel_speed, right_wheel_speed

    def _move2aim(
        self,
        target_pos: Position,
        final_pos: Position,
        motion: MotionConfig,
        arrival_distance_m: float,
    ) -> None:
        # cocube_rvo2.RVO2Robot.move2aim — target_pos / final_pos in RVO sim (meter) frame.
        # With our pos_p↔sim mapping, hypot(distance) == _MAP_TO_WHEEL_CMD * pos_p distance to goal.
        ref_theta = math.atan2(
            target_pos[1] - self.node.pos_m[0],
            target_pos[0] - self.node.pos_m[1],
        )
        vel = self._calculate_vel(target_pos, motion)
        distance = (
            final_pos[0] - self.node.pos_m[1],
            final_pos[1] - self.node.pos_m[0],
        )
        if math.hypot(distance[0], distance[1]) <= arrival_distance_m + 1e-9:
            vel = 0.0
            yaw_vel = 0.0
        else:
            yaw_vel = self._yaw_pid_controller(ref_theta, motion)
        self._set_velocity(vel, yaw_vel, motion)


    def move_towards(
        self,
        waypoint_sim: Position,
        final_goal_sim: Position,
        motion: MotionConfig,
        soccer_live_position: Position = None,
        pickup_approach_func=None,
    ) -> None:
        """
        waypoint_sim / final_goal_sim: RVO agent coordinates (meters), same as
        cocube_rvo2.Game.Step → move2aim(target_pos, final_pos).
        soccer_live_position: 当前帧最新的球位置（如果有）
        pickup_approach_func: 计算夹取点的函数（如果有）
        """
        with self._lock:
            mode = self.mode
            # 在MOVING_TO_SOCCER模式下，持续更新target_position为最新的球位置
            if mode == RobotMode.MOVING_TO_SOCCER and soccer_live_position is not None and pickup_approach_func is not None:
                # 只要球位置有变化就更新
                pickup_position = pickup_approach_func(self.get_position(), soccer_live_position, motion.gripper_offset)
                self.soccer_position = soccer_live_position
                self.target_position = [pickup_position[0], pickup_position[1]]

        # ...existing code...
        with self._lock:
            mode = self.mode

        if mode in {RobotMode.PICKING_UP, RobotMode.DROPPING_OFF}:
            return

        if mode not in {RobotMode.MOVING_TO_SOCCER, RobotMode.MOVING_TO_DROPOFF}:
            self.node.wheels_break()
            return

        robot_x, robot_y = self.get_position()
        final_px, final_py = float(self.target_position[0]), float(self.target_position[1])
        final_distance = math.hypot(final_px - robot_x, final_py - robot_y)
        threshold = (
            motion.pickup_arrival_tolerance
            if mode == RobotMode.MOVING_TO_SOCCER
            else motion.dropoff_distance
        )

        if final_distance <= threshold + 1e-3:
            self.node.wheels_break()
            self._start_arrival_action(mode, motion)
            return

        now = time.monotonic()
        if now - self.last_motion_command_time < motion.command_interval_seconds:
            return
        self.last_motion_command_time = now
        arrival_m = threshold * _MAP_TO_WHEEL_CMD
        self._move2aim(waypoint_sim, final_goal_sim, motion, arrival_m)

    def _start_arrival_action(self, mode: RobotMode, motion: MotionConfig) -> None:
        with self._lock:
            if self.action_thread is not None and not self.action_thread.is_alive():
                self.action_thread = None
            if self.mode != mode:
                return
            if self.action_thread is not None and self.action_thread.is_alive():
                return
            if mode == RobotMode.MOVING_TO_SOCCER:
                self.mode = RobotMode.PICKING_UP
                target = self._pickup_soccer
            else:
                self.mode = RobotMode.DROPPING_OFF
                target = self._dropoff_soccer

            self.action_thread = threading.Thread(target=target, args=(motion,), daemon=True)
            self.action_thread.start()

    def join_action(self, timeout: float = 1.0) -> None:
        thread = self.action_thread
        if thread is not None:
            thread.join(timeout=timeout)
            return

    def stop(self) -> None:
        self.join_action()
        self.node.wheels_stop()
        self.node.gripper_open()
        self.node.stop()

    def _pickup_soccer(self, motion: MotionConfig) -> None:
        print(f"[CoCube {self.robot_id}]: start grasping soccer {self.active_soccer_device_id}")
        time.sleep(motion.pre_grasp_wait_seconds)
        self.node.gripper_close()
        time.sleep(motion.pickup_wait_seconds)
        print(f"[CoCube {self.robot_id}]: finished grasping soccer {self.active_soccer_device_id}")
        with self._lock:
            self.mode = RobotMode.CARRYING
            self.action_thread = None

    def _dropoff_soccer(self, motion: MotionConfig) -> None:
        print(f"[CoCube {self.robot_id}]: start dropping off soccer {self.active_soccer_device_id}")
        time.sleep(0.3)
        self.node.gripper_open()
        time.sleep(motion.dropoff_wait_seconds)
        self.node.set_wheel_speed(-40, -40)
        time.sleep(0.8)
        with self._lock:
            self.last_completed_soccer_device_id = self.active_soccer_device_id
            self.active_soccer_device_id = None
            self.soccer_position = None
            self.mode = RobotMode.IDLE
            self.action_thread = None
