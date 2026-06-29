import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from cocube_udp import Soccer
from robot_controller import RobotMode, SortingRobot
from sorting_config import MotionConfig, NetworkConfig, RvoConfig, SortingConfig
from sorting_rules import (
    SoccerTarget,
    build_dropoff_positions,
    build_robot_soccer_plan,
    distance,
    is_sorted_position,
    pickup_approach_position,
    soccer_device_id,
    team_dropoff_angle,
    team_id_for_robot_order,
)


Position = Tuple[float, float]

# Same meters-per-map-unit as cocube_udp CoCube.pos_m (= pos_p * 1.35e-3).
# RVO runs in (sim_x, sim_y) = (pos_p_y * k, pos_p_x * k) like cocube_rvo2.Game.play().
POS_M_SCALE = 1.35 * 0.001


def pos_p_to_rvo_sim(p: Position) -> Position:
    k = POS_M_SCALE
    return (p[1] * k, p[0] * k)


class MultiAgentSortingGame:
    def __init__(
        self,
        sorting: Optional[SortingConfig] = None,
        network: Optional[NetworkConfig] = None,
        motion: Optional[MotionConfig] = None,
        rvo: Optional[RvoConfig] = None,
        robot_ids: Optional[Sequence[int]] = None,
    ):
        self.sorting = sorting or SortingConfig()
        self.sorting.validate()
        self.network = network or NetworkConfig()
        self.motion = motion or MotionConfig()
        self.rvo = rvo or RvoConfig()

        self.robot_ids = list(robot_ids or range(1, self.sorting.robot_count + 1))
        if len(self.robot_ids) != self.sorting.robot_count:
            raise ValueError("robot_ids length must match sorting.robot_count")

        self.team_by_robot = {
            robot_id: team_id_for_robot_order(index, self.sorting)
            for index, robot_id in enumerate(self.robot_ids)
        }
        self.soccer_plan = build_robot_soccer_plan(
            self.robot_ids,
            self.sorting.soccer_count,
        )
        self.dropoff_by_robot = build_dropoff_positions(self.robot_ids, self.team_by_robot, self.sorting)
        self.robots = [
            SortingRobot(robot_id, self.team_by_robot[robot_id], self.network)
            for robot_id in self.robot_ids
        ]
        self.robot_by_id = {robot.robot_id: robot for robot in self.robots}
        for robot in self.robots:
            robot.dropoff_position = list(self.dropoff_by_robot[robot.robot_id])
            robot.dropoff_angle = team_dropoff_angle(robot.team_id, self.sorting.team_count)

        self.soccers = {
            logical_id: Soccer(
                soccer_device_id(logical_id, self.sorting.soccer_start_id),
                gateway=self.network.gateway,
                local_ip=self.network.local_ip,
                ip_prefix=self.network.ip_prefix,
            )
            for logical_id in range(1, self.sorting.soccer_count + 1)
        }
        rvo2 = load_rvo2()
        self.simulator = rvo2.PyRVOSimulator(
            self.rvo.time_step,
            self.rvo.neighbor_distance,
            self.rvo.max_neighbors,
            self.rvo.time_horizon,
            self.rvo.obstacle_time_horizon,
            self.rvo.robot_radius,
            self.rvo.max_speed,
        )
        self.agent_ids: List[int] = []
        self.claimed_soccer_ids = set()
        self.completed_soccer_ids = set()
        self.sim_lock = threading.Lock()

    def initialize(self, wait_seconds: float = 1.0) -> None:
        time.sleep(wait_seconds)
        for robot in self.robots:
            self.agent_ids.append(self.simulator.addAgent(pos_p_to_rvo_sim(robot.get_position())))
            robot.prepare_for_sorting()
            time.sleep(0.1)

    def play(self, max_steps: int = 10000, rvo_substeps: Optional[int] = None) -> None:
        substeps = self.rvo.substeps if rvo_substeps is None else rvo_substeps
        for i in range(max_steps):
            self.collect_completed_soccer()
            targets_by_robot = self.refresh_targets()
            # print("targets_by_robot", {k: [t.device_id for t in v] for k, v in targets_by_robot.items()})
            # print(self.completed_soccer_ids)
            self.update_robot_tasks(targets_by_robot)

            with self.sim_lock:
                for index, robot in enumerate(self.robots):
                    self.simulator.setAgentPosition(
                        self.agent_ids[index], pos_p_to_rvo_sim(robot.get_position())
                    )
                self.set_preferred_velocities()
                self.step_rvo(substeps)

            # if self.is_finished():
            #     print("All sorting tasks are finished.")
            #     return
            time.sleep(0.01)

    def refresh_targets(self) -> Dict[int, List[SoccerTarget]]:
        targets_by_robot = {robot_id: [] for robot_id in self.robot_ids}
        # print("-------------------------------------- target refresh --------------------------------------")
        for robot_id, logical_soccer_ids in self.soccer_plan.items():
            for logical_id in logical_soccer_ids:
                device_id = soccer_device_id(logical_id, self.sorting.soccer_start_id)
                soccer = self.soccers[logical_id]
                position = soccer.get_position()
                # print(f"soccer {device_id} position: {position}")
                
                if device_id in self.completed_soccer_ids:
                    continue
                if position == (0, 0):
                    print(f"soccer {device_id} position has not been received yet.")
                    continue
                if is_sorted_position(position, self.sorting):
                    continue

                targets_by_robot[robot_id].append(SoccerTarget(logical_id, device_id, position))
        return targets_by_robot

    def update_robot_tasks(self, targets_by_robot: Dict[int, List[SoccerTarget]]) -> None:
        for robot in self.robots:
            # print("robot", robot.robot_id, "mode", robot.mode.value, "target", robot.target_position, "dropoff", robot.dropoff_position)
            if robot.mode == RobotMode.CARRYING:
                robot.send_to_dropoff()
                continue
            if robot.mode == RobotMode.FINISHED:
                continue
            # if robot.mode != RobotMode.IDLE
            #     continue
            if robot.mode == RobotMode.MOVING_TO_DROPOFF or robot.mode == RobotMode.PICKING_UP or robot.mode == RobotMode.DROPPING_OFF:
                continue
            targets = targets_by_robot.get(robot.robot_id, [])
            if targets:
                nearest = min(
                    targets,
                    key=lambda target: distance(
                        robot.get_position(),
                        self.soccers[target.logical_id].get_position()
                    )
                )
                pickup_position = pickup_approach_position(
                    robot.get_position(),
                    nearest.position,
                    self.motion.gripper_offset,
                )
                self.claimed_soccer_ids.add(nearest.device_id)
                robot.assign_soccer(nearest.device_id, nearest.position, pickup_position)
            else:
                robot.mark_finished_if_idle_too_long(self.sorting.max_idle_cycles_before_finish)

    def collect_completed_soccer(self) -> None:
        for robot in self.robots:
            completed = robot.consume_completed_soccer()
            if completed is None:
                continue
            self.completed_soccer_ids.add(completed)
            self.claimed_soccer_ids.discard(completed)

    def set_preferred_velocities(self) -> None:
        v0 = self.rvo.pref_velocity_gain
        for index, robot in enumerate(self.robots):
            if robot.mode not in {RobotMode.MOVING_TO_SOCCER, RobotMode.MOVING_TO_DROPOFF}:
                self.simulator.setAgentPrefVelocity(self.agent_ids[index], (0.0, 0.0))
                continue
            sx, sy = pos_p_to_rvo_sim(robot.get_position())
            tx, ty = float(robot.target_position[0]), float(robot.target_position[1])
            gx, gy = pos_p_to_rvo_sim((tx, ty))
            self.simulator.setAgentPrefVelocity(
                self.agent_ids[index], ((gx - sx) * v0, (gy - sy) * v0)
            )

    def step_rvo(self, step_count: int = 1) -> List[Position]:
        waypoints: List[Position] = []
        for _ in range(step_count):
            self.simulator.doStep()
            waypoints = [
                (
                    round(self.simulator.getAgentPosition(agent_id)[0], 3),
                    round(self.simulator.getAgentPosition(agent_id)[1], 3),
                )
                for agent_id in self.agent_ids
            ]
            for index, robot in enumerate(self.robots):
                sim_wp = waypoints[index]
                final_sim = pos_p_to_rvo_sim(
                    (float(robot.target_position[0]), float(robot.target_position[1]))
                )
                robot.move_towards(sim_wp, final_sim, self.motion)

        return waypoints

    def is_finished(self) -> bool:
        return len(self.completed_soccer_ids) >= self.sorting.soccer_count and all(
            robot.mode in {RobotMode.IDLE, RobotMode.FINISHED}
            for robot in self.robots
        )

    def stop(self) -> None:
        print("Stopping the game...")
        for robot in self.robots:
            robot.stop()
        for soccer in self.soccers.values():
            soccer.stop()


def load_rvo2():
    try:
        import rvo2

        return rvo2
    except ModuleNotFoundError as original_error:
        project_root = Path(__file__).resolve().parent
        build_dir = project_root / "Python-RVO2" / "build"
        candidates = sorted(build_dir.glob("lib*/rvo2*.so"))
        for candidate in candidates:
            import sys

            sys.path.insert(0, str(candidate.parent))
            try:
                import rvo2

                return rvo2
            except ModuleNotFoundError:
                continue
        raise ModuleNotFoundError(
            "Cannot import rvo2. Install Python-RVO2 or build it with "
            "`cd Python-RVO2 && python setup.py build_ext --inplace`."
        ) from original_error
