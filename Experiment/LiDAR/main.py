import argparse
from collections import deque
from dataclasses import dataclass
import math
import os
import time

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

import utils


LOCAL_IP = "192.168.3.10"
ROBOT_IPS = {
    1: "192.168.3.101",
    2: "192.168.3.102",
    3: "192.168.3.103",
}
GATEWAY = "192.168.3.1"
IP_PREFIX = 100
UDP_PORT = 5000


@dataclass
class RobotFrame:
    robot_id: int
    pose: object
    world_lidar_points: np.ndarray


class MultiRobotMapVisualizer:
    def __init__(self, map_width, map_height, scale, title="CoCube Multi-Robot Mapping"):
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.scale = float(scale)
        self.window_width = max(1, int(round(self.map_width * self.scale)))
        self.window_height = max(1, int(round(self.map_height * self.scale)))

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont(None, 18)
        self.clock = pygame.time.Clock()

        self.robot_colors = {
            1: (255, 80, 80),
            2: (80, 170, 255),
            3: (255, 210, 70),
        }

    def poll_events(self):
        reset_map = False
        quit_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_requested = True
                elif event.key == pygame.K_q:
                    reset_map = True
        return reset_map, quit_requested

    def render(self, grid_map, frames_by_robot, recent_clouds, message=""):
        self.screen.fill((255, 255, 255))
        self._draw_grid()
        self._draw_occupancy(grid_map)
        self._draw_clouds(recent_clouds)
        for frame in frames_by_robot.values():
            self._draw_robot(frame.robot_id, frame.pose)
        self._draw_status(grid_map, frames_by_robot, message)
        pygame.display.flip()

    def tick(self, fps):
        self.clock.tick(fps)

    def close(self):
        pygame.quit()

    def _world_to_screen(self, x, y):
        return int(round(x * self.scale)), int(round((self.map_height - y) * self.scale))

    def _draw_grid(self):
        grid_color = (220, 220, 220)
        step = 50
        for gx in range(0, int(self.map_width) + 1, step):
            x, _ = self._world_to_screen(gx, 0)
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.window_height), 1)
        for gy in range(0, int(self.map_height) + 1, step):
            _, y = self._world_to_screen(0, gy)
            pygame.draw.line(self.screen, grid_color, (0, y), (self.window_width, y), 1)

    def _draw_occupancy(self, grid_map):
        probabilities = grid_map.probability_grid()
        inflated = grid_map.inflated_obstacle_mask()
        cell_w = max(1, int(math.ceil(grid_map.resolution * self.scale)))
        cell_h = max(1, int(math.ceil(grid_map.resolution * self.scale)))

        for y in range(grid_map.rows):
            for x in range(grid_map.cols):
                probability = probabilities[y, x]
                if probability < 0.55 and not inflated[y, x]:
                    continue
                world_x = x * grid_map.resolution
                world_y = (y + 1) * grid_map.resolution
                px, py = self._world_to_screen(world_x, world_y)
                if inflated[y, x]:
                    color = (210, 95, 95)
                else:
                    shade = int(245 - 130 * probability)
                    color = (shade, shade, shade)
                pygame.draw.rect(self.screen, color, (px, py, cell_w, cell_h))

    def _draw_clouds(self, recent_clouds):
        for robot_id, clouds in recent_clouds.items():
            color = self.robot_colors.get(robot_id, (180, 180, 180))
            frames = list(clouds)
            if not frames:
                continue
            for idx, cloud in enumerate(frames):
                alpha = (idx + 1) / len(frames)
                cloud_color = tuple(max(0, min(255, int(c * (0.35 + 0.65 * alpha)))) for c in color)
                for point in cloud:
                    px, py = self._world_to_screen(point[0], point[1])
                    if 0 <= px < self.window_width and 0 <= py < self.window_height:
                        pygame.draw.circle(self.screen, cloud_color, (px, py), 2)

    def _draw_robot(self, robot_id, pose):
        if pose is None:
            return
        color = self.robot_colors.get(robot_id, (220, 220, 220))
        rx, ry = self._world_to_screen(pose.position[0], pose.position[1])
        radius = max(4, int(round(6 * self.scale / max(self.scale, 1.0))))
        pygame.draw.circle(self.screen, color, (rx, ry), radius)
        dx = int(round(15 * math.cos(pose.yaw) * self.scale))
        dy = int(round(-15 * math.sin(pose.yaw) * self.scale))
        pygame.draw.line(self.screen, color, (rx, ry), (rx + dx, ry + dy), 2)

        label = self.font.render(str(robot_id), True, (255, 255, 255))
        self.screen.blit(label, (rx + 8, ry - 8))

    def _draw_status(self, grid_map, frames_by_robot, message):
        occupied = int(grid_map.obstacle_mask().sum())
        lines = [
            "q: reset map | esc/window: quit",
            f"robots={len(frames_by_robot)} occ={occupied} scale={self.scale:.2f}",
            f"local={LOCAL_IP} robot_ips={', '.join(ROBOT_IPS.values())}",
        ]
        if message:
            lines.append(message)
        for i, text in enumerate(lines):
            surf = self.font.render(text, True, (35, 35, 35))
            self.screen.blit(surf, (8, 8 + i * 18))


class MultiRobotMappingApp:
    def __init__(self, args, lidar_sensor_cls, occupancy_grid_map_cls):
        self.args = args
        self.robot_ids = list(ROBOT_IPS.keys())
        self._validate_robot_network_config()
        self.sensors = {
            robot_id: lidar_sensor_cls(
                robot_id=robot_id,
                gateway=GATEWAY,
                local_ip=LOCAL_IP,
                ip_prefix=IP_PREFIX,
                udp_port=UDP_PORT,
                lidar_offset=args.lidar_offset,
                range_calibration_file=args.lidar_range_calibration,
                keep_cloud_frames=args.keep_cloud_frames,
                simulated=args.simulated,
            )
            for robot_id in self.robot_ids
        }
        self.grid_map = occupancy_grid_map_cls(
            width=args.map_width,
            height=args.map_height,
            resolution=args.map_resolution,
            raycast_max_range=args.raycast_max_range,
            inflation_radius=args.inflation_radius,
        )
        self.recent_clouds = {
            robot_id: deque(maxlen=args.keep_cloud_frames) for robot_id in self.robot_ids
        }
        self.frames_by_robot = {}
        self.visualizer = MultiRobotMapVisualizer(
            args.map_width,
            args.map_height,
            self._display_scale(args),
        )
        self.message = "Joint mapping started."

    def _validate_robot_network_config(self):
        expected_ips = {
            robot_id: ".".join(GATEWAY.split(".")[:-1]) + f".{IP_PREFIX + robot_id}"
            for robot_id in self.robot_ids
        }
        for robot_id, robot_ip in ROBOT_IPS.items():
            expected_ip = expected_ips[robot_id]
            if robot_ip != expected_ip:
                raise ValueError(
                    f"Robot {robot_id} IP is {robot_ip}, but CoCube will use {expected_ip}. "
                    "Update GATEWAY/IP_PREFIX or ROBOT_IPS so they match."
                )

    def run(self):
        running = True
        try:
            while running:
                reset_map, quit_requested = self.visualizer.poll_events()
                if quit_requested:
                    running = False
                if reset_map:
                    self.grid_map.reset()
                    for clouds in self.recent_clouds.values():
                        clouds.clear()
                    self.message = "Map reset."

                frames = self._read_all_frames()
                self._update_joint_map(frames)

                self.visualizer.render(
                    self.grid_map,
                    self.frames_by_robot,
                    self.recent_clouds,
                    message=self.message,
                )
                self.visualizer.tick(self.args.fps)
        finally:
            for sensor in self.sensors.values():
                sensor.wheels_stop()
                sensor.stop()
            self.visualizer.close()

    def _read_all_frames(self):
        frames = {}
        for robot_id, sensor in self.sensors.items():
            frame = sensor.read()
            frames[robot_id] = RobotFrame(
                robot_id=robot_id,
                pose=frame.pose,
                world_lidar_points=frame.world_lidar_points,
            )
        self.frames_by_robot = frames
        return frames

    def _update_joint_map(self, frames):
        poses = {robot_id: frame.pose for robot_id, frame in frames.items()}
        updated = 0
        for robot_id, frame in frames.items():
            points = self._filter_robot_body_points(
                robot_id,
                frame.world_lidar_points,
                poses,
            )
            if len(points) == 0:
                continue
            self.grid_map.update_from_scan(frame.pose.position, points)
            self.recent_clouds[robot_id].append(points)
            updated += len(points)
        self.message = f"Updated with {updated} lidar points."

    def _filter_robot_body_points(self, source_robot_id, points, poses):
        if points is None or len(points) == 0 or self.args.robot_filter_radius <= 0:
            return points

        filtered = []
        radius = float(self.args.robot_filter_radius)
        for point in np.array(points, dtype=float):
            keep = True
            for robot_id, pose in poses.items():
                if robot_id == source_robot_id:
                    continue
                if np.linalg.norm(point - pose.position) <= radius:
                    keep = False
                    break
            if keep:
                filtered.append(point)
        return np.array(filtered, dtype=float)

    @staticmethod
    def _display_scale(args):
        if args.display_scale > 0:
            return args.display_scale
        return min(
            args.max_window_width / max(args.map_width, 1.0),
            args.max_window_height / max(args.map_height, 1.0),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="CoCube multi-robot joint mapping")
    parser.add_argument("--simulated", action="store_true")
    parser.add_argument("--lidar-range-calibration", default="lidar_range_calibration.json")
    parser.add_argument("--lidar-offset", type=float, default=0.0)

    parser.add_argument("--map-width", type=float, default=600.0)
    parser.add_argument("--map-height", type=float, default=600.0)
    parser.add_argument("--map-resolution", type=float, default=2.0)
    parser.add_argument("--raycast-max-range", type=float, default=600.0)
    parser.add_argument("--inflation-radius", type=float, default=2.0)
    parser.add_argument("--robot-filter-radius", type=float, default=20.0)
    parser.add_argument("--keep-cloud-frames", type=int, default=8)

    parser.add_argument("--display-scale", type=float, default=0.0)
    parser.add_argument("--max-window-width", type=float, default=1100.0)
    parser.add_argument("--max-window-height", type=float, default=900.0)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def configure_map_size(map_width, map_height):
    utils.MAP_W = map_width
    utils.MAP_H = map_height


def main():
    args = parse_args()
    configure_map_size(args.map_width, args.map_height)

    from grid_map import OccupancyGridMap
    from sensor import LidarSensor

    MultiRobotMappingApp(args, LidarSensor, OccupancyGridMap).run()


if __name__ == "__main__":
    main()
