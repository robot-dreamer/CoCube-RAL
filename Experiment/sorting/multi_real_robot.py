import argparse

from sorting_config import MotionConfig, NetworkConfig, RvoConfig, SortingConfig
from sorting_game import MultiAgentSortingGame
import time


Game = MultiAgentSortingGame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CoCube multi-agent soccer sorting demo.")
    parser.add_argument("--teams", type=int, default=3, help="number of teams")
    parser.add_argument("--robots", type=int, default=6, help="number of robots, with IDs starting from 1")
    parser.add_argument("--soccers", type=int, default=7, help="number of soccer balls, with device IDs starting from 21")
    parser.add_argument("--map-width", type=int, default=480)
    parser.add_argument("--map-height", type=int, default=480)
    parser.add_argument("--soccer-start-id", type=int, default=21)
    parser.add_argument("--gateway", default="192.168.3.1")
    parser.add_argument("--local-ip", default="192.168.3.118")
    parser.add_argument("--ip-prefix", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sorting = SortingConfig(
        team_count=args.teams,
        robot_count=args.robots,
        soccer_count=args.soccers,
        map_size=(args.map_width, args.map_height),
        soccer_start_id=args.soccer_start_id,
    )
    network = NetworkConfig(
        gateway=args.gateway,
        local_ip=args.local_ip,
        ip_prefix=args.ip_prefix,
    )
    game = MultiAgentSortingGame(
        sorting=sorting,
        network=network,
        motion=MotionConfig(),
        rvo=RvoConfig(),
        robot_ids=list(range(1, args.robots + 1)),
    )
    time.sleep(1)
    try:
        game.initialize()
        game.play(max_steps=args.max_steps)
    except KeyboardInterrupt:
        print("Stopping all robots...")
    finally:
        game.stop()
        print("All robots stopped.")


if __name__ == "__main__":
    main()
