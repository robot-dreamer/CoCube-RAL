import unittest

from sorting_config import SortingConfig
from sorting_rules import (
    build_dropoff_positions,
    build_robot_soccer_plan,
    distance,
    is_sorted_position,
    pickup_approach_position,
    soccer_device_id,
    team_id_for_robot_order,
)


class SortingRulesTest(unittest.TestCase):
    def test_soccer_owner_cycles_when_soccer_count_exceeds_robot_count(self):
        self.assertEqual(build_robot_soccer_plan([1, 2], 5), {1: [1, 3, 5], 2: [2, 4]})

    def test_soccer_device_id_starts_at_21(self):
        self.assertEqual(soccer_device_id(1), 21)
        self.assertEqual(soccer_device_id(5), 25)

    def test_team_assignment_uses_equal_contiguous_groups(self):
        config = SortingConfig(team_count=4, robot_count=8, soccer_count=8)
        teams = [team_id_for_robot_order(index, config) for index in range(config.robot_count)]
        self.assertEqual(teams, [1, 1, 2, 2, 3, 3, 4, 4])

    def test_dropoff_positions_are_bounded_and_not_too_close(self):
        config = SortingConfig(team_count=2, robot_count=4, soccer_count=4, random_seed=1)
        robot_ids = [1, 2, 3, 4]
        team_by_robot = {1: 1, 2: 1, 3: 2, 4: 2}
        positions = build_dropoff_positions(robot_ids, team_by_robot, config)

        for x, y in positions.values():
            self.assertGreaterEqual(x, config.dropoff_margin)
            self.assertLessEqual(x, config.map_size[0] - config.dropoff_margin)
            self.assertGreaterEqual(y, config.dropoff_margin)
            self.assertLessEqual(y, config.map_size[1] - config.dropoff_margin)

        for team_id in [1, 2]:
            team_positions = [
                position
                for robot_id, position in positions.items()
                if team_by_robot[robot_id] == team_id
            ]
            self.assertGreaterEqual(distance(team_positions[0], team_positions[1]), config.dropoff_min_distance)

    def test_sorted_position_is_outside_work_area_margin(self):
        config = SortingConfig(
            team_count=1,
            robot_count=1,
            soccer_count=1,
            map_size=(500, 500),
            sorted_margin=100,
        )
        self.assertFalse(is_sorted_position((250, 250), config))
        self.assertTrue(is_sorted_position((50, 250), config))
        self.assertTrue(is_sorted_position((510, 250), config))

    def test_pickup_approach_position_keeps_gripper_offset_from_soccer(self):
        pickup_position = pickup_approach_position((100, 100), (200, 100), 42)
        self.assertEqual(pickup_position, (158, 100))
        self.assertAlmostEqual(distance(pickup_position, (200, 100)), 42)


if __name__ == "__main__":
    unittest.main()
