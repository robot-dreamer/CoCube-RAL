# CoCube 多智能体足球分拣 Demo

这个项目用于运行多机器人足球分拣实验：`n` 个机器人按 IP/ID 从 1 开始编号，`m` 支队伍平均分配机器人，每个足球设备从 ID 21 开始编号。逻辑足球 1 由机器人 1 负责，逻辑足球 2 由机器人 2 负责；当足球数量 `p > n` 时继续循环分配，例如机器人 1 还负责逻辑足球 `n + 1`。

## 项目结构

- `multi_real_robot.py`：实验入口，负责读取命令行参数并启动任务。
- `sorting_config.py`：任务、网络、RVO、运动控制参数。
- `sorting_rules.py`：不依赖硬件的纯规则，包括分队、足球分配、已分拣判断、投放点规划。
- `robot_controller.py`：单个 CoCube 的状态机和抓取/投放动作封装。
- `sorting_game.py`：多机器人任务编排，负责目标刷新、任务分配、RVO 避障和停止清理。
- `cocube_udp/`：原始 UDP 控制接口，包含 `CoCube` 和 `Soccer`。
- `tests/`：纯逻辑单元测试，不连接真实机器人。

## 运行方式

默认配置为 8 个机器人、4 支队伍、8 个足球、地图大小 500 x 500：

```bash
python multi_real_robot.py
```

常用参数：

```bash
python multi_real_robot.py \
  --teams 2 \
  --robots 4 \
  --soccers 6 \
  --map-width 500 \
  --map-height 500 \
  --gateway 192.168.3.1 \
  --local-ip 192.168.3.118 \
  --ip-prefix 100
```

约束：

- `--robots` 必须能被 `--teams` 整除，因为每队固定 `n / m` 个机器人。
- 机器人 ID 默认是 `1..n`。
- 足球设备 ID 默认从 21 开始，即逻辑足球 1 对应设备 21。

## 任务逻辑

1. 按机器人顺序平均分队，例如 8 个机器人、4 支队伍时为 `[1,1,2,2,3,3,4,4]`。
2. 足球按逻辑编号循环分配给机器人：`owner = (logical_soccer_id - 1) % robot_count + 1`。
3. 只有未分拣足球会被加入目标列表。足球位于地图工作区外侧边缘，或超出地图边界时，会被视为已分拣。
4. 空闲机器人只会选择自己负责且未被其他机器人声明的足球，优先选择距离最近的目标。
5. 机器人不会直接把足球中心作为夹取目标，而是先移动到距离足球约 42 的预夹取点。
6. 投放点按队伍数量分布在地图边缘，并对同队机器人加入随机扰动和最小距离约束，避免目标过近。
7. 到达预夹取点后，机器人朝向足球等待约 1 秒，再关闭夹爪并进入回收状态。
8. 机器人移动过程通过 RVO2 生成局部 waypoint，再调用 CoCube 的 `move_to_target` 执行。

## 测试

纯规则测试不依赖摄像头、UDP 或 RVO 真机环境：

```bash
python -m unittest discover -s tests
```

语法检查：

```bash
python -m py_compile multi_real_robot.py sorting_config.py sorting_rules.py robot_controller.py sorting_game.py
```

## 关键参数

`SortingConfig`：

- `team_count`：队伍数量 `m`。
- `robot_count`：机器人数量 `n`。
- `soccer_count`：足球数量 `p`。
- `map_size`：地图宽高。
- `soccer_start_id`：足球设备起始 ID，默认 21。
- `sorted_margin`：地图边缘已分拣区域宽度。
- `dropoff_min_distance`：同队投放点之间的最小距离。

`MotionConfig`：

- `gripper_offset`：夹爪到机器人期望停靠点的距离，默认 42。
- `pickup_arrival_tolerance`：到达预夹取点的容差。
- `pre_grasp_wait_seconds`：到达预夹取点后、关闭夹爪前的等待时间，默认 1 秒。
- `dropoff_distance`：靠近投放点多少距离内执行投放。
- `command_speed`：发送给 CoCube 的移动速度。
- `rvo_speed_gain`：RVO 偏好速度增益。

`NetworkConfig`：

- `gateway`：机器人网关。
- `local_ip`：本机 IP。
- `ip_prefix`：设备 IP 的末段偏移，保持和原项目一致。
