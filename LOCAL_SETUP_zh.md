# 本机配置与 task 导览

更新日期：2026-11-13。
Isaac Lab 位于相邻的 `../IsaacLab-3.0.0`，版本 v3.0.0，commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`。
Python 3.12.9，PyTorch 2.11.0+cu130，Newton 1.6.1，Warp 1.17.0，RTX 4090 24GB。

## 本项目已移植任务

仅注册 `Hiking-G1-AMP-Unfiltered-v0`（`hikinglab3/__init__.py`），配置为 `HikingEnvCfg`，完整保留 AMP 参考数据。旧过滤任务及配置开关已删除。
环境类 `HikingEnvCfg` 位于 `hikinglab3/config.py`；训练类 `G1ParkourPPORunnerCfg` 位于 `hikinglab3/agent_cfg.py`。
`run.sh` 默认使用该任务；也可显式传入 `--task Hiking-G1-AMP-Unfiltered-v0`。

机器人为 Unitree G1，29 个关节，带鞋、浮动基座、自碰撞。物理 200Hz，策略 50Hz。
策略通过关节位置目标和 PD 控制机器人，根据目标位置生成速度/航向指令，到达阈值 0.4m。
输入包含 8 帧本体历史和 8 帧 18×32 深度图；原始深度射线为 64×36，最大 2.5m。
训练为 PPO + AMP + 4 专家 MoE；深度编码输出 128 维；actor/critic 隐层为 [256,128,64]。
AMP 使用 10 帧、670 维状态窗口。26 项奖励涵盖速度/航向跟踪、存活、足部安全、姿态、能耗和动作平滑等。

## 10 类训练地形

地形源文件：`hikinglab3/terrain_cfg.py`；速度/奖励/随机化：`hikinglab3/task_terms_cfg.py`。
下列比例是配置权重；实际列数由地形生成器按地图离散分配。

| 配置键 | 权重 | 内容和范围 |
|---|---:|---|
| perlin_rough | 5% | 粗糙路面，噪声幅度 0–0.1m，目标行走速度 0.45–1.0m/s |
| perlin_rough_stand | 5% | 同类粗糙路面，零速度指令，训练站立 |
| square_gaps | 10% | 沟壑，宽 0.1–0.7m、深 0.4–0.6m |
| pyramid_stairs | 15% | 上楼，台阶高 0.05–0.23m、宽 0.3m |
| pyramid_stairs_high | 10% | 上高平台，台阶高 0.05–0.45m、宽 1.5m |
| pyramid_stairs_inv | 15% | 下楼，台阶高 0.05–0.23m、宽 0.3m |
| pyramid_stairs_inv_high | 10% | 下高平台，台阶高 0.05–0.45m、宽 1.5m |
| boxes | 10% | 20 个离散大障碍，高 0.05–0.45m、宽 0.8–1.5m |
| mesh_boxes | 10% | 随机小箱体，平均高随难度为 0.1–0.4m，长宽约 0.4m |
| hf_pyramid_slope_inv | 10% | 倒金字塔坡面，坡度参数 0–0.7（高宽比） |

这些是一个混合训练任务的子地形，并非十个独立 Gym ID。
默认地形块 8×8m，完整地图 10×20 块，启用难度课程，初始等级 0–5。
除站立和普通粗糙路面外，目标速度范围一般为 0.45–0.8m/s。

## profile 与 mode

| 参数 | 行为 |
|---|---|
| --profile original | 正式配置；train 默认 4096 环境、30000 更新、10×20 地形、20秒回合 |
| --profile smoke | 小地图 2×20，初始地形等级 0；默认迭代数 10；train 若未指定环境数默认 16 |
| --mode smoke | 零动作步进，检查浮动基座、观测、奖励、AMP 数值；默认 16 环境 |
| --mode train | 训练/续训；--iterations 表示本次新增更新数；--checkpoint 指定恢复模型 |
| --mode play | 必须给 --checkpoint；默认 10 环境，--steps 控制步数 |
| --play-settings original | 默认回放配置：4×10地图、无墙、10秒回合，移除根高度终止和材料随机化、使用确定关节重置 |
| --play-settings training | 保留训练地形与终止规则，用于诊断；其他参数仍可覆盖 |
| --video | 使用无窗口 Newton GL 渲染并录制视频 |

单独地形的正式 checkpoint 评估在 `scripts/evaluate_checkpoints.py`，提供 rough/gap/stair_down/platform_down/stair_up/platform_up/large_box/small_box/ramp 九类测试。
评估使用九类固定地形；训练使用包含站立区域的十个子地形。

## 常用命令

```bash
cd hiking-in-the-wild-isaaclab3
./run.sh --profile smoke --mode smoke --num-envs 16 --steps 200 --log-dir "$PWD/logs/smoke"

# 4096 环境，30000 更新，每 1000 次保存
./run.sh --task Hiking-G1-AMP-Unfiltered-v0 --mode train --num-envs 4096 --iterations 30000 --save-interval 1000 --log-dir "$PWD/logs/g1"

# 按实际生成的 checkpoint 路径替换下面示例
./run.sh --mode play --num-envs 1 --steps 750 --checkpoint /path/to/model.pt --video --log-dir "$PWD/logs/play"

PYTHONPATH="$PWD" ../IsaacLab-3.0.0/.venv/bin/python -m unittest discover -s tests -v
```

每个训练更新收集 24 个策略步；PPO 默认 5 个 epoch、4 个 minibatch，初始学习率 1e-3，adaptive KL。
4096 环境、30000 更新的训练与评估已完成，统计见 [评估结果](results/30000/README.md)。

## 子模块保留的其他任务配置

下表为静态读取 `vendor/InstinctLab` 中的注册项，每项都有 Train 和 Play 两种 ID。
这些任务使用上游旧版 InstinctLab 入口运行；当前 `run.sh` 接入上表中的两个 G1 徒步任务。

| Train Gym ID | 用途 |
|---|---|
| Instinct-Parkour-Target-Amp-G1-v0 | 上游 G1 目标驱动 AMP 徒步，当前移植的来源 |
| Instinct-Parkour-Target-Amp-X2-v0 | X2 徒步，点采样足部安全 |
| Instinct-Parkour-Target-Amp-X2-Cylinder-v0 | X2 徒步，圆柱体足部安全 |
| Instinct-Parkour-Target-Amp-X2-Waist-Camera-v0 | X2 徒步，腰部深度相机 |
| Instinct-Parkour-Target-Amp-X2-Cylinders-Waist-Camera-v0 | X2 徒步，圆柱体足部安全与腰部相机 |
| Instinct-Locomotion-Flat-G1-v0 | G1 平地运动控制 |
| Instinct-BeyondMimic-Plane-G1-v0 | G1 BeyondMimic 动作跟踪 |
| Instinct-Shadowing-WholeBody-Plane-G1-v0 | G1 全身动作跟踪 |
| Instinct-Perceptive-HOI-Shadowing-G1-v0 | G1 感知驱动的人物交互动作跟踪 |
| Instinct-Perceptive-Shadowing-G1-v0 | G1 感知动作跟踪 |
| Instinct-Perceptive-Vae-G1-v0 | G1 感知 VAE 任务 |

Play ID 在末尾 `-v0` 前加 `-Play`，共 11 组、22 个上游注册项。

## 验证与结果

两套 4096 环境、30000 更新模型均已完成训练与评估。无过滤 Lab3 / Newton 总体成功率 98.53%，官方策略 / Newton 为 97.62%，官方策略 / PhysX 为 98.84%。每列九类地形，每类 1000 次。

回归测试覆盖参考流、坐标映射、镜像、速度与重力单位、深度射线和足部边缘。命令与训练设置见 [README.md](README.md)，完整实验条件见 [results/30000](results/30000/README.md)。
