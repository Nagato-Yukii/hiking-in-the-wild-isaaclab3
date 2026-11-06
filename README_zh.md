# Hiking in the Wild → Isaac Lab 3.0

> 最新发布状态与通用安装说明见 [README.md](README.md)。以下包含本机历史实验记录；提及的 logs、checkpoint 和原始数据不随仓库发布。最新模型为续训后的 17000，出现斜坡退化。

本目录是 **Unitree G1 29 DoF、带鞋模型**的独立移植工程，使用本机 `/home/ye/Desktop/IsaacLab-3.0.0` 的 Python 3.12 / Newton MJWarp 环境。原有 X2 工程和 Isaac Lab 源码未改动。Gym 任务名：`Hiking-G1-AMP-v0`。

**更新：已修复旧视频对应训练中的速度阈值与重力单位错误，并恢复官方实验默认值。** 见 [实验设计对齐与学习空间审计](audit/实验设计对齐.md)。旧 `final-*` checkpoint 保留供审计，不再作为续训起点。

这是可执行训练链路的移植，**不是论文性能复现或已经收敛的野外行走策略**。短训练 checkpoint 用于验证保存、续训和回放，不能当作实机策略。移植范围是 G1 parkour；其他 InstinctLab 任务和 X2 训练不在本次移植范围内。

2026-11-04 的[训练预分析与容量实测](audit/训练预分析_20261104.md)进一步核对了随机化和台阶边界：Newton 材料随机化语义不同，边缘检测前尚缺原版全局网格清理，深度截断顺序也有差异；因此配置参数对齐不能视为全部行为等价。该报告区分已验证的训练容量、短测试吞吐与尚未验证的最终效果。

随后完成[2048环境×100更新速度对照](audit/速度对比_2048环境_100更新.md)：两套完整训练链路均通过；Newton/MJWarp稳态约16,041 transitions/s，原Instinct/PhysX约11,270 transitions/s，吞吐比1.423。此为完整实现对照，不是孤立求解器微基准。

长训练已按用户要求停止，最新保存模型为 `logs/g1-newton-2048-20k-20261104_062959/model_15000.pt`。已完成与官方公开楼梯ONNX的[配对效果评估](audit/Checkpoint对比_15000_vs_官方楼梯.md)，每类1000次、两模型共18000次：本地模型的20cm上楼成功率0%，官方楼梯模型100%；本地尚不具备可靠上行越障能力。

## 已接入的组件

| 组件 | 实现 |
|---|---|
| 机器人 | 原项目 torso-base G1 29 DoF 带鞋 URDF；转换 USD；29 关节、浮动基座启动断言 |
| 控制 | 官方带鞋 G1 最终配置的隐式 PD、惯量、动作缩放；200 Hz 物理步、50 Hz 策略 |
| 地形 | 原始 10 类配置：Perlin、静止区域、间隙、高低台阶、上下楼、离散障碍、网格障碍、坡面；难度课程 |
| 目标 | 原始 Flat Patch Sampling、目标位置→速度指令、到达目标判定、站立概率 |
| 视觉 | 64×36 深度射线、机器人自身遮挡、0.1 m 最小命中距离；裁剪至 18×32、高斯模糊、0–2.5 m 归一化 |
| 时序 | 8 帧本体历史；37 帧深度缓存，以 5 帧间隔选 8 帧并使用原配置延迟范围 |
| 网络 | 原 `EncoderMoEActorCritic`，4 个专家；深度编码器输出 128 维；原 actor/critic 规模 |
| AMP | 原 `WasabiPPO`、MSE 判别器目标、梯度惩罚、二次型风格奖励；10 帧、670 维状态窗口 |
| 足部安全 | 原 Greedyconcat 边缘检测与空间索引；每只脚 100 个体积点、半径 0.05 m 边缘圆柱、速度加权穿透惩罚 |
| 训练 | 原 26 项奖励、终止条件、摩擦/恢复系数随机化、PPO 优化器、TensorBoard、checkpoint 续训 |

## AMP 数据

已经检查本地已有数据，并从[官方项目页](https://project-instinct.github.io/hiking-in-the-wild/)的 Data & Model 链接重新下载核验。

- 本地原件：`/home/ye/hiking-in-the-wild_Data&Model/data&model/parkour_motion_reference/`，保持原样。
- 本工程使用：`data/official/data&model/parkour_motion_reference/parkour_motion_without_run_retargetted.npz`。
- 18,982 帧，50 Hz，29 个关节，约 379.64 秒；这是官方公开的参考动作包，不是额外下载的完整 AMASS 数据库。
- NPZ SHA-256：`7cfb7c1dcaa6f2a55a13c4849be9e17b4c960ce4015c500ac0ddfb9d77f4ba5b`。
- `scripts/fetch_data.py` 优先复用匹配校验和的本地文件，缺失时下载；`data/provenance.json` 记录来源与校验和。
- 按**关节名称**映射到 Newton 的顺序；数据 WXYZ 显式转为 XYZW，速度使用 50 Hz 前向差分，线/角速度转换到机身坐标系。
- 训练使用连续参考流和 10 帧历史，重置时填充首帧，并以 50% 概率左右镜像；检测到明显拼接跳变后重采样参考流。公开 NPZ 缺少原始分段元数据，因此不声称抽样分布完全等价。该实现替代了旧版基于 PhysX 的 MotionReferenceManager；不需要把参考机器人实例加载进场景。

数据包中的 `checkpoints/*/exported/*.onnx` 是官方推理模型，已保留。它们不是 PPO 优化器 checkpoint，本移植未验证其直接导入；**不得将这些 ONNX 与本机新训练的 `.pt` 混称**。Newton 与旧 PhysX 的关节顺序不同，官方推理模型还需要单独处理输入/输出映射。

## 使用

```bash
cd /home/ye/Desktop/HikingLab3

# 首次安装或重新生成资产；本机已经完成
./scripts/bootstrap.sh

# 短步进检查；缩小地图必须显式选择 smoke profile
./run.sh --profile smoke --mode smoke --num-envs 16 --steps 200 \
  --log-dir "$PWD/logs/my-smoke"

# 官方一般行走实验默认值：4096 环境、30000 更新、10×20 地形、初始等级 0–5
# 本机16GB实测：4096初始化成功，但首次完整更新前触发显存余量保护，尚未训练通过。
# 已短测通过1024/2048环境；容量与实验批次差异见 audit/训练预分析_20261104.md。
./run.sh --profile original --mode train --log-dir "$PWD/logs/g1-original"

# 本机修复验证：保持完整地形/课程，显式减少并行数和更新数
./run.sh --mode train --num-envs 256 --iterations 1000 \
  --log-dir "$PWD/logs/g1-local-validation"

# 修复后的 checkpoint 续训；iterations 表示新增更新数
./run.sh --mode train --num-envs 256 --iterations 1000 \
  --checkpoint "$PWD/logs/learning-space-final/model_1000.pt" \
  --log-dir "$PWD/logs/g1-continued"

# 官方 PLAY 设置：移除墙、4×10 地形、10 秒回合
# 与训练地形/终止规则不同，不能将此视频当作正式成功率评估
 env -u DISPLAY ./run.sh --mode play --num-envs 1 --steps 750 \
  --checkpoint "$PWD/logs/learning-space-final/model_1000.pt" --video \
  --log-dir "$PWD/logs/my-video"
```

`--iterations` 在续训时表示**新增迭代数**。更改网络或观测尺寸后不能直接续训旧 checkpoint。训练配置在 `hikinglab3/agent_cfg.py`；环境配置在 `hikinglab3/config.py`；地形配置在 `hikinglab3/terrain_cfg.py`。替换动作数据时可修改 `motions.py` 的路径，加载器会检查字段、帧率、关节集合和四元数。

`run.sh` 使用锁定的 Isaac Lab 环境和显式依赖；不会安装新的 GPU 驱动，也不会启动后台训练服务。

## 验证与差异

验证记录位于 `logs/`，最终汇总见 `validation.json`。独立数值测试覆盖了关节映射/坐标系/速度单位、足部边缘圆柱轴线处的数值奇点，以及近距离机器人几何不能遮住远处地形的深度过滤。

```bash
PYTHONPATH="$PWD" ../IsaacLab-3.0.0/.venv/bin/python -m unittest discover -s tests -v
```

主要迁移变化：3.0 ProxyArray 显式 `.torch` 访问、XYZW 四元数、执行器实时增益接口、重置 slice 索引、3.0 地形生成返回协议、浮动基座 USD 转换、Newton 深度相机与近距离过滤。保留旧地形装饰器的原点算法，以保持目标与地形中心对齐。

与上游的实质区别是 **Newton/MJWarp 物理后端**、原生多网格射线相机，以及直接 AMP 窗口采样；它们需要重新训练和后续性能评估，不能视为 PhysX 的数值等价实现。此前短验证使用 2×20 地形；本次修复验证恢复 10×20 地形，仍不代表全部难度已学会。未进行实机部署、完整 30,000 迭代、多随机种子复现或 sim-to-real 验证。

调试阶段曾发现转换器把模型固定到世界。相关早期记录已标记为无效，见 `logs/INVALID_EARLY_CHECKPOINTS.txt`；这些结果还存在本次发现的速度阈值问题；当前仅推荐 `learning-space-final` 及其后续训练。现在启动会拒绝固定基座模型。

来源、版本与许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 本次学习空间修复验证

- `logs/learning-space-final/model_1000.pt`：256 环境、1000 更新、6,144,000 transitions；完整 10×20 地形，原版奖励权重和课程。训练循环约 918 秒。
- 同一轮训练曲线的前 100 次更新平均回合 **1.03 s**，后 100 次为 **8.92 s**；后 100 次平均 terrain level **0.315**。这是训练改善的证据，尚不是论文成功率评估或全难度收敛。
- `audit/learning_final.json` 保存全部 1000 个更新的曲线；`logs/design-tests.log` 记录 5 项通过的数值测试。
- 保持训练规则的无探索噪声回放：`--mode play --play-settings training`。默认 PLAY 仍按官方单独的回放配置，二者不混算。


2026-11-05：修复AMP参考流重复读取推进和网格清理遗漏，详见[audit/AMP与网格缺陷核查_20261105.md](audit/AMP与网格缺陷核查_20261105.md)。现有15000 checkpoint为修复前训练产物，未重训。
