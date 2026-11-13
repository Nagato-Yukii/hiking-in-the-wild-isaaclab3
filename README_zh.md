# Hiking in the Wild — Isaac Lab 3.0

Unitree G1 29 自由度带鞋模型，使用 Isaac Lab 3.0 / Newton MJWarp，包含深度感知、PPO、AMP、四专家 MoE、地形课程和 checkpoint 续训。[安装与命令](README.md) · [任务与地形配置](LOCAL_SETUP_zh.md)。

## 本次 fix

仅保留已完成 30000 更新训练与评估的 `Hiking-G1-AMP-Unfiltered-v0`，默认训练、播放和评估均使用此任务。删除旧过滤任务的注册、配置开关及采样分支，保留机器人、地形、观测、奖励和训练配置。具体修复：

1. **移除速度阈值过滤。** 原实现把根线速度 >10 m/s 或关节速度 >100 rad/s 的参考帧标记为边界，剔除包含这些帧的十帧窗口。当前数据有 49 处标记，影响 490 个窗口。新任务保留这些帧及对应窗口，不裁剪速度。
2. **移除内部跳变前的提前重采样。** 原连续参考流将有效窗口起点掩码当作游标有效性条件，会在标记帧前最多九帧重新采样。新任务顺序经过所有内部相邻帧，只在回合重置或文件结尾重新选择参考游标。
3. **覆盖完整文件。** 取消前 90% 起点限制，全部 18,982 帧均可作为参考起点；随机窗口接口包含全部 18,973 个完整十帧窗口，补齐原实现遗漏的最后一个窗口。
4. **统一任务入口和保存周期。** 省略 `--task` 即使用无过滤任务，`--save-interval 1000` 每千次更新保存。
5. **评估按需加载模型。** `--policies local` 只加载本地策略；`--official-checkpoint` 可加载官方训练的 `.pt`，按关节名称转换观测历史和动作，逐批核对初始机器人状态及目标哈希。

参考动作的关节映射、WXYZ→XYZW 转换、50 Hz 前向速度差分、机身坐标变换、特征缩放和 50% 镜像增强沿用现有实现。恢复训练同时恢复策略、判别器、两个优化器、更新编号和 PPO 自适应学习率标量。

## 与官方实现的区别

| 项目 | 本仓库 Isaac Lab 3.0 实现 | 官方 InstinctLab 训练实现 |
|---|---|---|
| 运行环境 | Isaac Lab 3.0、Newton MJWarp、Python 3.12、PyTorch 2.11 | Isaac Lab / Isaac Sim 5.1、PhysX、Python 3.11、PyTorch 2.7 |
| 物理求解 | Newton `implicitfast`，每个物理步含两个 MJWarp 子步 | PhysX 刚体关节与接触求解器 |
| AMP 参考 | 直接读取 NPZ，显式转换四元数和关节顺序，完整参考流 | `MotionReferenceManager` / `AmassMotion` 管理参考动作 |
| 深度处理 | Lab3 射线相机与本地历史缓冲；先截断至 2.5 m，再裁剪、缩放、模糊 | InstinctLab 射线相机与传感器处理；先裁剪、缩放、模糊，再截断深度 |
| 材质随机化 | Newton 摩擦参数接口 | PhysX 静/动摩擦与 64 桶材质采样 |
| 训练算法 | 上游 `instinct_rl` PPO + AMP、四专家 MoE、24 步 rollout、5 epoch、4 minibatch | 相同算法与网络配置 |

两者均使用 G1 29 自由度模型、地形课程、26 项奖励权重、200 Hz 物理步、50 Hz 控制和 50% 参考镜像增强。物理求解、接触材质处理和深度预处理存在上述差异。

### 实测训练耗时

同一张 RTX 4090，4096 环境、seed 42；两次初始 10000 更新均连续完成。Lab3 取训练入口计时，官方取训练循环累计时间（含进程启动为 14.06 小时）：

| 阶段 | Lab3 / Newton | 官方 / PhysX |
|---|---:|---:|
| 0 → 10000 更新 | 11.85 小时 | 14.03 小时 |
| 平均每次更新 | 4.27 秒 | 5.05 秒 |

该 1 万更新阶段 Lab3 耗时减少约 **15.5%**。Lab3 后续 20000 更新耗时 23.85 小时，两个训练阶段合计 35.70 小时。官方续训在记录到 23291 更新时中断，从已保存的 23000 checkpoint 恢复，最后 7000 更新训练耗时 9.80 小时。[计时原始值与统计范围](results/30000/training_timing.json)。

## 4096 环境、30000 更新实验

两套模型均为 seed 42，先训练 10000 更新，再追加 20000。Lab3 使用无过滤任务，官方使用 InstinctLab / PhysX。两个最终模型在同一个 Newton 场景各测试九类地形、每类 1000 次；官方模型另在原生 PhysX 测试 9000 次。

| 地形 | 无过滤 Lab3 / Newton | 官方 / Newton | 官方 / PhysX |
|---|---:|---:|---:|
| 粗糙地面 | 100.0% | 100.0% | 100.0% |
| 沟壑 | 100.0% | 100.0% | 99.9% |
| 下楼梯 | 99.9% | 99.6% | 100.0% |
| 下高台 | 100.0% | 100.0% | 100.0% |
| 上楼梯 | 100.0% | 100.0% | 100.0% |
| 上高台 | 99.7% | 100.0% | 100.0% |
| 大箱体 | 99.6% | 92.2% | 94.1% |
| 小箱体 | 87.6% | 86.8% | 95.7% |
| 坡面 | 100.0% | 100.0% | 99.9% |
| **总体** | **98.53%** | **97.62%** | **98.84%** |

官方 / Newton 和官方 / PhysX 使用**同一个 PhysX 训练的 checkpoint**，分别在两个引擎执行。同一 Newton 场景中，Lab3 比官方高 0.91 个百分点，主要差异来自大箱体。小箱体仍是两者在 Newton 下成功率最低的地形。10000 更新时，无过滤 Lab3 总体为 72.20%，官方 / Newton 为 64.63%；续训后上高台、大箱体和小箱体均有提升。官方原生 PhysX 总体为 98.84%，小箱体为 95.7%；Lab3 / Newton 小箱体为 87.6%。表中成功率衡量到达目标的能力，步态表现另见视频和 AMP 审计。

[完整评估条件、计数和模型校验和](results/30000/README.md) · [训练曲线](results/30000/training_comparison.png) · [原始统计](results/30000/success_rates.json)。

[45 秒左右对比视频](artifacts/lab3_vs_official_30000.mp4)：左侧无过滤 Lab3，右侧官方策略，均在 Newton 中执行。依次为粗糙地面、沟壑、上楼，取固定首个试验；回合结束前冻结画面。

## 训练与续训

```bash
./run.sh --task Hiking-G1-AMP-Unfiltered-v0 --mode train \
  --num-envs 4096 --iterations 10000 --save-interval 1000 --log-dir "$PWD/logs/g1-10k"

# iterations 表示新增更新数，最终保存 model_30000.pt。
./run.sh --task Hiking-G1-AMP-Unfiltered-v0 --mode train \
  --num-envs 4096 --iterations 20000 --save-interval 1000 \
  --checkpoint "$PWD/logs/g1-10k/model_10000.pt" --log-dir "$PWD/logs/g1-30k"
```

配置入口：`hikinglab3/config.py`、`agent_cfg.py`、`terrain_cfg.py`、`motions.py`。原始动作文件 SHA-256：`7cfb7c1dcaa6f2a55a13c4849be9e17b4c960ce4015c500ac0ddfb9d77f4ba5b`。

## 验证

新增回归测试覆盖高速根位置与关节变化穿过参考历史、全部窗口和尾部帧可访问；现有测试覆盖坐标/关节映射、镜像、重复读取幂等性、重力单位、速度惩罚、足部边缘和深度射线。运行：

```bash
PYTHONPATH="$PWD:$PWD/vendor/instinct_rl" "$ISAACLAB_PATH/.venv/bin/python" -m unittest discover -s tests -v
```

来源、固定版本和许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。
