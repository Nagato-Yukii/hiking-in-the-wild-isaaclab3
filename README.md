# Hiking in the Wild — Isaac Lab 3.0

An independent Unitree G1 (29 DoF, shoes) port of [Hiking in the Wild](https://project-instinct.github.io/hiking-in-the-wild/) to **Isaac Lab 3.0 / Newton MJWarp**, with depth perception, AMP and the upstream MoE policy. Task: `Hiking-G1-AMP-v0`.

这是研究中的移植工程，**尚未复现论文性能**。包括训练、续训、配对评估、数值检查和缺陷审计。中文实验细节见 [README_zh.md](README_zh.md)。

## Current results

2048 environments, 15000 updates followed by 2000 additional updates. Both checkpoints were evaluated in the same corrected environment, with matching initial-state signatures and 1000 trials per terrain.

| Terrain | 15000 | 17000 |
|---|---:|---:|
| Rough | 100% | 100% |
| Gap | 0% | 0% |
| Stair down | 94.1% | 98.6% |
| Platform down | 100% | 100% |
| Stair up | 0% | 0% |
| Platform up | 0% | 0% |
| Large box | 18.1% | 17.4% |
| Small box | 7.0% | 13.5% |
| Ramp | 93.2% | 0% |

The additional training **did not improve stair ascent and regressed on ramps**. AMP reference-step caching and terrain mesh cleanup were fixed before resuming; optimizer state and adaptive learning rate were restored, but simulator/curriculum state was reset. This is not an unchanged-environment continuation or a causal ablation. Reward weights were unchanged.

[Detailed evaluation](audit/续训2000更新对比_20261105.md) · [AMP/mesh audit](audit/AMP与网格缺陷核查_20261105.md) · [Reward call audit](audit/奖励计算次数核查_20261105.md)

[Side-by-side ramp video](artifacts/G1_斜坡_15000_vs_17000_续训前后.mp4): left 15000 reaches the goal in 8.78 s (then freezes), right 17000 times out at 15 s. This is one paired example, not the full evaluation.

## Setup

Tested on Linux with an RTX 5070 Ti 16 GB, Python 3.12, Isaac Lab `v3.0.0` (`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`), Newton/MJWarp. Install the prerequisites for [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and `uv` first.

```bash
git clone --recurse-submodules https://github.com/Nagato-Yukii/hiking-in-the-wild-isaaclab3.git
cd hiking-in-the-wild-isaaclab3
# Install Isaac Lab v3.0.0 separately; set its checkout path:
export ISAACLAB_PATH=/path/to/IsaacLab-3.0.0
# Optional: export UV_BIN=/path/to/uv
./scripts/bootstrap.sh
./run.sh --profile smoke --mode smoke --num-envs 16 --steps 200 --log-dir "$PWD/logs/smoke"
```

`bootstrap.sh` synchronizes the Isaac Lab environment, fetches checksum-verified official motion data and converts the vendored G1 URDF to USD. InstinctLab is pinned as a submodule; `instinct_rl` is vendored. The submodule's Git LFS assets may require `git lfs pull` inside `vendor/InstinctLab`. Generated assets, motion releases, full training logs and checkpoints are excluded from Git.

## Training and evaluation

```bash
./run.sh --mode train --num-envs 2048 --iterations 20000 --log-dir "$PWD/logs/g1"
# Resume: iterations is the number of ADDITIONAL updates.
./run.sh --mode train --num-envs 2048 --iterations 2000 \
  --checkpoint "$PWD/logs/g1/model_15000.pt" --log-dir "$PWD/logs/resume"
```

Formal evaluation is implemented in `scripts/evaluate_checkpoints.py`; invoke it with the Isaac Lab Python environment and the same `PYTHONPATH` as `run.sh`. `--video --view-terrain ramp` selects the ramp camera. Checkpoints are not bundled; example paths in historical audits refer to local experiments.

```bash
PYTHONPATH="$PWD" "$ISAACLAB_PATH/.venv/bin/python" -m unittest discover -s tests -v
```

## Scope and limitations

The port includes terrain curricula, target commands, depth history, AMP reference windows, foot-edge penalties, PPO/MoE training and checkpoint evaluation. Physics/material semantics and some perception/reference-processing details differ from the original implementation. There is no hardware validation or multi-seed reproduction. Historical audit scripts may contain machine-specific paths; the main bootstrap/training entry points accept `ISAACLAB_PATH` and `UV_BIN`.

## Attribution and license

Unofficial; not an Isaac Lab or Project Instinct release. See [THIRD_PARTY.md](THIRD_PARTY.md) for pinned sources and modifications. The port follows **CC BY-NC 4.0**; inherited file-level licenses remain applicable. Upstream licenses/notices are retained. Motion data is obtained from the upstream release, not redistributed here.
