# Hiking in the Wild — Isaac Lab 3.0

Unitree G1 (29 DoF, shoes) parkour training on **Isaac Lab 3.0 / Newton MJWarp**, with depth perception, AMP, PPO and the upstream four-expert MoE policy. Based on [Hiking in the Wild](https://project-instinct.github.io/hiking-in-the-wild/). [中文说明](README_zh.md).

## Task

`Hiking-G1-AMP-Unfiltered-v0` is the only registered task and the default for training, play and evaluation. It retains all 18,982 source frames and all 18,973 complete ten-frame AMP windows, and plays continuously across every internal transition. The old filtered task, its configuration and its sampling branches have been removed.

The task uses 200 Hz physics, 50 Hz control, 26 reward terms, ten-frame AMP features and eight-frame policy/depth histories.

## Fix: retain the complete AMP reference

The original port rejected reference frames with root speed above 10 m/s or joint speed above 100 rad/s. This removed 490 of 18,972 candidate windows. During continuous playback, the same valid-start mask also resampled the reference up to nine frames before a flagged transition. Reset starts were restricted to the first 90% of the file, and the final complete window was omitted.

`Hiking-G1-AMP-Unfiltered-v0` removes those content-based rejection and resampling operations, includes the final complete window, and allows every source frame as a reset cursor. Root/joint velocities pass through without clipping. Frame conversion, joint-name mapping, feature scales, mirroring and episode/file-end resets remain part of reference construction. All entrypoints now use this sampling behavior; there is no filtering option.

The training entrypoint also accepts `--save-interval`. Resume restores the policy, discriminator, both optimizers, update counter and PPO adaptive learning-rate scalar. Local-only evaluation now loads only the requested model. `--official-checkpoint` evaluates a trained InstinctLab `.pt` with joint-name mapping for observation histories and actions.

## Isaac Lab 3.0 versus official InstinctLab

| Component | This implementation | Official training implementation |
|---|---|---|
| Runtime | Isaac Lab 3.0, Newton MJWarp, Python 3.12, PyTorch 2.11 | InstinctLab on Isaac Lab / Isaac Sim 5.1, PhysX, Python 3.11, PyTorch 2.7 |
| Physics integration | Newton `implicitfast`, two MJWarp substeps per physics step | PhysX articulation/contact solver |
| AMP reference | Direct NPZ loading, explicit WXYZ→XYZW and joint-name conversion; complete reference stream | `MotionReferenceManager` / `AmassMotion` reference pipeline |
| Depth | Lab3 ray-cast camera and local temporal buffer; clamp to 2.5 m before crop/resize/blur | InstinctLab ray-cast camera and sensor processing; crop/resize/blur before depth clipping |
| Material randomization | Newton friction API | PhysX static/dynamic friction with 64 material buckets |
| Learning | Upstream `instinct_rl` PPO + AMP, four-expert MoE, 24 rollout steps, five epochs, four minibatches | Same algorithm and network configuration |

Both use the G1 29-DoF model, terrain curriculum, 26 reward weights and 50% reference mirroring. Solver, contact/material handling and depth preprocessing differ between implementations.

### Measured training time

RTX 4090, 4096 environments, seed 42. The first 10,000 updates completed uninterrupted in both runs. Lab3 timing uses entrypoint elapsed time; official timing uses accumulated training-loop time (14.06 h including process startup):

| Stage | Lab3 / Newton | Official / PhysX |
|---|---:|---:|
| 0 → 10,000 updates | 11.85 h | 14.03 h |
| Average time per update | 4.27 s | 5.05 s |

Lab3 used approximately **15.5% less time** for this 10k stage. Its next 20k updates took 23.85 h, for 35.70 h across the two training stages. The official continuation was interrupted at update 23291 and recovered from checkpoint 23000; the final 7000-update training segment took 9.80 h. [Timing measurements and scope](results/30000/training_timing.json).

## Results: 4096 environments, 30000 updates

Both policies started with 10,000 updates and resumed for another 20,000, seed 42. The official policy was trained in InstinctLab/PhysX. Each evaluation column contains 9,000 trials: 100 environments per terrain, ten reset waves. The two Newton policies ran in one scene, with matching initial robot-state and target hashes for every wave.

| Terrain | Unfiltered Lab3 / Newton | Official / Newton | Official / PhysX |
|---|---:|---:|---:|
| Rough | 100.0% | 100.0% | 100.0% |
| Gap | 100.0% | 100.0% | 99.9% |
| Stair down | 99.9% | 99.6% | 100.0% |
| Platform down | 100.0% | 100.0% | 100.0% |
| Stair up | 100.0% | 100.0% | 100.0% |
| Platform up | 99.7% | 100.0% | 100.0% |
| Large box | 99.6% | 92.2% | 94.1% |
| Small box | 87.6% | 86.8% | 95.7% |
| Ramp | 100.0% | 100.0% | 99.9% |
| **Overall** | **98.53%** | **97.62%** | **98.84%** |

The official Newton and PhysX columns evaluate **the same PhysX-trained checkpoint** in two engines. Lab3 leads by 0.91 percentage points in the paired Newton evaluation; official native PhysX scores 98.84%, with the largest remaining terrain difference on small boxes. These are target-reaching success rates; the video and AMP audit describe motion behavior separately.

[Evaluation details and 10k comparisons](results/30000/README.md) · [Raw counts](results/30000/success_rates.json) · [Training curves](results/30000/training_comparison.png)

[45-second comparison video](artifacts/lab3_vs_official_30000.mp4): left unfiltered Lab3, right official policy; both run in Newton. Rough, gap and stair-up use the fixed first trial; the picture freezes just before each episode ends.

## Setup

Linux, RTX 4090 24 GB, Python 3.12, Isaac Lab `v3.0.0` (`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`), Newton 1.6.1 / MJWarp, Warp 1.17.0, PyTorch 2.11.0+cu130.

```bash
git clone --recurse-submodules https://github.com/Nagato-Yukii/hiking-in-the-wild-isaaclab3.git
cd hiking-in-the-wild-isaaclab3
export ISAACLAB_PATH=/path/to/IsaacLab-3.0.0
./scripts/bootstrap.sh
./run.sh --task Hiking-G1-AMP-Unfiltered-v0 --profile smoke --mode smoke \
  --num-envs 16 --steps 200 --log-dir "$PWD/logs/smoke"
```

The bootstrap synchronizes the Isaac Lab environment, fetches the motion data with SHA-256 verification and converts the vendored G1 URDF to USD. Run `git lfs pull` inside `vendor/InstinctLab` to fetch LFS assets. Generated robot assets, motion data, checkpoints and full training logs are stored locally.

## Train, resume and play

```bash
./run.sh --task Hiking-G1-AMP-Unfiltered-v0 --mode train \
  --num-envs 4096 --iterations 10000 --save-interval 1000 --log-dir "$PWD/logs/g1-10k"

# --iterations is the number of ADDITIONAL updates: 10000 + 20000 = 30000.
./run.sh --task Hiking-G1-AMP-Unfiltered-v0 --mode train \
  --num-envs 4096 --iterations 20000 --save-interval 1000 \
  --checkpoint "$PWD/logs/g1-10k/model_10000.pt" --log-dir "$PWD/logs/g1-30k"

PYGLET_HEADLESS=1 ./run.sh --task Hiking-G1-AMP-Unfiltered-v0 --mode play \
  --num-envs 1 --steps 750 --checkpoint "$PWD/logs/g1-30k/model_30000.pt" \
  --video --log-dir "$PWD/logs/play"
```

```bash
export PYTHONPATH="$PWD:$PWD/vendor/instinct_rl${PYTHONPATH:+:$PYTHONPATH}"
"$ISAACLAB_PATH/.venv/bin/python" scripts/evaluate_checkpoints.py \
  --checkpoint "$PWD/logs/g1-30k/model_30000.pt" \
  --policies local --per-terrain 100 --waves 10 --log-dir "$PWD/logs/eval-30k"

# Compare an official trained checkpoint; agent.json and joint_names.json live beside it.
"$ISAACLAB_PATH/.venv/bin/python" scripts/evaluate_checkpoints.py \
  --checkpoint "$PWD/logs/g1-30k/model_30000.pt" \
  --official-checkpoint /path/to/official/model_30000.pt \
  --policies local official --per-terrain 100 --waves 10 --log-dir "$PWD/logs/compare-30k"

"$ISAACLAB_PATH/.venv/bin/python" -m unittest discover -s tests -v
```

[Task and terrain configuration](LOCAL_SETUP_zh.md) · [Sources and licenses](THIRD_PARTY.md). License: **CC BY-NC 4.0**, with inherited file-level notices retained.
