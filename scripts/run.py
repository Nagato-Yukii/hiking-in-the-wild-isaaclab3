"""Train, validate or play Hiking in the Wild on Isaac Lab 3.0."""

import argparse, json, time
from pathlib import Path
import torch
import gymnasium as gym
from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from hikinglab3.config import HikingEnvCfg

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["Hiking-G1-AMP-Unfiltered-v0"], default="Hiking-G1-AMP-Unfiltered-v0")
p.add_argument("--mode", choices=["smoke", "train", "play"], default="smoke")
p.add_argument("--profile", choices=["original", "smoke"], default="original")
p.add_argument("--num-envs", type=int)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--diagnostics", action="store_true")
p.add_argument("--steps", type=int, default=100)
p.add_argument("--iterations", type=int)
p.add_argument("--save-interval", type=int, help="Checkpoint interval in training updates")
p.add_argument("--terrain-rows", type=int)
p.add_argument("--terrain-cols", type=int)
p.add_argument("--max-init-terrain-level", type=int)
p.add_argument("--checkpoint")
p.add_argument(
    "--play-settings",
    choices=["original", "training"],
    default="original",
    help="Use official PLAY settings or retain training terrain/termination settings for diagnostics",
)
p.add_argument("--log-dir", default=str(Path(__file__).resolve().parents[1] / "logs/run"))
p.add_argument("--video", action="store_true")
p.add_argument("--camera-distance", type=float, default=4.0, help="Wider context around the 8 m terrain tile")
p.add_argument("--view-env", type=int, default=0)
args = p.parse_args()
if args.save_interval is not None and args.save_interval <= 0:
    p.error("--save-interval must be positive")
if args.camera_distance <= 0:
    p.error("--camera-distance must be positive")
if args.num_envs is None:
    args.num_envs = 4096 if args.mode == "train" and args.profile == "original" else (10 if args.mode == "play" else 16)
if not 0 <= args.view_env < args.num_envs:
    p.error("--view-env must index an active environment")
if args.iterations is None:
    args.iterations = 30000 if args.profile == "original" else 10
if args.terrain_rows is None:
    args.terrain_rows = (
        2 if args.profile == "smoke" else (4 if args.mode == "play" and args.play_settings == "original" else 10)
    )
if args.terrain_cols is None:
    args.terrain_cols = 10 if args.mode == "play" and args.play_settings == "original" else 20
if args.max_init_terrain_level is None:
    args.max_init_terrain_level = 0 if args.profile == "smoke" else 5
if args.checkpoint:
    prior = Path(args.checkpoint).parent / "result.json"
    if prior.exists() and any(
        json.loads(prior.read_text()).get(key) is False for key in ("valid_for_locomotion", "valid_for_training")
    ):
        raise ValueError("This checkpoint has known simulation/reward defects; use a learning-space-final checkpoint.")
log = Path(args.log_dir)
log.mkdir(parents=True, exist_ok=True)
cfg = HikingEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.terrain.terrain_generator.num_rows = args.terrain_rows
cfg.scene.terrain.terrain_generator.num_cols = args.terrain_cols
cfg.scene.terrain.max_init_terrain_level = args.max_init_terrain_level
cfg.seed = args.seed
if args.mode == "play" and args.profile == "original" and args.play_settings == "original":
    # Original G1ParkourEnvCfg_PLAY: do not apply these changes to training.
    for sub in cfg.scene.terrain.terrain_generator.sub_terrains.values():
        sub.wall_prob = [0.0, 0.0, 0.0, 0.0]
    cfg.episode_length_s = 10
    cfg.terminations.root_height = None
    cfg.events.physics_material = None
    cfg.events.reset_robot_joints.params = {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}
cfg.sim.visualizer_cfgs = []
cfg.video_recorders = []
if args.video:
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    cfg.sim.physics.load_visual_shapes = True
    cfg.sim.visualizer_cfgs = [
        NewtonGLVisualizerCfg(headless=True, window_width=1280, window_height=720, streaming_view=False)
    ]
    cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:newton_gl", output_dir=str(log / "videos"), video_length=args.steps, fps=50)
    ]
(log / "arguments.json").write_text(json.dumps(vars(args), indent=2))
(log / "env_config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
import hashlib

source_root = Path(__file__).resolve().parents[1]
(log / "source_sha256.json").write_text(
    json.dumps(
        {
            str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("hikinglab3", "scripts")
            for path in (source_root / folder).rglob("*.py")
        },
        indent=2,
    )
)
with launch_simulation(cfg, {"enable_cameras": args.video}):
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        from hikinglab3.wrapper import InstinctRlVecEnvWrapper

        wrapped = InstinctRlVecEnvWrapper(env)
        robot = env.scene["robot"]
        assert not robot.is_fixed_base, "Locomotion requires a floating-base articulation"
        assert robot.num_joints == 29
        assert (robot.data.joint_velocity_limits.torch > 0).all(), (
            "Missing actuator speed ratings would penalize all motion"
        )
        if args.diagnostics:
            state = {"joint_names": robot.joint_names}
            for key in (
                "soft_joint_vel_limits",
                "joint_velocity_limits",
                "joint_effort_limits",
                "joint_stiffness",
                "joint_damping",
                "default_joint_pos",
                "joint_armature",
            ):
                try:
                    value = getattr(robot.data, key)
                    value = value.torch if hasattr(value, "torch") else value
                    state[key] = value[0].tolist()
                except AttributeError:
                    pass
            (log / "actuator_diagnostics.json").write_text(json.dumps(state, indent=2))
        if args.checkpoint:
            prior_motion = Path(args.checkpoint).parent / "motion_manifest.json"
            if prior_motion.exists() and json.loads(prior_motion.read_text())["joint_names"] != robot.joint_names:
                raise ValueError("Checkpoint joint order differs from the active articulation")
        (log / "motion_manifest.json").write_text(json.dumps(wrapped.motion_dataset.manifest, indent=2))
        (log / "observation_format.json").write_text(json.dumps(wrapped.get_obs_format(), indent=2))
        if args.mode in ("train", "play"):
            from instinct_rl.runners import OnPolicyRunner
            from hikinglab3.agent_cfg import G1ParkourPPORunnerCfg

            acfg = G1ParkourPPORunnerCfg().to_dict()
            acfg["save_interval"] = max(
                1, min(args.save_interval if args.save_interval is not None else acfg["save_interval"], args.iterations)
            )
            (log / "agent.json").write_text(json.dumps(acfg, indent=2))
            runner = OnPolicyRunner(wrapped, acfg, log_dir=str(log), device=env.device)
            if args.checkpoint:
                runner.load(args.checkpoint)
                # The optimizer restores its LR, but upstream PPO keeps a separate
                # adaptive-KL scalar initialized from config. Resume both together.
                if args.mode == "train":
                    restored_lrs = {group["lr"] for group in runner.alg.optimizer.param_groups}
                    if len(restored_lrs) != 1:
                        raise ValueError("Cannot resume adaptive PPO with heterogeneous optimizer learning rates")
                    runner.alg.learning_rate = restored_lrs.pop()
                    (log / "resume_state.json").write_text(json.dumps({
                        "checkpoint": str(Path(args.checkpoint).resolve()),
                        "start_iteration": runner.current_learning_iteration,
                        "additional_iterations": args.iterations,
                        "target_iteration": runner.current_learning_iteration + args.iterations,
                        "restored_learning_rate": runner.alg.learning_rate,
                        "environment_state_restored": False,
                    }, indent=2))
            if args.mode == "play" and not args.checkpoint:
                raise ValueError("Playback requires --checkpoint")
        if args.mode == "train":
            before = {k: v.detach().clone() for k, v in runner.alg.discriminator.state_dict().items()}
            start_iteration = runner.current_learning_iteration
            start = time.monotonic()
            runner.learn(args.iterations, init_at_random_ep_len=True)
            updated = any(not torch.equal(v, runner.alg.discriminator.state_dict()[k]) for k, v in before.items())
            assert updated, "AMP discriminator did not update"
            assert all(
                torch.isfinite(v).all()
                for model in (runner.alg.actor_critic, runner.alg.discriminator)
                for v in model.state_dict().values()
            )
            report = {
                "mode": "train",
                "iterations": args.iterations,
                "num_envs": args.num_envs,
                "seconds": time.monotonic() - start,
                "start_iteration": start_iteration,
                "end_iteration": runner.current_learning_iteration,
                "discriminator_updated": updated,
                "finite_parameters": True,
                "task": args.task,
            }
        else:
            obs, extras = wrapped.reset()
            failures = 0
            termination_counts = {name: 0 for name in env.termination_manager.active_terms}
            diag = []
            episode_steps = torch.zeros(args.num_envs, device=env.device)
            completed_lengths = []
            with torch.inference_mode():
                for i in range(args.steps):
                    action = (
                        torch.zeros(args.num_envs, wrapped.num_actions, device=env.device)
                        if args.mode == "smoke"
                        else runner.alg.actor_critic.act_inference(obs)
                    )
                    if args.video:
                        pos = robot.data.root_pos_w.torch[args.view_env].cpu().tolist()
                        d = args.camera_distance
                        env.sim.set_camera_view(
                            (pos[0] + d, pos[1] - d, pos[2] + d * 0.55), (pos[0], pos[1], pos[2] - 0.1)
                        )
                    if args.diagnostics and i < 100:
                        diag.append(
                            {
                                "step": i,
                                "root": robot.data.root_pos_w.torch[0].tolist(),
                                "quat": robot.data.root_quat_w.torch[0].tolist(),
                                "ang_vel": robot.data.root_ang_vel_b.torch[0].tolist(),
                                "joint_vel_max": robot.data.joint_vel.torch[0].abs().max().item(),
                                "action_max": action[0].abs().max().item(),
                            }
                        )
                    obs, rew, dones, extras = wrapped.step(action)
                    episode_steps += 1
                    completed_lengths.extend(episode_steps[dones.bool()].cpu().tolist())
                    episode_steps[dones.bool()] = 0
                    for name in termination_counts:
                        termination_counts[name] += int(env.termination_manager.get_term(name).sum())
                    failures += int(env.reset_terminated.sum())
                    assert torch.isfinite(rew).all() and torch.isfinite(obs).all()
                    assert extras["observations"]["amp_policy"].shape == extras["observations"]["amp_reference"].shape
                    assert torch.isfinite(extras["observations"]["amp_reference"]).all()
                    if i % 20 == 0:
                        print("STEP", i, "reward", rew.mean().item(), flush=True)
            report = {
                "mode": args.mode,
                "steps": args.steps,
                "num_envs": args.num_envs,
                "finite_observations": True,
                "failure_resets": failures,
                "termination_counts": termination_counts,
                "completed_episode_mean_seconds": (sum(completed_lengths) / len(completed_lengths) * env.step_dt)
                if completed_lengths
                else None,
                "completed_episode_count": len(completed_lengths),
                "amp_shape": list(extras["observations"]["amp_policy"].shape),
                "depth_min": float(env._depth_output.min()),
                "depth_max": float(env._depth_output.max()),
                "edge_count": len(env._edge_detector.edges_pyt),
                "root_height_mean": float(robot.data.root_pos_w.torch[:, 2].mean()),
                "fixed_base": bool(robot.is_fixed_base),
            }
        if args.mode != "train" and args.diagnostics:
            (log / "diagnostics.json").write_text(json.dumps(diag, indent=2))
        (log / "result.json").write_text(json.dumps(report, indent=2))
        print("RESULT", json.dumps(report), flush=True)
    finally:
        env.close()
