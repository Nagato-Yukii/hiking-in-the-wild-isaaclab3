"""Paired deterministic inference on fixed paper-sized terrain in Newton."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from isaaclab.app import launch_simulation
from isaaclab.managers import TerminationTermCfg
from hikinglab3.config import HikingEnvCfg
from hikinglab3.wrapper import InstinctRlVecEnvWrapper
from hikinglab3.official_policy import OfficialStairPolicy
import instinct_rl.modules as modules


def reached_target(env):
    command = env.command_manager.get_term('base_velocity')
    distance = torch.linalg.vector_norm(env.scene['robot'].data.root_pos_w.torch[:, :2]
                                        - command.pos_command_w[:, :2], dim=1)
    env._eval_distance = distance.clone()
    return distance <= 0.25


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--log-dir', required=True, type=Path)
    parser.add_argument('--per-terrain', type=int, default=100)
    parser.add_argument('--waves', type=int, default=10)
    parser.add_argument('--policies', nargs='+', default=['local', 'official'])
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--view-terrain', choices=['rough','gap','stair_down','platform_down','stair_up','platform_up','large_box','small_box','ramp'], default='stair_up')
    parser.add_argument('--amp-audit', action='store_true')
    parser.add_argument('--reward-audit', action='store_true', help='Record stair ascent state and weighted task rewards')
    args = parser.parse_args()
    log = args.log_dir.resolve(); log.mkdir(parents=True, exist_ok=False)
    cfg = HikingEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.per_terrain * 9
    gen = cfg.scene.terrain.terrain_generator
    del gen.sub_terrains['perlin_rough_stand']
    names = list(gen.sub_terrains)
    labels = ['rough', 'gap', 'stair_down', 'platform_down', 'stair_up', 'platform_up',
              'large_box', 'small_box', 'ramp']
    assert names == ['perlin_rough', 'square_gaps', 'pyramid_stairs', 'pyramid_stairs_high',
                     'pyramid_stairs_inv', 'pyramid_stairs_inv_high', 'boxes', 'mesh_boxes',
                     'hf_pyramid_slope_inv']
    gen.num_rows = 10; gen.num_cols = 9; gen.use_cache = True
    cfg.scene.terrain.max_init_terrain_level = 9
    for sub in gen.sub_terrains.values():
        sub.proportion = 1 / 9
        sub.wall_prob = [0.] * 4
    gen.sub_terrains['perlin_rough'].noise_scale = .05
    gen.sub_terrains['square_gaps'].gap_distance_range = (.6, .6)
    for key in ('pyramid_stairs', 'pyramid_stairs_inv'):
        gen.sub_terrains[key].step_height_range = (.2, .2)
    for key in ('pyramid_stairs_high', 'pyramid_stairs_inv_high'):
        gen.sub_terrains[key].step_height_range = (.4, .4)
    gen.sub_terrains['boxes'].obstacle_height_range = (.4, .4)
    gen.sub_terrains['mesh_boxes'].box_height_mean = .3
    gen.sub_terrains['hf_pyramid_slope_inv'].slope_range = (math.tan(.4), math.tan(.4))
    cfg.curriculum.terrain_levels = None
    cfg.observations.policy.enable_corruption = False
    cfg.events.physics_material = None
    cfg.events.reset_base.params['velocity_range'] = {k:(0.,0.) for k in cfg.events.reset_base.params['velocity_range']}
    cfg.events.reset_robot_joints.params = {'position_range':(0.,0.), 'velocity_range':(0.,0.)}
    command = cfg.commands.base_velocity
    command.velocity_ranges = {k:v for k,v in command.velocity_ranges.items() if k in names}
    command.random_velocity_terrain = []
    command.rel_standing_envs = 0.
    command.resampling_time_range = (100.,100.)
    command.target_dis_threshold = .2
    cfg.episode_length_s = 15.
    # The origin-relative root-height rule wrongly terminates legitimate descent;
    # official PLAY also removes it. Keep torso contact and tilt failure rules.
    cfg.terminations.root_height = None
    cfg.terminations.target_reached = TerminationTermCfg(func=reached_target, time_out=True)
    cfg.sim.visualizer_cfgs = []; cfg.video_recorders = []
    if args.video:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
        assert len(args.policies) == 1 and args.waves == 1, 'Record one policy/trial wave per video'
        cfg.sim.physics.load_visual_shapes = True
        cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(headless=True,window_width=960,window_height=720,streaming_view=False)]
        cfg.video_recorders = [VideoRecorderCfg(source='visualizer:newton_gl',output_dir=str(log/'videos'),video_length=750,fps=50)]
    (log/'env_config.json').write_text(json.dumps(cfg.to_dict(), default=str, indent=2))
    protocol = {'backend':'Newton/MJWarp', 'checkpoint':str(args.checkpoint),
                'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'labels':dict(zip(names,labels)), 'per_terrain_per_wave':args.per_terrain,
                'waves':args.waves,'seed':42,'success':'within 0.25m before 15s and without simultaneous physical failure',
                'command_max_forward_mps':.55, 'note':'Common Newton transfer comparison, paper-sized geometry; not a native PhysX paper reproduction. Official release is stair-specialized.'}
    (log/'protocol.json').write_text(json.dumps(protocol,indent=2))
    started = time.monotonic()
    with launch_simulation(cfg, {'enable_cameras':args.video}):
        env = gym.make('Hiking-G1-AMP-v0',cfg=cfg).unwrapped
        try:
            wrapped = InstinctRlVecEnvWrapper(env)
            robot = env.scene['robot']; terrain = env.scene.terrain
            view_env = int(torch.where(terrain.terrain_types == labels.index(args.view_terrain))[0][0])
            local_cfg = json.loads((args.checkpoint.parent/'agent.json').read_text())['policy']
            cls = local_cfg.pop('class_name')
            model = modules.build_actor_critic(cls,local_cfg,wrapped.get_obs_format(),29,1).to(env.device)
            model.load_state_dict(torch.load(args.checkpoint,map_location=env.device,weights_only=False)['model_state_dict'])
            model.eval()
            if args.amp_audit:
                from instinct_rl.modules.discriminator import Discriminator
                algorithm_cfg = json.loads((args.checkpoint.parent/'agent.json').read_text())['algorithm']
                disc = Discriminator(wrapped.get_obs_format()['amp_policy'], **algorithm_cfg['discriminator_kwargs']).to(env.device)
                disc.load_state_dict(torch.load(args.checkpoint,map_location=env.device,weights_only=False)['discriminator'])
                disc.eval()

            official = OfficialStairPolicy(robot.joint_names,env.device)
            (log/'adapter_validation.json').write_text(json.dumps(official.validation,indent=2))
            (log/'joint_names.json').write_text(json.dumps(robot.joint_names,indent=2))
            policies = {'local':model.act_inference,'official':official}
            cmd = env.command_manager.get_term('base_velocity')
            fixed_targets = cmd.valid_targets[terrain.terrain_levels,terrain.terrain_types]
            score = fixed_targets[:,:,0] - .2*(fixed_targets[:,:,1]-terrain.env_origins[:,None,1]).abs()
            goals = fixed_targets[torch.arange(env.num_envs,device=env.device),score.argmax(dim=1)].clone()
            results = {}
            all_records = []
            initial_signatures = {}
            with torch.inference_mode():
                for policy_name in args.policies:
                    records = []
                    reward_audit = []
                    for wave in range(args.waves):
                        wrapped.seed(4200+wave)
                        obs, _ = wrapped.reset()
                        cmd.pos_command_w[:] = goals
                        cmd.max_command_b[:] = torch.tensor([.55,.3,1.],device=env.device)
                        cmd.is_standing_env[:] = False
                        cmd._update_command()
                        env.observation_manager.reset()
                        env._depth_step = -1
                        obs, _ = wrapped.get_observations()
                        initial = torch.cat([robot.data.root_pos_w.torch,robot.data.root_quat_w.torch,
                                             robot.data.joint_pos.torch,goals],dim=1).cpu().numpy()
                        signature = hashlib.sha256(initial.tobytes()).hexdigest()
                        if wave in initial_signatures:
                            assert initial_signatures[wave] == signature, 'Unpaired initial states'
                        initial_signatures[wave] = signature
                        active = torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
                        outcome = torch.full((env.num_envs,),-1,device=env.device,dtype=torch.int64)
                        end_step = torch.full_like(outcome,750)
                        last_distance = torch.linalg.vector_norm(robot.data.root_pos_w.torch[:,:2]-goals[:,:2],dim=1)
                        start_distance = last_distance.clone()
                        failure_counts = {k:0 for k in ('base_contact','bad_orientation','terrain_out_bound')}
                        for step in range(750):
                            if args.video:
                                pos = robot.data.root_pos_w.torch[view_env].tolist()
                                env.sim.set_camera_view((pos[0]+4,pos[1]-4,pos[2]+2.2),(pos[0],pos[1],pos[2]-.1))
                            if args.reward_audit or args.amp_audit:
                                audit_ids = torch.where(active & (terrain.terrain_types == 4))[0]
                                audit_state = torch.cat((cmd.command[audit_ids], robot.data.root_lin_vel_b.torch[audit_ids], robot.data.root_pos_w.torch[audit_ids]), dim=1).cpu().tolist()
                            actions = policies[policy_name](obs)
                            assert torch.isfinite(actions).all()
                            obs, _, dones, extras = wrapped.step(actions)
                            if (args.reward_audit or args.amp_audit) and len(audit_ids):
                                weighted = env.reward_manager._step_reward[audit_ids].cpu().tolist()
                                distances = env._eval_distance[audit_ids].cpu().tolist()
                                if args.amp_audit:
                                    d = disc(extras['observations']['amp_policy'][audit_ids]).flatten()
                                    amp_values = list(zip(d.cpu().tolist(), (.25 * (1 - .25 * (d - 1).square()).clamp(min=0)).cpu().tolist()))
                                for idx, state, reward, distance in zip(audit_ids.cpu().tolist(), audit_state, weighted, distances):
                                    reward_audit.append({'wave':wave,'env':idx,'seconds':step*.02,'command':state[:3],'velocity':state[3:6],'position':state[6:],'distance_after':distance,'weighted_reward_per_second':dict(zip(env.reward_manager.active_terms,reward))})
                                    if args.amp_audit:
                                        value, amp = amp_values.pop(0)
                                        reward_audit[-1].update(discriminator=value, amp_reward_per_step=amp, done=bool(dones[idx]))
                            success = env.termination_manager.get_term('target_reached') & ~env.reset_terminated
                            finished = active & dones.bool()
                            last_distance[active] = env._eval_distance[active]
                            for term in failure_counts:
                                failure_counts[term] += int((active & env.termination_manager.get_term(term)).sum())
                            outcome[finished] = torch.where(success[finished],1,
                                torch.where(env.reset_terminated[finished] | env.termination_manager.get_term('terrain_out_bound')[finished],0,2))
                            end_step[finished] = step+1
                            active[finished] = False
                            if not active.any(): break
                        outcome[active] = 2
                        arrays = [v.cpu().tolist() for v in (terrain.terrain_types,outcome,end_step,start_distance,last_distance)]
                        for idx,(col,status,steps,sd,ed) in enumerate(zip(*arrays)):
                            records.append({'policy':policy_name,'wave':wave,'env':idx,'terrain':labels[col],
                                            'status':status,'seconds':steps*.02,'start_distance':sd,'end_distance':ed})
                        print('WAVE',policy_name,wave,'success',int((outcome==1).sum()),'failure',int((outcome==0).sum()),
                              'timeout',int((outcome==2).sum()),'seconds',round(time.monotonic()-started,1),flush=True)
                        (log/f'{policy_name}_episodes.json').write_text(json.dumps(records))
                    if args.reward_audit or args.amp_audit:
                        (log/f'{policy_name}_reward_audit.json').write_text(json.dumps(reward_audit))
                    summary = {}
                    for label in labels:
                        subset = [r for r in records if r['terrain']==label]
                        won = [r for r in subset if r['status']==1]
                        summary[label] = {'trials':len(subset),'successes':len(won),
                                          'success_rate':len(won)/len(subset),
                                          'mean_reaching_seconds':sum(r['seconds'] for r in won)/len(won) if won else None,
                                          'failures':sum(r['status']==0 for r in subset),
                                          'timeouts':sum(r['status']==2 for r in subset),
                                          'mean_end_distance':sum(r['end_distance'] for r in subset)/len(subset)}
                    results[policy_name] = summary
                    all_records += records
                    (log/'results.json').write_text(json.dumps(results,indent=2))
                    print('SUMMARY',policy_name,json.dumps(summary),flush=True)
            (log/'initial_state_signatures.json').write_text(json.dumps(initial_signatures,indent=2))
        finally:
            env.close()

if __name__ == '__main__':
    main()
