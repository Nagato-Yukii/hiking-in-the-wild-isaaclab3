"""Count actual reward/history calls along the unchanged PPO rollout path; no learning."""
import json
from pathlib import Path
from collections import Counter
import torch
import gymnasium as gym
from isaaclab.app import launch_simulation
from hikinglab3.config import HikingEnvCfg
from hikinglab3.wrapper import InstinctRlVecEnvWrapper
from hikinglab3.motions import MotionDataset
from hikinglab3.agent_cfg import G1ParkourPPORunnerCfg
from instinct_rl.runners import OnPolicyRunner

out=Path('/home/ye/Desktop/HikingLab3/audit/reward_calls_20261105');out.mkdir(exist_ok=True)
cfg=HikingEnvCfg();cfg.scene.num_envs=8
cfg.scene.terrain.terrain_generator.num_rows=2
cfg.scene.terrain.max_init_terrain_level=1
cfg.episode_length_s=.12  # Force several timeout resets without changing the step/reward implementation.
cfg.sim.visualizer_cfgs=[];cfg.video_recorders=[]
with launch_simulation(cfg,{}):
    env=gym.make('Hiking-G1-AMP-v0',cfg=cfg).unwrapped
    try:
        wrapped=InstinctRlVecEnvWrapper(env)
        robot=env.scene['robot']
        wrapped.motion_dataset=MotionDataset(robot.joint_names,robot.data.default_joint_pos.torch,env.device)
        runner=OnPolicyRunner(wrapped,G1ParkourPPORunnerCfg().to_dict(),log_dir=None,device=env.device)
        runner.load('/home/ye/Desktop/HikingLab3/logs/g1-newton-2048-20k-20261104_062959/model_15000.pt')
        runner.alg.actor_critic.eval();runner.alg.discriminator.eval()
        obs,extras=wrapped.get_observations();critic=extras['observations']['critic']
        counts=Counter();dts=[];term_counts=Counter();errors=[];duplicate_errors=[];done_count=0;timeouts=0
        reward_compute=env.reward_manager.compute
        def reward_counted(dt):
            counts['task_reward_compute']+=1;dts.append(dt)
            reward=reward_compute(dt)
            errors.append(float((reward-env.reward_manager._step_reward.sum(-1)*dt).abs().max()))
            return reward
        env.reward_manager.compute=reward_counted
        for name,term in zip(env.reward_manager.active_terms,env.reward_manager._term_cfgs):
            original=term.func
            def counted(*a,_name=name,_func=original,**kw):
                term_counts[_name]+=1
                return _func(*a,**kw)
            term.func=counted
        aux=runner.alg.compute_auxiliary_reward
        def aux_counted(*a,**kw):
            counts['amp_reward_compute']+=1
            return aux(*a,**kw)
        runner.alg.compute_auxiliary_reward=aux_counted
        handle=runner.alg.discriminator.register_forward_hook(lambda *a:counts.update(['discriminator_forward']))
        history=env.observation_manager._group_obs_term_history_buffer['amp_policy']['joint_pos_rel']
        append=history.append
        def append_counted(*a,**kw):
            counts['amp_policy_history_append']+=1
            return append(*a,**kw)
        history.append=append_counted
        storage_errors=[];reference_advances=0
        with torch.inference_mode():
            for step in range(24):
                obs,critic,reward,done,infos=runner.rollout_step(obs,critic)
                done_count+=int(done.sum());timeouts+=int(infos['time_outs'].sum())
                amp=infos['step']['discriminator_reward']
                expected=reward+.25*amp+runner.alg.gamma*runner.alg.storage.values[step]*infos['time_outs'][:,None]
                storage_errors.append(float((runner.alg.storage.rewards[step]-expected).abs().max()))
                before_ref=infos['observations']['amp_reference'].clone()
                before_hist=history.buffer.clone()
                before_cursor=wrapped.motion_dataset._cursor.clone()
                before_counts=counts.copy()
                for _ in range(2):
                    _,extra_read=wrapped.get_observations()
                    duplicate_errors.append(float((extra_read['observations']['amp_reference']-before_ref).abs().max()))
                    duplicate_errors.append(float((history.buffer-before_hist).abs().max()))
                assert counts==before_counts, 'Observation read recomputed rewards/history'
                assert torch.equal(wrapped.motion_dataset._cursor,before_cursor)
        handle.remove()
        result={'policy_steps':24,'num_envs':8,'physics_dt':env.physics_dt,'policy_dt':env.step_dt,'decimation':cfg.decimation,'counts':dict(counts),'per_term_calls':dict(term_counts),'reward_dt_unique':sorted(set(dts)),'task_sum_max_error':max(errors),'stored_ppo_reward_max_error':max(storage_errors),'extra_observation_reads':48,'extra_read_history_max_error':max(duplicate_errors),'done_transitions':done_count,'timeout_transitions':timeouts,'optimizer_updates':0,'note':'8 environments, 2 terrain rows, 0.12s episodes to exercise timeout resets. Real runner.rollout_step; no optimizer updates.'}
        assert all(v==24 for v in counts.values()),result
        assert all(v==24 for v in term_counts.values()),result
        assert max(errors+storage_errors+duplicate_errors)<1e-6,result
        (out/'result.json').write_text(json.dumps(result,indent=2));print('AUDIT_RESULT',json.dumps(result),flush=True)
    finally:env.close()
