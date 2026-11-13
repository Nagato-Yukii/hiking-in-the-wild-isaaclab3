# SPDX-License-Identifier: CC-BY-NC-4.0
import gymnasium as gym

gym.register(
    id="Hiking-G1-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hikinglab3.config:HikingEnvCfg",
        "instinct_rl_cfg_entry_point": "hikinglab3.agent_cfg:G1ParkourPPORunnerCfg",
    },
)

gym.register(
    id="Hiking-G1-AMP-Unfiltered-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hikinglab3.config:HikingUnfilteredAmpEnvCfg",
        "instinct_rl_cfg_entry_point": "hikinglab3.agent_cfg:G1ParkourPPORunnerCfg",
    },
)
