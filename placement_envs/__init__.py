from gym.envs.registration import register

register(
    id="placement_envs/NanoPlacementEnv-v0",
    entry_point="placement_envs.envs:NanoPlacementEnv",
)
