from gym.envs.registration import register

register(
    id="placement_envs/NanoPlacementEnv",
    entry_point="placement_envs.envs:NanoPlacementEnv",
)
