from gym.envs.registration import register

register(
    id="fiction_env/QCAEnv-v6",
    entry_point="custom_envs.envs:QCAEnv6",
)

register(
    id="fiction_env/QCAEnv-v6-USE",
    entry_point="custom_envs.envs:QCAEnv6USE",
)
