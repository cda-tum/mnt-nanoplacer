from gym.envs.registration import register

register(
    id="fiction_env/QCAEnv-v6",
    entry_point="custom_envs.envs:QCAEnv6",
)

register(
    id="fiction_env/QCAEnv-v7",
    entry_point="custom_envs.envs:QCAEnv7",
)

register(
    id="fiction_env/QCAEnv-v8",
    entry_point="custom_envs.envs:QCAEnv8",
)

register(
    id="fiction_env/QCAEnv-v9",
    entry_point="custom_envs.envs:QCAEnv9",
)
