import gym

import os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import torch as th
import custom_envs

# dir_path = os.path.dirname(os.path.realpath(__file__))
env_id = "fiction_env/QCAEnv-v6-USE"
clocking_scheme = "2DDWave"
layout_width = 16
layout_height = 14
benchmark = "fontes18"
function = "xor5_r1"


if __name__ == "__main__":
    env = gym.make(
        env_id,
        clocking_scheme=clocking_scheme,
        layout_width=layout_width,
        layout_height=layout_height,
        benchmark=benchmark,
        function=function,
        disable_env_checker=True,
    )
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 128])
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, batch_size=512)  # , policy_kwargs=policy_kwargs)
    # model.save(os.path.join("models", f"ppo_fiction_v6_{function}"))
    # model = MaskablePPO.load(os.path.join("models", f"ppo_fiction_v6_{function}"), env)
    model.learn(total_timesteps=300000, log_interval=100)
    env.plot_placement_times()

    model.save(os.path.join("models", f"ppo_fiction_v6_{function}"))
    obs = env.reset()

    terminated = False
    reward = 0
    while not terminated:
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        env.render()
    env.create_cell_layout()
