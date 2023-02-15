import gym

import os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import torch as th
import custom_envs

# dir_path = os.path.dirname(os.path.realpath(__file__))
env_id = "fiction_env/QCAEnv-v8"
clocking_scheme = "2DDWave"
layout_width = 100
layout_height = 100
benchmark = "ISCAS85"
function = "c432"


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
    model = MaskablePPO("MultiInputPolicy",
                        env,
                        batch_size=512,
                        verbose=1,
                        gamma=0.995,
                        learning_rate=0.001,
                        tensorboard_log=f"./tensorboard/{function}/",
                        # policy_kwargs = policy_kwargs
                        )
    # model.save(os.path.join("models", f"ppo_fiction_v8_{function}"))
    # model = MaskablePPO.load(os.path.join("models", f"ppo_fiction_v8_{function}"), env)
    model.learn(total_timesteps=250000, log_interval=1) #, reset_num_timesteps=False)
    # env.plot_placement_times()

    model.save(os.path.join("models", f"ppo_fiction_v8_{function}"))
    obs = env.reset()

    # actions = [1, 5, 16, 20, 22]
    terminated = False
    reward = 0
    i = 0
    while not terminated: # and i < len(actions):
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        # action = actions[i]
        obs, reward, terminated, info = env.step(action)
        # env.render()
        i += 1
    # env.create_cell_layout()
