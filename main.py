import gym

import os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import torch as th
import custom_envs
from argparse import ArgumentParser


# dir_path = os.path.dirname(os.path.realpath(__file__))
env_id = "fiction_env/QCAEnv-v8"
clocking_scheme = "2DDWave"
layout_width = 50
layout_height = 50
benchmark = "fontes18"
function = "parity"
time_steps = 200000
mode = "TRAIN"  # "INIT", "TRAIN"
save = True
verbose = 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", "--benchmark", type=str, choices=["fontes18", "trindade16", "EPFL", "TOY", "ISCAS85"], default=benchmark)
    parser.add_argument("-f", "--function", type=str, default=function)
    parser.add_argument("-c", "--clocking_scheme", type=str, choices=["2DDWave", "USE"], default=clocking_scheme)
    parser.add_argument("-lw", "--layout_width", type=int, choices=range(1, 1000), default=layout_width)
    parser.add_argument("-lh", "--layout_height", type=int, choices=range(1, 1000), default=layout_height)
    parser.add_argument("-t", "--time_steps", type=int, default=time_steps)
    parser.add_argument("-m", "--mode", type=str, choices=["INIT", "TRAIN"], default=mode)
    parser.add_argument("-s", "--save", type=bool, default=save)
    parser.add_argument("-v", "--verbose", type=int, choices=[0, 1], default=verbose)
    args = parser.parse_args()
    print(args)

    env = gym.make(
        env_id,
        clocking_scheme=args.clocking_scheme,
        layout_width=args.layout_width,
        layout_height=args.layout_height,
        benchmark=args.benchmark,
        function=args.function,
        verbose=args.verbose,
        disable_env_checker=True,
    )
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 128])
    if args.mode == "INIT":
        model = MaskablePPO("MultiInputPolicy",
                            env,
                            batch_size=512,
                            verbose=1,
                            gamma=0.995,
                            learning_rate=0.001,
                            tensorboard_log=f"./tensorboard/{args.function}/",
                            # policy_kwargs = policy_kwargs,
                            create_eval_env=False,
                            )
    elif args.mode == "TRAIN":
        model = MaskablePPO.load(os.path.join("models", f"ppo_fiction_v8_{args.function}"), env)
    else:
        raise Exception

    model.learn(total_timesteps=args.time_steps, log_interval=1, reset_num_timesteps=False)
    # env.plot_placement_times()

    if args.save:
        model.save(os.path.join("models", f"ppo_fiction_v8_{args.function}"))

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
    env.create_cell_layout()
