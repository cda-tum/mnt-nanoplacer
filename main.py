import gym

import os
from ppo_masked import MaskablePPO
from ppo_masked.common.maskable.utils import get_action_masks
import placement_envs
from argparse import ArgumentParser


env_id = "placement_envs/NanoPlacementEnv-v0"
clocking_scheme = "2DDWave"
technology = "QCA"
minimal_layout_dimension = True  # if False, user specified layout dimensions are chosen
layout_width = 200
layout_height = 200
benchmark = "fontes18"
function = "parity"
time_steps = 1000000
reset_model = False
verbose = 0  # 0: Only show number of placed gates
#              1: print layout after every new best placement
#              2: print training metrics
#              3: print layout and training metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        choices=["fontes18", "trindade16", "EPFL", "TOY", "ISCAS85"],
        default=benchmark,
    )
    parser.add_argument("-f", "--function", type=str, default=function)
    parser.add_argument(
        "-c",
        "--clocking_scheme",
        type=str,
        choices=["2DDWave", "USE"],
        default=clocking_scheme,
    )
    parser.add_argument(
        "-t",
        "--technology",
        type=str,
        choices=["QCA", "SiDB"],
        default=technology,
    )
    parser.add_argument(
        "-l",
        "--minimal_layout_dimension",
        action="store_true",
        default=minimal_layout_dimension,
    )
    parser.add_argument(
        "-lw",
        "--layout_width",
        type=int,
        choices=range(1, 1000),
        default=layout_width,
    )
    parser.add_argument(
        "-lh",
        "--layout_height",
        type=int,
        choices=range(1, 1000),
        default=layout_height,
    )
    parser.add_argument("-ts", "--time_steps", type=int, default=time_steps)
    parser.add_argument("-r", "--reset_model", action="store_true", default=reset_model)
    parser.add_argument("-v", "--verbose", type=int, choices=[0, 1], default=verbose)
    args = parser.parse_args()

    if args.minimal_layout_dimension:
        pass
    print(args)

    env = gym.make(
        env_id,
        clocking_scheme=args.clocking_scheme,
        technology=technology,
        layout_width=args.layout_width,
        layout_height=args.layout_height,
        benchmark=args.benchmark,
        function=args.function,
        verbose=1 if args.verbose in (1, 3) else 0,
    )
    if args.reset_model or not os.path.exists(f"ppo_fiction_v8_{args.technology}_{args.function}_{args.clocking_scheme}"):
        model = MaskablePPO(
            "MlpPolicy",
            env,
            batch_size=512,
            verbose=1 if args.verbose (2, 3) else 0,
            gamma=0.995,
            learning_rate=0.001,
            tensorboard_log=f"./tensorboard/{args.function}/",
        )
        reset_num_timesteps = True
    else:
        model = MaskablePPO.load(
            os.path.join(
                "models",
                f"ppo_fiction_v8_{args.technology}_{args.function}_{args.clocking_scheme}",
            ),
            env,
        )
        reset_num_timesteps = False

    model.learn(
        total_timesteps=args.time_steps,
        log_interval=1,
        reset_num_timesteps=reset_num_timesteps,
    )
    # env.plot_placement_times()

    model.save(
        os.path.join(
            "models",
            f"ppo_fiction_{args.technology}_{args.function}_{args.clocking_scheme}",
        )
    )

    # reset environment
    obs = env.reset()
    terminated = False

    while not terminated:
        # calculate unfeasible layout positions
        action_masks = get_action_masks(env)

        # Predict coordinate for next gate based on the gate to be placed and the action mask
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

        # place gate, route it and receive reward of +1 if successful, 0 else
        # placement is terminated if no further feasible placement is possible
        obs, reward, terminated, info = env.step(action)

        # print current layout
        if args.verbose == 1:
            env.render()
