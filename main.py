import gym

import os
from ppo_masked import MaskablePPO
from ppo_masked.common.maskable.utils import get_action_masks
import placement_envs
from argparse import ArgumentParser
from placement_envs.utils import layout_dimensions


env_id = "placement_envs/NanoPlacementEnv-v0"
clocking_scheme = "2DDWave"
technology = "SiDB"
minimal_layout_dimension = True  # if False, user specified layout dimensions are chosen
layout_width = 200
layout_height = 200
benchmark = "trindade16"
function = "mux21"
time_steps = 10000
reset_model = False
verbose = 3  # 0: Only show number of placed gates
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
        help="Benchmark set.",
    )
    parser.add_argument(
        "-f",
        "--function",
        type=str,
        default=function,
        help="Logic function to generate layout for.",
    )
    parser.add_argument(
        "-c",
        "--clocking_scheme",
        type=str,
        choices=["2DDWave", "USE", "RES"],
        default=clocking_scheme,
        help="Underlying clocking scheme.",
    )
    parser.add_argument(
        "-t",
        "--technology",
        type=str,
        choices=["QCA", "SiDB"],
        default=technology,
        help="Underlying technology (QCA or SiDB).",
    )
    parser.add_argument(
        "-l",
        "--minimal_layout_dimension",
        action="store_true",
        default=minimal_layout_dimension,
        help="If True, experimentally found minimal layout dimensions are used.",
    )
    parser.add_argument(
        "-lw",
        "--layout_width",
        type=int,
        default=layout_width,
        help="User defined layout width.",
    )
    parser.add_argument(
        "-lh",
        "--layout_height",
        type=int,
        default=layout_height,
        help="User defined layout height.",
    )
    parser.add_argument(
        "-ts",
        "--time_steps",
        type=int,
        default=time_steps,
        help="Number of time steps to train the RL agent.",
    )
    parser.add_argument(
        "-r",
        "--reset_model",
        action="store_true",
        default=reset_model,
        help="If True, reset saved model and train from scratch.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 1],
        default=verbose,
        help="0: No information. 1: Print layout after every new best placement. "
        "2: Print training metrics. 3: 1 and 2 combined.",
    )
    args = parser.parse_args()

    if args.minimal_layout_dimension and args.function in layout_dimensions[args.clocking_scheme][args.benchmark]:
        args.layout_width, args.layout_height = layout_dimensions[args.clocking_scheme][args.benchmark][args.function]
    else:
        raise Exception(f"No predefined layout dimensions for {args.function} available")

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
    if args.reset_model or not os.path.exists(
        f"ppo_{args.technology}_{args.function}_{args.clocking_scheme}"
    ):
        model = MaskablePPO(
            "MlpPolicy",
            env,
            batch_size=512,
            verbose=1 if args.verbose in (2, 3) else 0,
            gamma=0.995,
            learning_rate=0.001,
            tensorboard_log=f"./tensorboard/{args.function}/",
        )
        reset_num_timesteps = True
    else:
        model = MaskablePPO.load(
            os.path.join(
                "models",
                f"ppo_{args.technology}_{args.function}_{args.clocking_scheme}",
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
            f"ppo_{args.technology}_{args.function}_{args.clocking_scheme}",
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
