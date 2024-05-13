import argparse
import os
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from mnt.nanoplacer.placement_envs.nano_placement_env import NanoPlacementEnv
from mnt.nanoplacer.placement_envs.utils import layout_dimensions


def create_layout(
    benchmark: str = "trindade16",
    function: str = "mux21",
    clocking_scheme: str = "2DDWave",
    technology: str = "Gate-level",
    minimal_layout_dimension: bool = True,
    layout_width: int = 3,
    layout_height: int = 4,
    time_steps: int = 10000,
    reset_model: bool = True,
    verbose: int = 1,
    optimize: bool = True,
):
    for folder in ["layouts", "models", "tensorboard"]:
        if not Path.exists(Path(folder)):
            Path.mkdir(Path(folder), parents=True)

    if minimal_layout_dimension:
        if function in layout_dimensions[clocking_scheme][benchmark]:
            layout_width, layout_height = layout_dimensions[clocking_scheme][benchmark][function]
        else:
            error_message = f"No predefined layout dimensions for {function} available"
            raise Exception(error_message)

    env = NanoPlacementEnv(
        clocking_scheme=clocking_scheme,
        technology=technology,
        layout_width=layout_width,
        layout_height=layout_height,
        benchmark=benchmark,
        function=function,
        verbose=1 if verbose in (1, 3) else 0,
        optimize=optimize,
    )

    if reset_model or not Path.exists(
        Path(f"ppo_{technology}_{function}_{'ROW' if technology == 'SiDB' else clocking_scheme}")
    ):
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            batch_size=512,
            verbose=1 if verbose in (2, 3) else 0,
            gamma=0.995,
            learning_rate=0.001,
            tensorboard_log=f"./tensorboard/{function}/",
        )
        reset_num_timesteps = True
    else:
        model = MaskablePPO.load(
            os.path.join(
                "models",
                f"ppo_{technology}_{function}_{'ROW' if technology == 'SiDB' else clocking_scheme}",
            ),
            env,
        )
        reset_num_timesteps = False

    model.learn(
        total_timesteps=time_steps,
        log_interval=1,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(
        os.path.join(
            "models",
            f"ppo_{technology}_{function}_{'ROW' if technology == 'SiDB' else clocking_scheme}",
        )
    )


def start():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        choices=["fontes18", "trindade16", "EPFL", "TOY", "ISCAS85"],
        default="trindade16",
        help="Benchmark set.",
    )
    parser.add_argument(
        "-f",
        "--function",
        type=str,
        default="mux21",
        help="Logic function to generate layout for.",
    )
    parser.add_argument(
        "-c",
        "--clocking_scheme",
        type=str,
        choices=["2DDWave", "USE", "RES", "ESR"],
        default="2DDWave",
        help="Underlying clocking scheme.",
    )
    parser.add_argument(
        "-t",
        "--technology",
        type=str,
        choices=["QCA", "SiDB", "Gate-level"],
        default="Gate-level",
        help="Underlying technology (QCA, SiDB or technology-independent Gate-level).",
    )
    parser.add_argument(
        "-l",
        "--minimal_layout_dimension",
        action="store_true",
        help="If True, experimentally found minimal layout dimensions are used (defaults to False).",
    )
    parser.add_argument(
        "-lw",
        "--layout_width",
        type=int,
        default=3,
        help="User defined layout width.",
    )
    parser.add_argument(
        "-lh",
        "--layout_height",
        type=int,
        default=4,
        help="User defined layout height.",
    )
    parser.add_argument(
        "-ts",
        "--time_steps",
        type=int,
        default=10000,
        help="Number of time steps to train the RL agent.",
    )
    parser.add_argument(
        "-r",
        "--reset_model",
        action="store_true",
        help="If True, reset saved model and train from scratch (defaults to False).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="0: No information. 1: Print layout after every new best placement. "
        "2: Print training metrics. 3: 1 and 2 combined.",
    )
    parser.add_argument(
        "-o",
        "--optimize",
        action="store_true",
        help="If True, layout will be further optimized after placement.",
    )
    args = parser.parse_args()
    create_layout(
        args.benchmark,
        args.function,
        args.clocking_scheme,
        args.technology,
        args.minimal_layout_dimension,
        args.layout_width,
        args.layout_height,
        args.time_steps,
        args.reset_model,
        args.verbose,
        args.optimize,
    )


if __name__ == "__main__":
    benchmark = "trindade16"
    function = "mux21"
    clocking_scheme = "2DDWave"
    technology = "QCA"
    minimal_layout_dimension = False  # if False, user specified layout dimensions are chosen
    layout_width = 3
    layout_height = 4
    time_steps = 10000
    reset_model = True
    verbose = 1  # 0: Only show number of placed gates
    #              1: print layout after every new best placement
    #              2: print training metrics
    #              3: print layout and training metrics
    optimize = True

    create_layout(
        benchmark=benchmark,
        function=function,
        clocking_scheme=clocking_scheme,
        technology=technology,
        minimal_layout_dimension=minimal_layout_dimension,
        layout_width=layout_width,
        layout_height=layout_height,
        time_steps=time_steps,
        reset_model=reset_model,
        verbose=verbose,
        optimize=optimize,
    )
