import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO

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
) -> None:
    effective_clocking_scheme = "2DDWave" if technology.lower() == "sidb" else clocking_scheme

    for folder in (Path("layouts"), Path("models"), Path("tensorboard")):
        folder.mkdir(parents=True, exist_ok=True)

    if minimal_layout_dimension:
        dimensions = layout_dimensions.get(effective_clocking_scheme, {}).get(benchmark, {}).get(function)
        if dimensions is None:
            msg = (
                f"No predefined layout dimensions for {benchmark}/{function} with the "
                f"{effective_clocking_scheme} clocking scheme"
            )
            raise ValueError(msg)
        layout_width, layout_height = dimensions

    env = NanoPlacementEnv(
        clocking_scheme=effective_clocking_scheme,
        technology=technology,
        layout_width=layout_width,
        layout_height=layout_height,
        benchmark=benchmark,
        function=function,
        verbose=1 if verbose in (1, 3) else 0,
        optimize=optimize,
    )

    model_path = Path("models") / (
        f"ppo_{technology}_{benchmark}_{function}_"
        f"{'ROW' if technology.lower() == 'sidb' else effective_clocking_scheme}_{layout_width}x{layout_height}.zip"
    )
    if reset_model or not model_path.exists():
        model = MaskablePPO(
            "MlpPolicy",
            env,
            batch_size=512,
            verbose=1 if verbose in (2, 3) else 0,
            gamma=0.995,
            learning_rate=0.001,
            tensorboard_log=str(
                Path("tensorboard")
                / f"{technology}_{benchmark}_{function}_{effective_clocking_scheme}_{layout_width}x{layout_height}"
            ),
        )
        reset_num_timesteps = True
    else:
        model = MaskablePPO.load(model_path, env=env)
        reset_num_timesteps = False

    model.learn(
        total_timesteps=time_steps,
        log_interval=1,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(model_path)


def start() -> None:
    parser = argparse.ArgumentParser(description="Place and route an FCN circuit with reinforcement learning.")
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
        "--clocking-scheme",
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
        "--minimal-layout-dimension",
        "--minimal_layout_dimension",
        action="store_true",
        help="If True, experimentally found minimal layout dimensions are used (defaults to False).",
    )
    parser.add_argument(
        "-lw",
        "--layout-width",
        "--layout_width",
        type=int,
        default=3,
        help="User defined layout width.",
    )
    parser.add_argument(
        "-lh",
        "--layout-height",
        "--layout_height",
        type=int,
        default=4,
        help="User defined layout height.",
    )
    parser.add_argument(
        "-ts",
        "--time-steps",
        "--time_steps",
        type=int,
        default=10000,
        help="Number of time steps to train the RL agent.",
    )
    parser.add_argument(
        "-r",
        "--reset-model",
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
        benchmark=args.benchmark,
        function=args.function,
        clocking_scheme=args.clocking_scheme,
        technology=args.technology,
        minimal_layout_dimension=args.minimal_layout_dimension,
        layout_width=args.layout_width,
        layout_height=args.layout_height,
        time_steps=args.time_steps,
        reset_model=args.reset_model,
        verbose=args.verbose,
        optimize=args.optimize,
    )


if __name__ == "__main__":
    start()
