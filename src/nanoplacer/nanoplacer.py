import os
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

from src.nanoplacer.placement_envs import NanoPlacementEnv
from src.nanoplacer.placement_envs.utils import layout_dimensions


def create_layout(
    benchmark,
    function,
    clocking_scheme,
    technology,
    minimal_layout_dimension,
    layout_width,
    layout_height,
    time_steps,
    reset_model,
    verbose,
    optimize,
):
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
                "../../models",
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
    # env.plot_placement_times()

    model.save(
        os.path.join(
            "../../models",
            f"ppo_{technology}_{function}_{'ROW' if technology == 'SiDB' else clocking_scheme}",
        )
    )

    # reset environment
    obs, info = env.reset()
    terminated = False

    while not terminated:
        # calculate infeasible layout positions
        action_masks = get_action_masks(env)

        # Predict coordinate for next gate based on the gate to be placed and the action mask
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

        # place gate, route it and receive reward of +1 if successful, 0 else
        # placement is terminated if no further feasible placement is possible
        obs, reward, terminated, truncated, info = env.step(action)

        # print current layout
        if verbose == 1:
            env.render()


if __name__ == "__main__":
    clocking_scheme = "2DDWave"
    technology = "QCA"
    minimal_layout_dimension = True  # if False, user specified layout dimensions are chosen
    layout_width = 3
    layout_height = 4
    benchmark = "trindade16"
    function = "mux21"
    time_steps = 10000
    reset_model = True
    verbose = 0  # 0: Only show number of placed gates
    #              1: print layout after every new best placement
    #              2: print training metrics
    #              3: print layout and training metrics
    optimize = True

    create_layout(
        benchmark,
        function,
        clocking_scheme,
        technology,
        minimal_layout_dimension,
        layout_width,
        layout_height,
        time_steps,
        reset_model,
        verbose,
        optimize,
    )
