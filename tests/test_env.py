from pathlib import Path
from unittest.mock import patch

import pytest
from gymnasium.utils.env_checker import check_env
from sb3_contrib import MaskablePPO

from mnt.nanoplacer.placement_envs.nano_placement_env import NanoPlacementEnv
from mnt.nanoplacer.placement_envs.utils import map_to_discrete


@pytest.fixture
def env() -> NanoPlacementEnv:
    return NanoPlacementEnv(
        clocking_scheme="2DDWave",
        technology="Gate-level",
        layout_width=3,
        layout_height=4,
        benchmark="trindade16",
        function="mux21",
        verbose=0,
        optimize=True,
    )


def test_reset_initializes_random_generator(env: NanoPlacementEnv) -> None:
    observation, info = env.reset(seed=42)

    assert observation == 0
    assert info == {}
    assert env.np_random is not None


def test_environment_follows_gymnasium_contract(
    env: NanoPlacementEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    check_env(env, skip_render_check=True)


def test_step_returns_gymnasium_values(env: NanoPlacementEnv) -> None:
    env.reset()
    observation, reward, terminated, truncated, info = env.step(0)

    assert isinstance(observation, int)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert info == {}


@pytest.mark.parametrize("action", [-1, 12])
def test_step_rejects_actions_outside_action_space(env: NanoPlacementEnv, action: int) -> None:
    with pytest.raises(ValueError, match="outside the action space"):
        env.step(action)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"technology": "InvalidTech"}, "Not a supported technology"),
        ({"layout_width": 0}, "Layout dimensions must be positive"),
        ({"layout_height": -1}, "Layout dimensions must be positive"),
    ],
)
def test_environment_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NanoPlacementEnv(**kwargs)


def test_save_layout_dispatches_by_technology(
    env: NanoPlacementEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    env.technology = "QCA"
    with (
        patch("mnt.pyfiction.apply_qca_one_library") as apply_qca,
        patch("mnt.pyfiction.write_qca_layout_svg") as write_qca,
    ):
        env.save_layout()
    apply_qca.assert_called_once_with(env.layout)
    assert write_qca.call_args.args[1] == str(Path("layouts/mux21_2DDWave_qca.svg"))

    env.technology = "SiDB"
    with (
        patch("mnt.pyfiction.hexagonalization") as hexagonalization,
        patch("mnt.pyfiction.write_dot_layout") as write_dot,
    ):
        env.save_layout()
    hexagonalization.assert_called_once_with(env.layout)
    assert write_dot.call_args.args[1] == str(Path("layouts/mux21_ROW_sidb.dot"))

    env.technology = "Gate-level"
    with patch("mnt.pyfiction.write_fgl_layout") as write_fgl:
        env.save_layout()
    assert write_fgl.call_args.args[1] == str(Path("layouts/mux21_ONE_2DDWave_NanoPlaceR_Opt_UnOrd_area.fgl"))

    env.technology = "InvalidTech"
    with pytest.raises(ValueError, match="Not a supported technology"):
        env.save_layout()


def test_place_and_serialize_mux(env: NanoPlacementEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    for action in (3, 6, 0, 1, 7, 2, 5, 8):
        env.step(action)
    _, reward, terminated, _, _ = env.step(11)

    assert reward > 1000
    assert terminated is True
    assert (tmp_path / "layouts/mux21_ONE_2DDWave_NanoPlaceR_Opt_UnOrd_area.fgl").is_file()

    env.technology = "QCA"
    env.save_layout()
    assert (tmp_path / "layouts/mux21_2DDWave_qca.svg").is_file()

    env.technology = "SiDB"
    env.save_layout()
    assert (tmp_path / "layouts/mux21_ROW_sidb.dot").is_file()


def test_action_masks_are_plain_booleans(env: NanoPlacementEnv) -> None:
    masks = env.action_masks()

    assert len(masks) == env.action_space.n
    assert all(isinstance(mask, bool) for mask in masks)
    assert any(masks)


def test_action_masks_return_safe_fallback_when_no_unoccupied_positions_remain(env: NanoPlacementEnv) -> None:
    env.occupied_tiles.fill(1)

    masks = env.action_masks()

    assert masks == [True] * env.action_space.n
    assert env.placement_possible is False


def test_action_masks_accept_all_four_edges_for_use_outputs() -> None:
    use_env = NanoPlacementEnv(
        clocking_scheme="USE",
        technology="Gate-level",
        layout_width=5,
        layout_height=5,
        benchmark="trindade16",
        function="mux21",
        verbose=0,
    )
    use_env.current_node = len(use_env.actions) - 1
    predecessor = next(iter(use_env.DG.predecessors(use_env.actions[use_env.current_node])))
    use_env.layout.create_pi("source", (2, 2))
    use_env.node_dict[predecessor] = use_env.layout.get_node((2, 2))
    use_env.occupied_tiles[2, 2] = 1

    with patch("mnt.pyfiction.a_star", return_value=[object()]):
        masks = use_env.action_masks()

    assert sum(masks) == 16
    assert all(masks[map_to_discrete(4, y, 5)] for y in range(5))


def test_action_masks_use_height_for_rectangular_routing_goal() -> None:
    use_env = NanoPlacementEnv(
        clocking_scheme="USE",
        technology="Gate-level",
        layout_width=5,
        layout_height=3,
        benchmark="trindade16",
        function="mux21",
        verbose=0,
    )
    use_env.current_node = len(use_env.actions) - 1
    predecessor = next(iter(use_env.DG.predecessors(use_env.actions[use_env.current_node])))
    use_env.layout.create_pi("source", (2, 1))
    use_env.node_dict[predecessor] = use_env.layout.get_node((2, 1))

    with patch("mnt.pyfiction.a_star", return_value=[object()]) as a_star:
        use_env.action_masks()

    assert a_star.call_args.args[2] == (0, 3)


def test_action_masks_accept_any_reachable_routing_goal() -> None:
    use_env = NanoPlacementEnv(
        clocking_scheme="USE",
        technology="Gate-level",
        layout_width=4,
        layout_height=3,
        benchmark="trindade16",
        function="mux21",
        verbose=0,
    )
    use_env.current_node = len(use_env.actions) - 1
    predecessor = next(iter(use_env.DG.predecessors(use_env.actions[use_env.current_node])))
    use_env.layout.create_pi("source", (2, 1))
    use_env.node_dict[predecessor] = use_env.layout.get_node((2, 1))

    with patch("mnt.pyfiction.a_star", side_effect=([object()], [])):
        use_env.action_masks()

    assert use_env.placement_possible is True


def test_action_masks_exclude_failed_route_position() -> None:
    retry_env = NanoPlacementEnv(
        clocking_scheme="2DDWave",
        technology="Gate-level",
        layout_width=5,
        layout_height=6,
        benchmark="trindade16",
        function="mux21",
        verbose=0,
    )
    for action in (3, 20, 10, 11, 28, 27):
        retry_env.step(action)

    assert retry_env.current_tries == 2
    masks = retry_env.action_masks()
    assert masks[28] is False
    assert masks[27] is False


def test_action_masks_allow_terminal_observation(env: NanoPlacementEnv) -> None:
    env.current_node = len(env.actions)

    assert env.observation_space.contains(env.current_node)
    assert env.action_masks() == [True] * env.action_space.n


def test_maskable_ppo_can_learn(env: NanoPlacementEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    model = MaskablePPO("MlpPolicy", env, n_steps=8, batch_size=8, seed=0, verbose=0)

    model.learn(total_timesteps=8)


def test_calculate_reward_is_deterministic_and_quiet(env: NanoPlacementEnv, capsys: pytest.CaptureFixture[str]) -> None:
    env.current_node = 1
    reward, terminated = env.calculate_reward(0, 0, placed_node=True)

    assert reward == 1.0
    assert terminated is False
    assert capsys.readouterr().out == ""
