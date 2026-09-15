from pathlib import Path
from unittest.mock import patch

import pytest
from gymnasium.utils.env_checker import check_env
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from mnt import pyfiction
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


def test_step_reuses_mask_count_for_both_gate_arities(env: NanoPlacementEnv, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for action in (3, 6, 0, 1, 7, 2, 5, 8, 11):
        masks = env.action_masks()
        gate_type = env.node_to_action[env.actions[env.current_node]]
        with patch.object(env, "action_masks", wraps=env.action_masks) as calculate_mask:
            env.step(action)
        calculate_mask.assert_not_called()
        if gate_type != "INPUT":
            assert env.max_tries == sum(masks)
        assert env._action_mask_count is None
    assert env.equivalent == "STRONG"


def test_direct_steps_recalculate_mask_count(env: NanoPlacementEnv) -> None:
    for action in (3, 6, 0):
        env.step(action)
    for action in (1, 7):
        with patch.object(env, "action_masks", wraps=env.action_masks) as calculate_mask:
            env.step(action)
        calculate_mask.assert_called_once()
        assert env._action_mask_count is None


def test_mask_count_is_invalidated_by_failed_step_and_reset(env: NanoPlacementEnv) -> None:
    env.step(0)
    env.action_masks()
    assert env._action_mask_count is not None
    assert env.step(0)[2]  # An occupied tile terminates without placing another node.
    assert env._action_mask_count is None
    env.action_masks()
    env.reset()
    assert env._action_mask_count is None
    assert env.layout_mask_width == env.layout_mask_height == 4


def test_repeated_mask_requests_recheck_mutated_placement(env: NanoPlacementEnv) -> None:
    env.action_masks()
    env.occupied_tiles.fill(1)
    assert env.action_masks() == [True] * env.action_space.n
    assert not env.placement_possible
    assert env._action_mask_count is None


def test_best_hook_precedes_partial_and_complete_vecenv_resets(
    env: NanoPlacementEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    snapshots = []
    env.on_best = lambda current: snapshots.append((current.current_node, current.equivalent, str(current.layout)))
    wrapped = DummyVecEnv([lambda: env])
    wrapped.reset()
    wrapped.step([0])
    _, _, done, _ = wrapped.step([0])  # Occupied tile ends this partial placement.
    assert done[0]
    assert env.current_node == 0
    assert snapshots[0][0] == 1
    assert snapshots[0][2] != str(env.layout)
    for action in (3, 6, 0, 1, 7, 2, 5, 8, 11):
        _, _, done, _ = wrapped.step([action])
    assert done[0]
    assert env.current_node == 0
    assert [snapshot[0] for snapshot in snapshots] == list(range(1, len(env.actions) + 1))
    assert snapshots[-1][1] == "STRONG"
    assert snapshots[-1][2] != str(env.layout)
    wrapped.close()


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


@pytest.mark.parametrize(("fallback", "blocked"), [(False, False), (True, False), (True, True)])
def test_reverse_routing_preserves_inputs_and_failed_obstructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback: bool, blocked: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    circuit = tmp_path / "and.v"
    circuit.write_text("module top(a, b, f);\ninput a, b;\noutput f;\nassign f = a & b;\nendmodule\n")
    network = pyfiction.read_technology_network(str(circuit))
    with patch("mnt.pyfiction.read_technology_network", return_value=network):
        routing_env = NanoPlacementEnv(
            layout_width=6,
            layout_height=3,
            technology="Gate-level",
            routing_fallback=fallback,
            optimize=False,
            verbose=0,
        )
    # Native fanin order may differ from PI order; fix the routing source positions.
    for pi in routing_env.actions[:2]:
        routing_env.step(1 if pi == routing_env._preceding_nodes[2][0] else 6)
    layout = routing_env.layout
    layout.obstruct_coordinate((5, 2, 1))  # An unrelated, pre-existing empty-tile mark.
    if blocked:
        layout.obstruct_coordinate((3, 0))  # Also blocks the alternative path for a.
    before = {
        (x, y, z): layout.is_obstructed_coordinate((x, y, z)) for x in range(6) for y in range(3) for z in range(2)
    }
    native_a_star = pyfiction.a_star
    route_sources = []

    def first_route_conflict(current_layout, source, target, params):
        if target == (4, 1):
            route_sources.append((source.x, source.y))
            if len(route_sources) == 1:
                # Pin one valid path instead of relying on native equal-cost tie-breaking.
                return [pyfiction.offset_coordinate(x, y, 0) for x, y in ((1, 0), (2, 0), (2, 1), (3, 1), (4, 1))]
        return native_a_star(current_layout, source, target, params)

    with patch("mnt.pyfiction.a_star", side_effect=first_route_conflict):
        routing_env.step(10)  # AND at (4, 1); only the first route is controlled.
    assert route_sources == [(1, 0), (0, 1)] + ([(0, 1), (1, 0)] if fallback else [])
    if fallback and not blocked:
        assert routing_env.current_node == 3
        ancestors = []
        for fanin in layout.fanins((4, 1)):
            ancestor = fanin
            while not layout.is_pi(layout.get_node(ancestor)):
                ancestor = layout.fanins(ancestor)[0]
            ancestors.append((ancestor.x, ancestor.y, ancestor.z))
        assert sorted(ancestors) == [(0, 1, 0), (1, 0, 0)]
        routing_env.step(11)  # PO at (5, 1).
        assert routing_env.equivalent == "STRONG"
    else:
        assert routing_env.current_node == 2
        assert routing_env.current_tries == 1
        if fallback:
            for coordinate, obstructed in before.items():
                if coordinate != (4, 1, 0):  # The failed gate itself was just placed here.
                    assert layout.is_obstructed_coordinate(coordinate) == obstructed
            # Moving a PI exposes its explicit mark, otherwise hidden by occupancy.
            layout.move_node(layout.get_node((1, 0)), (5, 2), [])
            assert layout.is_obstructed_coordinate((1, 0))


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
