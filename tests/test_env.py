from pathlib import Path
from unittest.mock import patch

import pytest
from gymnasium.utils.env_checker import check_env
from sb3_contrib import MaskablePPO

from mnt.nanoplacer.placement_envs.nano_placement_env import NanoPlacementEnv


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
    assert write_fgl.call_args.args[1] == str(Path("layouts/mux21_ONE_2DDWave_NanoPlaceR_Opt_UnOrd.fgl"))

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
    assert (tmp_path / "layouts/mux21_ONE_2DDWave_NanoPlaceR_Opt_UnOrd.fgl").is_file()

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
