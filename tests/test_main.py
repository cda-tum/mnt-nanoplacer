import sys
from pathlib import Path
from unittest.mock import ANY, Mock, patch

import pytest

from mnt.nanoplacer.main import create_layout, start


def test_create_layout_starts_and_saves_a_new_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    env = Mock()
    model = Mock()

    with (
        patch("mnt.nanoplacer.main.NanoPlacementEnv", return_value=env),
        patch("mnt.nanoplacer.main.MaskablePPO", return_value=model) as ppo,
    ):
        create_layout(minimal_layout_dimension=False, time_steps=42, reset_model=True)

    ppo.assert_called_once_with(
        ANY,
        env,
        batch_size=512,
        verbose=0,
        gamma=0.995,
        learning_rate=0.001,
        tensorboard_log=str(Path("tensorboard") / "mux21"),
    )
    model.learn.assert_called_once_with(total_timesteps=42, log_interval=1, reset_num_timesteps=True)
    model.save.assert_called_once_with(Path("models/ppo_Gate-level_trindade16_mux21_2DDWave_3x4.zip"))
    assert all((tmp_path / directory).is_dir() for directory in ("layouts", "models", "tensorboard"))


def test_create_layout_resumes_the_saved_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    model_path = Path("models/ppo_Gate-level_trindade16_mux21_2DDWave_3x4.zip")
    model_path.parent.mkdir()
    model_path.touch()
    env = Mock()
    model = Mock()

    with (
        patch("mnt.nanoplacer.main.NanoPlacementEnv", return_value=env),
        patch("mnt.nanoplacer.main.MaskablePPO") as ppo,
    ):
        ppo.load.return_value = model
        create_layout(minimal_layout_dimension=False, time_steps=42, reset_model=False)

    ppo.assert_not_called()
    ppo.load.assert_called_once_with(model_path, env=env)
    model.learn.assert_called_once_with(total_timesteps=42, log_interval=1, reset_num_timesteps=False)
    model.save.assert_called_once_with(model_path)


def test_create_layout_uses_iscas85_dimensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    with (
        patch("mnt.nanoplacer.main.NanoPlacementEnv") as env,
        patch("mnt.nanoplacer.main.MaskablePPO"),
    ):
        create_layout(benchmark="ISCAS85", function="c17", time_steps=0)

    assert env.call_args.kwargs["layout_width"] == 7
    assert env.call_args.kwargs["layout_height"] == 7


def test_create_layout_rejects_unknown_minimal_dimensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="No predefined layout dimensions"):
        create_layout(clocking_scheme="ESR")


def test_start_forwards_cli_arguments_by_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mnt.nanoplacer",
            "--benchmark",
            "TOY",
            "--function",
            "xor2",
            "--clocking-scheme",
            "USE",
            "--technology",
            "QCA",
            "--minimal-layout-dimension",
            "--layout-width",
            "8",
            "--layout-height",
            "9",
            "--time-steps",
            "12",
            "--reset-model",
            "--verbose",
            "3",
            "--optimize",
        ],
    )

    with patch("mnt.nanoplacer.main.create_layout") as create:
        start()

    create.assert_called_once_with(
        benchmark="TOY",
        function="xor2",
        clocking_scheme="USE",
        technology="QCA",
        minimal_layout_dimension=True,
        layout_width=8,
        layout_height=9,
        time_steps=12,
        reset_model=True,
        verbose=3,
        optimize=True,
    )
