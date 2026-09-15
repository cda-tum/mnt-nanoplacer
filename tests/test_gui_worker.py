import json
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mnt import pyfiction
from mnt.nanoplacer import gui_worker, main
from mnt.nanoplacer.placement_envs.nano_placement_env import NanoPlacementEnv

CONFIG = {
    "benchmark": "trindade16",
    "function": "mux21",
    "clocking_scheme": "2DDWave",
    "technology": "Gate-level",
    "layout_width": 3,
    "layout_height": 4,
    "time_steps": 8,
    "seed": 7,
    "optimize": True,
    "resume": False,
}


@pytest.mark.parametrize("clocking_scheme", ["2DDWave", "USE", "RES", "ESR"])
def test_preview_preserves_native_layers_edges_and_clocks(clocking_scheme: str) -> None:
    layout = pyfiction.cartesian_gate_layout((2, 2, 1), clocking_scheme)
    first = layout.create_pi("a", (0, 0, 0))
    upper = layout.create_buf(first, (1, 0, 1))  # No ground-layer gate below this wire.
    layout.create_po(upper, "y", (2, 0, 0))
    second = layout.create_pi("b", (0, 1, 0))
    layout.create_buf(second, (1, 1, 0))
    layout.create_buf(upper, (1, 1, 1))  # Both layers of a crossing must remain distinct.
    preview = gui_worker.layout_preview(SimpleNamespace(layout=layout, clocking_scheme=clocking_scheme))
    assert len(preview["cells"]) == 6
    assert {(cell["x"], cell["y"], cell["z"]) for cell in preview["cells"]} == {
        (0, 0, 0),
        (1, 0, 1),
        (2, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 1, 1),
    }
    expected_edges = {(0, 0, 0, 1, 0, 1), (1, 0, 1, 2, 0, 0), (0, 1, 0, 1, 1, 0), (1, 0, 1, 1, 1, 1)}
    if clocking_scheme == "USE":
        # USE clocks reject this connection; previews must not invent the missing native edge.
        expected_edges.remove((0, 1, 0, 1, 1, 0))
    assert {tuple(edge["source"]) + tuple(edge["target"]) for edge in preview["edges"]} == expected_edges
    assert preview["phases"] == [[layout.get_clock_number((x, y, 0)) + 1 for x in range(3)] for y in range(3)]
    for cell in preview["cells"]:
        assert cell["phase"] == layout.get_clock_number((cell["x"], cell["y"], cell["z"])) + 1


@pytest.mark.parametrize("complete", [False, True])
def test_worker_reports_run_relative_progress_and_keeps_best_after_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    config = CONFIG | {"resume": True, "technology": "QCA"}
    (tmp_path / "config.json").write_text(json.dumps(config))

    def place_known_actions(**kwargs) -> None:
        assert kwargs["reset_model"] is False
        assert kwargs["minimal_layout_dimension"] is False
        assert kwargs["verbose"] == 0
        assert kwargs["seed"] == 7
        env = NanoPlacementEnv(technology="QCA", verbose=0, on_best=kwargs["on_best"])
        wrapped = DummyVecEnv([lambda: env])
        wrapped.reset()
        model = SimpleNamespace(num_timesteps=120, get_env=lambda: wrapped)
        callback = kwargs["callback"]
        callback.init_callback(model)
        callback.on_training_start({}, {})
        actions = (3, 6, 0, 1, 7, 2, 5, 8, 11) if complete else (3, 3)
        for action in actions:
            _, _, done, infos = wrapped.step([action])
            model.num_timesteps += 1
            callback.update_locals({"dones": done, "infos": infos})
            assert callback.on_step() is True
        assert env.current_node == 0
        assert json.loads((tmp_path / "status.json").read_text())["status"] == "running"
        wrapped.close()

    monkeypatch.setattr(gui_worker, "create_layout", place_known_actions)
    assert gui_worker.run_worker(tmp_path) == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["timesteps"] == (9 if complete else 2)
    assert status["episodes"] == 1
    assert status["best_placed"] == (9 if complete else 1)
    assert status["total_nodes"] == 9
    assert status["solution_found"] is complete
    assert status["verified_solution"] is complete
    assert status["complete_candidate"] is complete
    assert status["successful_episodes"] == int(complete)
    assert status["complete_episodes"] == int(complete)
    assert status["routing_failures"] == 0
    assert status["equivalent"] == ("STRONG" if complete else None)
    assert status["preview_revision"] == (9 if complete else 1)
    assert status["mean_reward"] is None  # Unmonitored test environments have no episode-return data.
    assert status["reward_window"] == 0
    assert status["reward_history"] == []
    assert status["replay_count"] == (9 if complete else 1)
    assert status["stop_reason"] == "budget"
    if complete:
        assert status["first_solution_time"] >= 0
        assert status["best_metrics"]["area"] > 0
    preview = json.loads((tmp_path / "preview.json").read_text())
    assert len(preview["cells"]) >= status["best_placed"]
    assert (tmp_path / "layouts/layout.fgl").exists() is complete
    assert (tmp_path / "layouts/mux21_2DDWave_qca.svg").exists() is complete
    assert not list(tmp_path.glob("*.tmp"))


def test_replay_is_bounded_immutable_and_only_verified_candidates_are_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gui_worker, "MAX_REPLAY_FRAMES", 4)
    layout = pyfiction.cartesian_gate_layout((1, 0, 0), "2DDWave")
    source = layout.create_pi("a", (0, 0, 0))
    layout.create_po(source, "y", (1, 0, 0))
    env = SimpleNamespace(
        layout=layout,
        clocking_scheme="2DDWave",
        max_placed_nodes=2,
        actions=[0, 1],
        verified_solution=False,
        equivalent="NO",
        best_metrics={"width": 2, "height": 1, "area": 2, "wires": 0, "crossings": 0},
        first_solution_time=None,
        reported_dimensions=None,
        target_reproduced=False,
        target_equivalent=None,
    )
    callback = gui_worker._Progress(tmp_path)
    callback.model = SimpleNamespace(_n_updates=20)
    callback.baseline = 400
    callback.epoch_baseline = 20
    callback.num_timesteps = 400
    callback.best(env)
    first = (tmp_path / "replay/0.json").read_bytes()
    assert callback.status["complete_candidate"] is True
    assert callback.status["solution_found"] is False
    assert not (tmp_path / "layouts").exists()
    env.verified_solution = True
    env.equivalent = "STRONG"
    env.first_solution_time = 1.5
    for step in range(2, 7):
        callback.model._n_updates += 10
        callback.num_timesteps = 400 + step - 1
        callback.best(env)
    assert (tmp_path / "replay/0.json").read_bytes() == first
    assert json.loads(first)["metadata"]["timestep"] == 1
    assert len(list((tmp_path / "replay").glob("*.json"))) == 4
    assert callback.status["replay_count"] == 4
    assert callback.status["replay_truncated"] is True
    assert callback.status["solution_found"] is True
    assert callback.status["first_solution_time"] == 1.5
    assert callback.status["first_solution_timestep"] == 2
    assert callback.status["first_solution_ppo_epochs"] == 10
    assert callback.status["first_solution_ppo_epochs_total"] == 30
    assert callback.status["ppo_epochs"] == 50
    assert callback.status["ppo_epochs_total"] == 70
    assert callback.status["best_metrics"] == env.best_metrics
    assert json.loads((tmp_path / "preview.json").read_text())["metadata"]["timestep"] == 6
    assert (tmp_path / "layouts/layout.fgl").exists()


def test_cancel_stops_real_training_and_saves_a_loadable_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    (tmp_path / "cancel").touch()

    def small_model(*args, **kwargs):
        return MaskablePPO(*args, **(kwargs | {"n_steps": 8, "batch_size": 8}))

    monkeypatch.setattr(main, "MaskablePPO", small_model)
    assert gui_worker.run_worker(tmp_path) == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "cancelled"
    assert status["timesteps"] == 1
    assert status["solution_found"] is False
    assert status["mean_reward"] is None  # A partial episode is not a completed return.
    checkpoint = next((tmp_path / "models").glob("*.zip"))
    assert MaskablePPO.load(checkpoint).num_timesteps == 1
    assert status["checkpoint"]["kind"] == "final"
    assert status["checkpoint"]["timestep"] == status["checkpoint"]["total_timesteps"] == 1
    assert status["ppo_epochs"] == status["ppo_epochs_total"] == 0


def test_worker_failure_is_recorded_without_exposing_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))

    def fail(**_kwargs) -> None:
        msg = "private exception details"
        raise RuntimeError(msg)

    monkeypatch.setattr(gui_worker, "create_layout", fail)
    assert gui_worker.run_worker(tmp_path) == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["error"] == "Training failed. See worker.log for details."
    assert "private exception details" in capsys.readouterr().err


def test_periodic_recovery_and_final_training_epochs_are_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(CONFIG | {"time_steps": 16}))

    def small_model(*args, **kwargs):
        return MaskablePPO(*args, **(kwargs | {"n_steps": 8, "batch_size": 8, "n_epochs": 2}))

    monkeypatch.setattr(main, "MaskablePPO", small_model)
    monkeypatch.setattr(gui_worker, "CHECKPOINT_INTERVAL", 0)
    assert gui_worker.run_worker(tmp_path) == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["ppo_epochs"] == status["ppo_epochs_total"] == 4
    assert status["checkpoint"]["kind"] == "final"
    assert status["checkpoint"]["ppo_epochs_total"] == 4
    final = MaskablePPO.load(tmp_path / "models" / status["checkpoint"]["filename"])
    recovery = MaskablePPO.load(tmp_path / "models/recovery.zip")
    assert final.num_timesteps == recovery.num_timesteps == 16
    assert final._n_updates == 4
    assert recovery._n_updates == 2  # Last rollout collected, before its policy update.
    assert status["placement_history"][0] == [0, 0, 9]
    assert status["placement_history"][-1] == [16, status["best_placed"], 9]
    assert not list((tmp_path / "models").glob("*.tmp"))

    (tmp_path / "config.json").write_text(json.dumps(CONFIG | {"time_steps": 16, "resume": True}))
    monkeypatch.setattr(main, "MaskablePPO", MaskablePPO)
    assert gui_worker.run_worker(tmp_path) == 0
    resumed = json.loads((tmp_path / "status.json").read_text())
    assert resumed["timesteps"] == resumed["checkpoint"]["timestep"] == 16
    assert resumed["checkpoint"]["total_timesteps"] == 32
    assert resumed["ppo_epochs"] == 4
    assert resumed["ppo_epochs_total"] == resumed["checkpoint"]["ppo_epochs_total"] == 8


def test_failed_recovery_preserves_previous_checkpoint_and_metadata(tmp_path: Path, capsys) -> None:
    callback = gui_worker._Progress(tmp_path)
    callback.baseline = 80
    callback.epoch_baseline = 20
    callback.num_timesteps = 100
    callback.model = SimpleNamespace(_n_updates=30, save=lambda archive: archive.write(b"saved"))
    callback.last_checkpoint = 0
    callback.save_recovery()
    checkpoint = callback.status["checkpoint"].copy()
    assert checkpoint["timestep"] == 20
    assert checkpoint["total_timesteps"] == 100
    assert checkpoint["ppo_epochs_total"] == 30

    def failed_save(archive) -> None:
        archive.write(b"incomplete")
        msg = "private disk failure"
        raise OSError(msg)

    callback.model.save = failed_save
    callback.num_timesteps += 1
    callback.last_checkpoint = 0
    callback.save_recovery()
    assert (tmp_path / "models/recovery.zip").read_bytes() == b"saved"
    assert callback.status["checkpoint"] == checkpoint
    assert "private" not in callback.status["checkpoint_error"]
    assert "private disk failure" in capsys.readouterr().err


def test_placement_history_is_bounded_and_terminal_target_proof_is_not_lost(tmp_path: Path) -> None:
    callback = gui_worker._Progress(tmp_path)
    callback.status["total_nodes"] = 1000
    for step in range(1001):
        callback.status["best_placed"] = step
        callback.placement_sample(step)
        assert len(callback.status["placement_history"]) <= 256
    history = callback.status["placement_history"]
    assert history[0] == [0, 0, 1000]
    assert history[-1] == [1000, 1000, 1000]
    assert all(left[0] < right[0] for left, right in pairwise(history))
    callback.update_locals({"dones": [True], "infos": [{"target_reproduced": True, "target_equivalent": "STRONG"}]})
    callback._on_step()
    assert callback.status["target_reproduced"] is True
    assert callback.status["target_equivalent"] == "STRONG"


def test_reward_uses_real_monitor_episode_return_and_resume_relative_axis(tmp_path: Path) -> None:
    env = NanoPlacementEnv(technology="Gate-level", verbose=0)
    wrapped = DummyVecEnv([lambda: Monitor(env)])
    wrapped.reset()
    model = SimpleNamespace(num_timesteps=400, get_env=lambda: wrapped)
    callback = gui_worker._Progress(tmp_path)
    callback.init_callback(model)
    callback.on_training_start({}, {})
    for step in (1, 2):
        _, _, dones, infos = wrapped.step([0])  # Place one PI, then end at its occupied tile.
        model.num_timesteps += 1
        callback.update_locals({"dones": dones, "infos": infos})
        assert callback.on_step()
        if step == 1:
            assert callback.status["mean_reward"] is None
            assert callback.status["reward_history"] == []
    assert callback.status["mean_reward"] == infos[0]["episode"]["r"] == 1.0
    assert callback.status["reward_window"] == 1
    assert callback.status["reward_history"] == [[2, 1.0]]
    callback.publish()  # Completion uses this same final flush, independent of the throttle.
    assert json.loads((tmp_path / "status.json").read_text())["reward_history"] == [[2, 1.0]]
    wrapped.close()


@pytest.mark.parametrize("reward", [0.0, -7.5, 4.0])
def test_reward_rolling_window_bounded_history_and_cancel_flush(tmp_path: Path, reward: float) -> None:
    callback = gui_worker._Progress(tmp_path)
    callback.baseline = 120
    callback.last_write = float("inf")  # Keep these fast fake steps below the publish throttle.
    for step in range(1, 1026):
        callback.num_timesteps = 120 + step
        callback.update_locals({"dones": [True], "infos": [{"episode": {"r": reward}}]})
        assert callback._on_step()
        history = callback.status["reward_history"]
        assert len(history) <= 256
        assert history[0] == [1, reward]
        assert history[-1] == [step, reward]
    assert callback.status["reward_window"] == 100
    assert len(callback.episode_rewards) == 100
    callback.num_timesteps += 1
    callback.update_locals({"dones": [True], "infos": [{"episode": {"r": reward + 100}}]})
    (tmp_path / "cancel").touch()
    assert callback._on_step() is False
    saved = json.loads((tmp_path / "status.json").read_text())
    assert saved["mean_reward"] == reward + 1
    assert saved["reward_window"] == 100
    assert saved["reward_history"][-1] == [1026, reward + 1]
    assert [point[0] for point in saved["reward_history"]] == sorted({point[0] for point in saved["reward_history"]})


@pytest.mark.parametrize(("budget", "steps"), [(10000, 10000), (10000, 50000), (0, 1000)])
def test_reward_history_samples_uniform_timestep_buckets(tmp_path: Path, budget: int, steps: int) -> None:
    callback = gui_worker._Progress(tmp_path)
    callback.status["total_timesteps"] = budget
    callback.model = SimpleNamespace(get_env=lambda: SimpleNamespace(get_attr=lambda _name: [[]]))
    callback._on_training_start()
    assert callback.reward_stride == max(1, (budget + 239) // 240)
    callback.last_write = float("inf")
    interval = 6 if budget else 1
    measured = {}
    for step in range(interval, steps + 1, interval):
        callback.num_timesteps = step
        callback.update_locals({"dones": [True], "infos": [{"episode": {"r": float(step)}}]})
        assert callback._on_step()
        measured[step] = callback.status["mean_reward"]
    history = callback.status["reward_history"]
    assert 100 < len(history) <= 256
    assert history[0] == [interval, float(interval)]
    assert history[-1] == [step, measured[step]]
    assert all(mean == measured[timestep] for timestep, mean in history)
    gaps = [right[0] - left[0] for left, right in pairwise(history)]
    assert min(gaps) > 0
    assert max(gaps) <= 2 * callback.reward_stride + interval
