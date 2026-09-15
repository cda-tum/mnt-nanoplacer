"""Small local-GUI boundary and run-lifecycle checks; no training required."""

import io
import json
import re
import subprocess
import threading
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("flask")

from mnt.nanoplacer import gui


@pytest.fixture
def local_gui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    app = gui.create_app(tmp_path / "runs")
    app.config["TESTING"] = True
    process = Mock()
    process.poll.return_value = None
    launch = Mock(return_value=process)
    monkeypatch.setattr(gui.subprocess, "Popen", launch)
    monkeypatch.setattr(gui.platform, "platform", lambda: "test-platform")
    client = app.test_client()
    home = client.get("/")
    assert home.status_code == 200
    token = re.search(r'name="csrf-token" content="([^"]+)"', home.get_data(as_text=True))[1]
    client.environ_base["HTTP_X_NANOPLACER_TOKEN"] = token
    yield app, client, process, launch
    app.extensions["nanoplacer_runs"].close()


def test_gui_assets_catalog_and_local_request_protection(local_gui) -> None:
    app, client, _, launch = local_gui
    catalog = client.get("/api/catalog").get_json()
    assert "mux21" in catalog["benchmarks"]["trindade16"]
    assert catalog["dimensions"]["2DDWave"]["trindade16"]["mux21"] == [4, 3]
    assert catalog["clocking_schemes"] == ["2DDWave", "USE", "RES", "ESR"]
    assert client.get("/api/status").get_json() == {"run": None, "can_resume": False}
    for asset in ("gui.css", "gui.js"):
        assert client.get(f"/static/{asset}").status_code == 200
    assert "script-src 'self'" in client.get("/").headers["Content-Security-Policy"]
    assert client.post("/api/start", json={}, headers={"X-Nanoplacer-Token": "wrong"}).status_code == 403
    assert client.post("/api/start", json={}, headers={"X-Nanoplacer-Token": "é"}).status_code == 403
    assert client.get("/api/catalog", headers={"Host": "attacker.example"}).status_code == 400
    assert app.test_client().post("/api/start", json={}).status_code == 403
    oversized = client.post("/api/start", data="x" * 65537, content_type="application/json")
    assert oversized.status_code == 413
    assert oversized.is_json
    launch.assert_not_called()


@pytest.mark.parametrize(
    "configuration",
    [
        {"function": "../../secret"},
        {"clocking_scheme": "OTHER"},
        {"technology": []},
        {"time_steps": True},
        {"time_steps": 0},
        {"time_steps": 10000001},
        {"automatic_time_steps": "yes"},
        {"automatic_time_steps": False},
        {"seed": -1},
        {"seed": 2**32 - 1, "seed_count": 2},
        {"seed_count": 0},
        {"seed_count": 11},
        {"seed_count": True},
        {"seed_count": 2, "resume": True},
        {"stop_on_solution": "yes"},
        {"optimize": "yes"},
        {"resume": 1},
        {"resume_run_id": "../private"},
        {"resume_run_id": 123},
        {"minimal_layout_dimension": False, "layout_width": 0},
        {"minimal_layout_dimension": False, "layout_height": 129},
        {"minimal_layout_dimension": False, "layout_width": 2, "layout_height": 2},
        {"clocking_scheme": "ESR"},
        {"output_dir": "/tmp"},
    ],
)
def test_gui_rejects_invalid_runs(local_gui, configuration: dict) -> None:
    _, client, _, launch = local_gui
    response = client.post("/api/start", json=configuration)
    assert response.status_code == 400
    assert response.get_json()["error"]
    launch.assert_not_called()


@pytest.mark.parametrize(
    ("benchmark", "function", "nodes", "steps"),
    [("trindade16", "mux21", 9, 10_000), ("fontes18", "xor5Maj", 102, 102_000)],
)
def test_gui_circuit_budget_matches_training_nodes(local_gui, benchmark, function, nodes, steps) -> None:
    app, client, _, _ = local_gui
    response = client.get("/api/circuit", query_string={"benchmark": benchmark, "function": function})
    assert response.status_code == 200
    assert response.get_json() == {"placement_nodes": nodes, "recommended_timesteps": steps}
    run = client.post("/api/start", json={"benchmark": benchmark, "function": function}).get_json()["run"]
    assert run["config"]["time_steps"] == steps
    assert run["config"]["automatic_time_steps"] is True
    saved = json.loads((app.extensions["nanoplacer_runs"].current / "config.json").read_text())
    assert saved["time_steps"] == steps


@pytest.mark.parametrize("query", [{}, {"benchmark": "../secret", "function": "mux21"}])
def test_gui_circuit_details_reject_unbundled_paths(local_gui, monkeypatch, query) -> None:
    _, client, _, _ = local_gui
    count = Mock()
    monkeypatch.setattr(gui, "placement_node_count", count)
    assert client.get("/api/circuit", query_string=query).status_code == 400
    count.assert_not_called()


@pytest.mark.parametrize("flag", [{}, {"automatic_time_steps": False}, {"automatic_time_steps": True}])
def test_gui_preserves_explicit_and_recorded_budgets(local_gui, monkeypatch, flag) -> None:
    app, client, process, _ = local_gui
    run = client.post("/api/start", json={"time_steps": 1234, **flag}).get_json()["run"]
    assert run["config"]["time_steps"] == 1234
    assert run["config"]["automatic_time_steps"] is flag.get("automatic_time_steps", False)
    process.poll.return_value = 0
    directory = app.extensions["nanoplacer_runs"].current
    # Older runs lack the mode flag and may have been allowed to use an undersized grid.
    recorded = {**run["config"], "layout_width": 1, "layout_height": 1}
    recorded.pop("automatic_time_steps")
    (directory / "config.json").write_text(json.dumps(recorded))
    count = Mock(side_effect=AssertionError("History must not reparse circuits or recalculate budgets"))
    monkeypatch.setattr(gui, "placement_node_count", count)
    history = client.get(f"/api/runs/{run['id']}").get_json()["run"]
    assert history["config"]["time_steps"] == 1234
    assert "automatic_time_steps" not in history["config"]


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, OSError])
def test_gui_keeps_unexpected_errors_in_server_log(local_gui, caplog, error_type) -> None:
    app, client, _, launch = local_gui
    app.config["PROPAGATE_EXCEPTIONS"] = False
    launch.side_effect = error_type("private process details")
    response = client.post("/api/start", json={})
    assert response.status_code == 500
    assert response.get_json()["error"]
    assert "private process details" not in response.get_data(as_text=True)
    assert "private process details" in caplog.text


def test_gui_run_isolation_cancellation_and_artifacts(local_gui, monkeypatch: pytest.MonkeyPatch) -> None:
    app, client, process, launch = local_gui
    response = client.post("/api/start", json={"technology": "SiDB", "clocking_scheme": "USE"})
    assert response.status_code == 200
    run = response.get_json()["run"]
    assert run["status"] == "starting"
    assert run["config"]["clocking_scheme"] == "2DDWave"
    assert (run["config"]["layout_width"], run["config"]["layout_height"]) == (4, 3)
    runs = app.extensions["nanoplacer_runs"]
    directory = runs.current
    assert launch.call_args.kwargs["cwd"] == directory
    assert launch.call_args.args[0][-2:] == ["mnt.nanoplacer.gui_worker", str(directory)]
    assert client.post("/api/start", json={}).status_code == 409

    (directory / "layouts").mkdir()
    (directory / "layouts" / "layout.fgl").write_text("<fgl/>", encoding="utf-8")
    (directory / "preview.json").write_text(json.dumps({"width": 4, "height": 3, "cells": []}), encoding="utf-8")
    assert client.get("/api/preview").get_json()["width"] == 4
    assert client.get("/api/status").get_json()["run"]["files"] == []
    assert client.get(f"/api/files/{run['id']}/layout.fgl").status_code == 404
    process.poll.return_value = 0
    artifact = client.get("/api/status").get_json()["run"]["files"][0]
    downloaded = client.get(artifact["url"])
    assert downloaded.data == b"<fgl/>"
    assert "attachment" in downloaded.headers["Content-Disposition"]
    assert client.get(f"/api/files/{run['id']}/config.json").status_code == 404
    assert client.get("/api/files/not-a-run/layout.fgl").status_code == 404
    (directory / "layouts" / "outside.fgl").symlink_to(directory / "config.json")
    assert client.get(f"/api/files/{run['id']}/outside.fgl").status_code == 404
    (directory / "layouts" / "layout.tmp.fgl").write_text("in progress", encoding="utf-8")
    assert client.get(f"/api/files/{run['id']}/layout.tmp.fgl").status_code == 404
    linked_run = runs.directory / ("0" * 32)
    linked_run.symlink_to(directory, target_is_directory=True)
    assert client.get(f"/api/files/{linked_run.name}/layout.fgl").status_code == 404

    process.poll.return_value = None
    stop = Mock()
    monkeypatch.setattr(gui, "_finish_process", stop)
    assert client.post("/api/cancel", json={}).get_json()["run"]["status"] == "cancelling"
    assert (directory / "cancel").is_file()
    process.poll.return_value = -15
    assert client.get("/api/status").get_json()["run"]["status"] == "cancelled"
    assert client.post("/api/start", json={}).status_code == 200
    assert runs.current != directory
    assert (directory / "layouts" / "layout.fgl").exists()


def test_gui_resumes_only_matching_saved_models(local_gui) -> None:
    app, client, process, _ = local_gui
    assert client.post("/api/start", json={"resume": True}).status_code == 400
    client.post("/api/start", json={})
    directory = app.extensions["nanoplacer_runs"].current
    (directory / "models").mkdir()
    model = directory / "models" / "ppo.zip"
    model.write_bytes(b"unfinished checkpoint")
    process.poll.return_value = 0
    assert client.post("/api/start", json={"resume": True}).status_code == 400
    with zipfile.ZipFile(model, "w") as archive:
        archive.writestr("data", "checkpoint fixture")
    assert client.post("/api/start", json={"resume": True, "technology": "QCA"}).status_code == 400
    incompatible = client.post("/api/resume-check", json={"technology": "QCA"}).get_json()
    assert incompatible["can_resume"] is False
    assert incompatible["reason"]
    compatible = client.post("/api/resume-check", json={"seed": 9}).get_json()
    assert compatible["can_resume"] is True
    assert compatible["source_run_id"] == directory.name
    assert compatible["filename"] == model.name
    assert not client.post("/api/resume-check", json={"seed_count": 2, "resume": True}).get_json()["can_resume"]
    assert client.post("/api/start", json={"resume": True, "seed": 9, "time_steps": 20000}).status_code == 200
    new_directory = app.extensions["nanoplacer_runs"].current
    assert (new_directory / "models/ppo_Gate-level_trindade16_mux21_2DDWave_4x3.zip").read_bytes() == model.read_bytes()
    assert model.exists(), "resuming must preserve the source run"
    manifest = json.loads((new_directory / "manifest.json").read_text())
    assert manifest["source_checkpoint"] == {
        "run_id": directory.name,
        "filename": model.name,
        "sha256": gui._sha256(model),
    }
    assert manifest["packages"]["mnt.nanoplacer"]
    assert manifest["python"]
    assert len(manifest["circuit_sha256"]) == 64


def test_gui_continues_exact_run_and_recovers_checkpoint(local_gui) -> None:
    app, client, process, _ = local_gui
    runs = app.extensions["nanoplacer_runs"]
    first = client.post("/api/start", json={}).get_json()["run"]
    source = runs.current
    (source / "models").mkdir()
    with zipfile.ZipFile(source / "models/recovery.zip", "w") as archive:
        archive.writestr("data", "source recovery")
    # An interrupted atomic write can itself be a valid ZIP but must never be selected.
    with zipfile.ZipFile(source / "models/recovery.tmp.zip", "w") as archive:
        archive.writestr("data", "unfinished new recovery")
    process.poll.return_value = 1
    assert client.get(f"/api/runs/{first['id']}").get_json()["run"]["checkpoint_available"]
    second = client.post("/api/start", json={}).get_json()["run"]
    (runs.current / "models").mkdir()
    with zipfile.ZipFile(runs.current / "models/ppo.zip", "w") as archive:
        archive.writestr("data", "unrelated newer model")
    assert second["id"] != first["id"]
    settings = {"resume": True, "resume_run_id": first["id"], "time_steps": 1000}
    checked = client.post("/api/resume-check", json=settings).get_json()
    assert checked["source_run_id"] == first["id"]
    assert checked["checkpoint_kind"] == "recovery"
    assert checked["filename"] == "recovery.zip"
    assert checked["warnings"] == []
    continued = client.post("/api/start", json=settings).get_json()["run"]
    assert continued["manifest"]["source_checkpoint"]["run_id"] == first["id"]
    with zipfile.ZipFile(runs.current / "models/ppo_Gate-level_trindade16_mux21_2DDWave_4x3.zip") as archive:
        assert archive.read("data") == b"source recovery"
    assert (
        client.post("/api/resume-check", json={**settings, "resume_run_id": "0" * 32}).get_json()["can_resume"] is False
    )
    assert client.post("/api/start", json={**settings, "resume_run_id": "0" * 32}).status_code == 400
    assert client.post("/api/resume-check", json={**settings, "technology": "QCA"}).get_json()["can_resume"] is False


def test_gui_checkpoint_provenance_warnings(local_gui) -> None:
    app, client, process, _ = local_gui
    run = client.post("/api/start", json={}).get_json()["run"]
    source = app.extensions["nanoplacer_runs"].current
    (source / "models").mkdir()
    with zipfile.ZipFile(source / "models/ppo.zip", "w") as archive:
        archive.writestr("data", "saved agent")
    process.poll.return_value = 0
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["circuit_sha256"] = "changed"
    manifest["packages"]["mnt.pyfiction"] = "old version"
    manifest_path.write_text(json.dumps(manifest))
    checked = client.post("/api/resume-check", json={"resume_run_id": run["id"]}).get_json()
    assert checked["can_resume"]
    assert any("circuit has changed" in warning for warning in checked["warnings"])
    assert any("Package versions differ" in warning for warning in checked["warnings"])
    manifest_path.unlink()
    checked = client.post("/api/resume-check", json={"resume_run_id": run["id"]}).get_json()
    assert len(checked["warnings"]) == 2


def test_gui_history_survives_restart_without_starting_work(local_gui) -> None:
    app, client, process, launch = local_gui
    run = client.post("/api/start", json={}).get_json()["run"]
    runs = app.extensions["nanoplacer_runs"]
    directory = runs.current
    (directory / "status.json").write_text(json.dumps({"status": "completed", "verified_solution": True}))
    (directory / "preview.json").write_text(json.dumps({"width": 4, "height": 3}))
    process.poll.return_value = 0
    restored = gui.create_app(runs.directory).test_client()
    assert restored.get("/api/status").get_json()["run"] is None
    history = restored.get("/api/runs").get_json()
    assert history["active_run_id"] is None
    assert history["runs"][0]["id"] == run["id"]
    assert "log" not in history["runs"][0]
    assert restored.get(f"/api/runs/{run['id']}").get_json()["run"]["verified_solution"] is True
    assert restored.get(f"/api/preview?run_id={run['id']}").get_json()["width"] == 4
    (directory / "status.json").write_text(json.dumps({"status": "running"}))
    assert restored.get(f"/api/runs/{run['id']}").get_json()["run"]["status"] == "failed"
    (directory / "status.json").write_text(json.dumps({"status": "queued"}))
    assert restored.get(f"/api/runs/{run['id']}").get_json()["run"]["status"] == "cancelled"
    launch.assert_called_once()


@pytest.mark.parametrize(("equivalent", "verified"), [("NO", False), (None, False), ("STRONG", True), ("WEAK", True)])
def test_gui_normalizes_legacy_complete_candidates(local_gui, equivalent: str | None, verified: bool) -> None:
    app, client, process, _ = local_gui
    run = client.post("/api/start", json={}).get_json()["run"]
    directory = app.extensions["nanoplacer_runs"].current
    (directory / "status.json").write_text(
        json.dumps({"status": "completed", "solution_found": True, "equivalent": equivalent})
    )
    process.poll.return_value = 0
    state = client.get(f"/api/runs/{run['id']}").get_json()["run"]
    assert state["complete_candidate"] is True
    assert state["verified_solution"] is verified
    assert state["solution_found"] is verified


def test_gui_history_replay_and_export_reject_unsafe_files(local_gui, tmp_path: Path) -> None:
    app, client, process, _ = local_gui
    run = client.post("/api/start", json={}).get_json()["run"]
    runs = app.extensions["nanoplacer_runs"]
    directory = runs.current
    (directory / "replay").mkdir()
    (directory / "replay" / "0.json").write_text(json.dumps({"width": 2}))
    assert client.get(f"/api/preview?run_id={run['id']}&frame=0").get_json()["width"] == 2
    for frame in ("-1", "128", "0.5", "../config", "99999999999999999999"):
        assert client.get(f"/api/preview?run_id={run['id']}&frame={frame}").status_code == 400
    assert client.get(f"/api/preview?run_id={run['id']}&frame=1").status_code == 404
    for kind in ("metadata.json", "rewards.csv", "experiment.zip"):
        assert client.get(f"/api/export/{run['id']}/{kind}").status_code == 409
    process.poll.return_value = 0
    outside = tmp_path / "private.json"
    outside.write_text('{"private": true}')
    (directory / "replay" / "1.json").symlink_to(outside)
    (directory / "preview.json").symlink_to(outside)
    (directory / "worker.log").unlink()
    (directory / "worker.log").symlink_to(outside)
    assert client.get(f"/api/preview?run_id={run['id']}&frame=1").status_code == 404
    assert client.get(f"/api/preview?run_id={run['id']}").status_code == 404
    assert client.get(f"/api/runs/{run['id']}").get_json()["run"]["log"] == ""
    for name in ("1" * 32, "not-a-run"):
        linked_run = runs.directory / name
        linked_run.symlink_to(directory, target_is_directory=True)
        assert client.get(f"/api/runs/{name}").status_code == 404
        assert client.get(f"/api/export/{name}/metadata.json").status_code == 404
    invalid = runs.directory / ("2" * 32)
    invalid.mkdir()
    (invalid / "config.json").write_text('{"function":"mux21"}')
    assert len(client.get("/api/runs").get_json()["runs"]) == 1
    (directory / "config.json").unlink()
    (directory / "config.json").symlink_to(outside)
    assert client.get("/api/runs").get_json()["runs"] == []


def test_gui_exports_reproducible_bundle_and_rewards_without_log(local_gui, monkeypatch: pytest.MonkeyPatch) -> None:
    app, client, process, _ = local_gui
    run = client.post("/api/start", json={"stop_on_solution": True}).get_json()["run"]
    directory = app.extensions["nanoplacer_runs"].current
    (directory / "status.json").write_text(
        json.dumps({"status": "completed", "reward_history": [[10, -2.5], [20, 3.5]]})
    )
    (directory / "preview.json").write_text(json.dumps({"width": 4, "height": 3}))
    (directory / "layouts").mkdir()
    (directory / "layouts" / "layout.fgl").write_text("<fgl/>")
    (directory / "layouts" / "outside.fgl").symlink_to(directory / "worker.log")
    (directory / "layouts" / "unfinished.tmp.fgl").write_text("partial")
    process.poll.return_value = 0
    metadata = client.get(f"/api/export/{run['id']}/metadata.json").get_json()
    assert metadata["config"]["stop_on_solution"] is True
    assert metadata["status"]["status"] == "completed"
    rewards = client.get(f"/api/export/{run['id']}/rewards.csv")
    assert rewards.data.decode().splitlines() == ["timesteps,mean_reward", "10,-2.5", "20,3.5"]
    archive_write = zipfile.ZipFile.write

    def write_without_lock(archive, *args, **kwargs):
        assert not app.extensions["nanoplacer_runs"].lock.locked(), (
            "Exports must not delay cancellation of another run."
        )
        return archive_write(archive, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "write", write_without_lock)
    bundle = client.get(f"/api/export/{run['id']}/experiment.zip")
    with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
        assert set(archive.namelist()) == {
            "config.json",
            "manifest.json",
            "status.json",
            "preview.json",
            "rewards.csv",
            "layouts/layout.fgl",
        }
        assert archive.read("layouts/layout.fgl") == b"<fgl/>"
    bundle.close()
    assert client.get(f"/api/export/{run['id']}/worker.log").status_code == 404


def test_gui_seed_sweep_is_sequential_and_stop_cancels_pending(local_gui) -> None:
    app, client, process, launch = local_gui
    second_started = threading.Event()
    second = Mock()
    second.poll.return_value = None

    def start_process(*_args, **_kwargs):
        if launch.call_count == 1:
            return process
        second_started.set()
        return second

    launch.side_effect = start_process
    response = client.post("/api/start", json={"seed": 7, "seed_count": 3}).get_json()
    runs = app.extensions["nanoplacer_runs"]
    assert response["batch"]["total"] == 3
    assert response["batch"]["remaining"] == 3
    first = runs.current
    (first / "status.json").write_text(json.dumps({"status": "completed", "verified_solution": True}))
    process.poll.return_value = 0
    assert second_started.wait(3)
    with runs.lock:
        assert runs.current != first
        assert json.loads((runs.current / "config.json").read_text())["seed"] == 8
        assert len(runs.pending) == 1
    assert client.post("/api/start", json={}).status_code == 409
    batch = client.get("/api/runs").get_json()["batch"]
    assert batch["completed"] == 1
    assert batch["verified"] == 1
    client.post("/api/cancel", json={})
    second.poll.return_value = -15
    batch = client.get("/api/runs").get_json()["batch"]
    assert batch["completed"] == 3
    assert batch["cancelled"] == 2
    assert not runs.pending
    assert launch.call_count == 2


def test_gui_read_requests_do_not_advance_sweep(local_gui, monkeypatch: pytest.MonkeyPatch) -> None:
    app, client, process, launch = local_gui
    monkeypatch.setattr(gui.threading, "Thread", Mock())
    client.post("/api/start", json={"seed": 2**32 - 2, "seed_count": 2})
    process.poll.return_value = 0
    for endpoint in ("/api/status", "/api/runs", "/api/preview"):
        client.get(endpoint)
    assert len(app.extensions["nanoplacer_runs"].pending) == 1
    assert sorted(run["config"]["seed"] for run in client.get("/api/runs").get_json()["runs"]) == [2**32 - 2, 2**32 - 1]
    launch.assert_called_once()


def test_gui_failed_sweep_setup_clears_pending_and_allows_retry(local_gui, monkeypatch: pytest.MonkeyPatch) -> None:
    app, client, _, launch = local_gui
    write_json = gui._write_json
    manifests = 0

    def fail_second_manifest(path, data):
        nonlocal manifests
        if path.name == "manifest.json":
            manifests += 1
            if manifests == 2:
                msg = "unwritable manifest"
                raise OSError(msg)
        write_json(path, data)

    monkeypatch.setattr(gui, "_write_json", fail_second_manifest)
    assert client.post("/api/start", json={"seed_count": 3}).status_code == 500
    assert not app.extensions["nanoplacer_runs"].pending
    launch.assert_not_called()
    assert {run["status"] for run in client.get("/api/runs").get_json()["runs"]} == {"cancelled", "failed"}
    assert client.post("/api/start", json={}).status_code == 200
    launch.assert_called_once()


def test_gui_reports_worker_failure_and_stops_blocked_native_work(local_gui) -> None:
    _, client, process, _ = local_gui
    client.post("/api/start", json={})
    process.poll.return_value = 1
    run = client.get("/api/status").get_json()["run"]
    assert run["status"] == "failed"
    assert run["error"]
    process.wait.side_effect = [subprocess.TimeoutExpired("worker", 5), subprocess.TimeoutExpired("worker", 2), 0]
    gui._finish_process(process)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()


def test_gui_shutdown_waits_for_start_and_rejects_queued_runs(local_gui, monkeypatch: pytest.MonkeyPatch) -> None:
    app, client, process, launch = local_gui
    runs = app.extensions["nanoplacer_runs"]
    lock = runs.lock
    closing = threading.Event()
    finish = Mock()
    monkeypatch.setattr(gui, "_finish_process", finish)

    @contextmanager
    def closing_lock():
        closing.set()
        with lock:
            yield

    monkeypatch.setattr(runs, "lock", closing_lock())
    shutdown = threading.Thread(target=runs.close)
    try:
        # A Start request owns the lock before it launches and records its child.
        with lock:
            shutdown.start()
            assert closing.wait(2)
            runs.start(gui._configuration({}, {"trindade16": ["mux21"]}))
    finally:
        shutdown.join(2)
    assert not shutdown.is_alive()
    finish.assert_called_once_with(process)
    assert (runs.current / "cancel").is_file()

    monkeypatch.setattr(runs, "lock", lock)
    process.poll.return_value = 0
    response = client.post("/api/start", json={})
    assert response.status_code == 409
    assert "shutting down" in response.get_json()["error"]
    launch.assert_called_once()


def test_gui_binds_before_opening_browser_and_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    app, runs, server = Mock(), Mock(), Mock()
    app.extensions = {"nanoplacer_runs": runs}
    monkeypatch.setattr(gui, "create_app", Mock(return_value=app))
    monkeypatch.setattr(gui.sys, "argv", ["mnt.nanoplacer.gui", "--port", "5057"])
    bound = Mock()
    bound.__enter__ = Mock(return_value=server)
    bound.__exit__ = Mock(return_value=False)
    bind = Mock(return_value=bound)
    monkeypatch.setattr(gui, "make_server", bind)

    def open_browser(url: str) -> None:
        bind.assert_called_once_with("127.0.0.1", 5057, app, threaded=True)
        assert url == "http://127.0.0.1:5057"

    monkeypatch.setattr(gui.webbrowser, "open", open_browser)
    server.serve_forever.side_effect = KeyboardInterrupt
    gui.main()
    runs.close.assert_called_once()
