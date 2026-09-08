"""Small local-GUI boundary and run-lifecycle checks; no training required."""

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
    client = app.test_client()
    home = client.get("/")
    assert home.status_code == 200
    token = re.search(r'name="csrf-token" content="([^"]+)"', home.get_data(as_text=True))[1]
    client.environ_base["HTTP_X_NANOPLACER_TOKEN"] = token
    return app, client, process, launch


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
        {"seed": -1},
        {"optimize": "yes"},
        {"resume": 1},
        {"minimal_layout_dimension": False, "layout_width": 0},
        {"minimal_layout_dimension": False, "layout_height": 129},
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
    assert client.post("/api/start", json={"resume": True, "seed": 9, "time_steps": 20000}).status_code == 200
    new_directory = app.extensions["nanoplacer_runs"].current
    assert (new_directory / "models" / model.name).read_bytes() == model.read_bytes()
    assert model.exists(), "resuming must preserve the source run"


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
