"""Optional, local-only browser interface for isolated NanoPlaceR runs."""

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import uuid
import webbrowser
import zipfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

try:
    from flask import Flask, jsonify, render_template, request, send_file
    from werkzeug.exceptions import BadRequest, Conflict, HTTPException
    from werkzeug.serving import make_server
except ImportError as exc:
    msg = 'Install the browser interface with: python -m pip install "mnt.nanoplacer[gui]"'
    raise SystemExit(msg) from exc

from mnt.nanoplacer.placement_envs.utils import layout_dimensions

CLOCKS = ("2DDWave", "USE", "RES", "ESR")
TECHNOLOGIES = ("Gate-level", "QCA", "SiDB")
MODEL_KEYS = ("benchmark", "function", "clocking_scheme", "technology", "layout_width", "layout_height")
MAX_DIMENSION = 128
MAX_TIMESTEPS = 10_000_000


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _files(run_dir: Path) -> list[Path]:
    if run_dir.is_symlink():
        return []
    return sorted(
        path
        for folder in ("layouts", "models")
        if not (run_dir / folder).is_symlink()
        for path in (run_dir / folder).glob("*")
        if path.is_file()
        and not path.is_symlink()
        and ".tmp." not in path.name
        and path.suffix in {".fgl", ".svg", ".dot", ".zip"}
    )


class Runs:
    """One local training process, with persistent artifacts in separate run folders."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.closed = False
        self.process: subprocess.Popen | None = None
        self.current: Path | None = None

    def saved_model(self, config: dict[str, Any] | None = None) -> Path | None:
        models = []
        for path in self.directory.glob("*/models/*.zip"):
            if (
                path.is_symlink()
                or path.parent.is_symlink()
                or path.parents[1].is_symlink()
                or not path.is_file()
                or not _valid_id(path.parents[1].name)
                or (path.parents[1] == self.current and self.process is not None and self.process.poll() is None)
                or not zipfile.is_zipfile(path)
            ):
                continue
            previous = _read_json(path.parents[1] / "config.json")
            if config is None or all(previous.get(key) == config[key] for key in MODEL_KEYS):
                models.append(path)
        return max(models, key=lambda path: path.stat().st_mtime, default=None)

    def snapshot(self) -> dict[str, Any]:
        if self.current is None:
            return {"run": None, "can_resume": self.saved_model() is not None}
        run_dir = self.current
        config = _read_json(run_dir / "config.json")
        state = {
            "status": "starting",
            "timesteps": 0,
            "total_timesteps": config["time_steps"],
            "episodes": 0,
            "elapsed": 0,
            "best_placed": 0,
            "total_nodes": 0,
            "solution_found": False,
            "equivalent": None,
            "preview_revision": 0,
            "error": None,
            **_read_json(run_dir / "status.json"),
        }
        alive = self.process is not None and self.process.poll() is None
        cancelling = (run_dir / "cancel").exists()
        if alive and cancelling:
            state["status"] = "cancelling"
        elif not alive and state["status"] not in {"completed", "cancelled", "failed"}:
            state["status"] = "cancelled" if cancelling else "failed"
            if not cancelling:
                state["error"] = "The training process stopped unexpectedly. Check the run log."
        # A worker may publish its final state just before exiting; keep starts locked until it exits.
        if alive and state["status"] in {"completed", "cancelled", "failed"}:
            state["status"] = "cancelling" if cancelling else "running"
        with (run_dir / "worker.log").open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 8000))
            log = stream.read().decode("utf-8", errors="replace")
        state.update(
            id=run_dir.name,
            config=config,
            files=[
                {"name": path.name, "size": path.stat().st_size, "url": f"/api/files/{run_dir.name}/{path.name}"}
                for path in _files(run_dir)
                if path.stat().st_size > 0 and not alive
            ],
            log=log,
        )
        return {"run": state, "can_resume": self.saved_model() is not None}

    def start(self, config: dict[str, Any]) -> None:
        if self.closed:
            msg = "The GUI is shutting down. Restart it before starting another run."
            raise Conflict(msg)
        if self.process is not None and self.process.poll() is None:
            msg = "A run is already active. Cancel it or wait for it to finish."
            raise Conflict(msg)
        model = self.saved_model(config) if config["resume"] else None
        if config["resume"] and model is None:
            msg = "No saved model matches this benchmark, technology, clocking scheme and grid size. Start a fresh run."
            raise BadRequest(msg)
        run_dir = self.directory / uuid.uuid4().hex
        run_dir.mkdir()
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        if model is not None:
            (run_dir / "models").mkdir()
            shutil.copy2(model, run_dir / "models" / model.name)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("MPLCONFIGDIR", str(self.directory / ".matplotlib"))
        # Keep one experiment from consuming every CPU core on the user's workstation.
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        with (run_dir / "worker.log").open("wb") as log:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "mnt.nanoplacer.gui_worker", str(run_dir)],
                cwd=run_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        self.current = run_dir

    def cancel(self) -> None:
        if self.current is None or self.process is None or self.process.poll() is not None:
            return
        marker = self.current / "cancel"
        if marker.exists():
            return
        marker.touch()
        # Native routing/optimization can block callbacks. Only this run's child is stopped.
        thread = threading.Thread(target=_finish_process, args=(self.process,), daemon=True)
        thread.start()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            if self.process is not None and self.process.poll() is None:
                if self.current is not None:
                    (self.current / "cancel").touch()
                _finish_process(self.process)


def _finish_process(process: subprocess.Popen) -> None:
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _valid_id(value: str) -> bool:
    return len(value) == 32 and all(character in "0123456789abcdef" for character in value)


def _configuration(data: Any, benchmarks: dict[str, list[str]]) -> dict[str, Any]:
    if not isinstance(data, dict):
        msg = "Supply a JSON configuration."
        raise BadRequest(msg)
    config = {
        "benchmark": "trindade16",
        "function": "mux21",
        "clocking_scheme": "2DDWave",
        "technology": "Gate-level",
        "layout_width": 4,
        "layout_height": 3,
        "time_steps": 10000,
        "seed": 42,
        "optimize": True,
        "minimal_layout_dimension": True,
        "resume": False,
    }
    if data.keys() - config.keys():
        msg = "Unknown configuration parameter."
        raise BadRequest(msg)
    config.update(data)
    benchmark, function = config["benchmark"], config["function"]
    if not isinstance(benchmark, str) or not isinstance(function, str) or function not in benchmarks.get(benchmark, []):
        msg = "Select a bundled benchmark and circuit."
        raise BadRequest(msg)
    if config["clocking_scheme"] not in CLOCKS or config["technology"] not in TECHNOLOGIES:
        msg = "Select a supported technology and clocking scheme."
        raise BadRequest(msg)
    for key in ("optimize", "minimal_layout_dimension", "resume"):
        if type(config[key]) is not bool:
            msg = f"{key} must be true or false."
            raise BadRequest(msg)
    if config["technology"] == "SiDB":
        config["clocking_scheme"] = "2DDWave"
    if config["clocking_scheme"] != "2DDWave":
        config["optimize"] = False
    if config["minimal_layout_dimension"]:
        dimensions = layout_dimensions.get(config["clocking_scheme"], {}).get(benchmark, {}).get(function)
        if dimensions is None:
            msg = "No published dimensions exist for this circuit and clocking scheme. Choose a custom grid."
            raise BadRequest(msg)
        config["layout_width"], config["layout_height"] = dimensions
    for key, lower, upper in (
        ("layout_width", 1, MAX_DIMENSION),
        ("layout_height", 1, MAX_DIMENSION),
        ("time_steps", 1, MAX_TIMESTEPS),
        ("seed", 0, 2**32 - 1),
    ):
        if type(config[key]) is not int or not lower <= config[key] <= upper:
            msg = f"{key} must be an integer between {lower} and {upper}."
            raise BadRequest(msg)
    return config


def create_app(runs_dir: Path | str = "nanoplacer-runs") -> Flask:
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=64 * 1024, TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"])
    token = secrets.token_urlsafe(32)
    runs = Runs(Path(runs_dir).expanduser())
    app.extensions["nanoplacer_runs"] = runs
    benchmark_dir = Path(__file__).parent / "benchmarks"
    benchmarks = {
        folder.name: sorted(path.stem for path in folder.glob("*.v"))
        for folder in sorted(benchmark_dir.iterdir())
        if folder.is_dir()
    }

    @app.before_request
    def protect_local_requests():
        if request.method == "POST" and not secrets.compare_digest(
            request.headers.get("X-Nanoplacer-Token", "").encode(), token.encode()
        ):
            return jsonify(error="Reload this page before starting or cancelling a run."), 403
        return None

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if not request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.get("/")
    def index():
        return render_template("gui.html", csrf_token=token)

    @app.get("/api/catalog")
    def catalog():
        return jsonify(
            benchmarks=benchmarks,
            dimensions=layout_dimensions,
            clocking_schemes=CLOCKS,
            technologies=TECHNOLOGIES,
            limits={"max_dimension": MAX_DIMENSION, "max_tiles": MAX_DIMENSION**2, "max_timesteps": MAX_TIMESTEPS},
            version=version("mnt.nanoplacer"),
        )

    @app.get("/api/status")
    def status():
        with runs.lock:
            return jsonify(runs.snapshot())

    @app.post("/api/start")
    def start_run():
        try:
            config = _configuration(request.get_json(silent=True), benchmarks)
            with runs.lock:
                runs.start(config)
                return jsonify(runs.snapshot())
        except OSError:
            app.logger.exception("Could not start a NanoPlaceR process")
            return jsonify(error="Could not start training. Check that the run directory is writable."), 500

    @app.post("/api/cancel")
    def cancel_run():
        with runs.lock:
            runs.cancel()
            return jsonify(runs.snapshot())

    @app.get("/api/preview")
    def preview():
        with runs.lock:
            data = _read_json(runs.current / "preview.json") if runs.current else {}
            return jsonify(data) if data else (jsonify(error="No placement preview yet."), 404)

    @app.get("/api/files/<run_id>/<filename>")
    def download(run_id: str, filename: str):
        with runs.lock:
            run_dir = runs.directory / run_id
            active = run_dir == runs.current and runs.process is not None and runs.process.poll() is None
            if _valid_id(run_id) and not active:
                for path in _files(run_dir):
                    if path.name == filename:
                        return send_file(path, as_attachment=True, download_name=path.name)
        return jsonify(error="Run artifact not found."), 404

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the local NanoPlaceR browser interface.")
    parser.add_argument("--port", type=int, default=5056, help="Local port (default: 5056).")
    parser.add_argument("--runs-dir", type=Path, default=Path("nanoplacer-runs"), help="Directory for isolated runs.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    app = create_app(args.runs_dir)
    url = f"http://127.0.0.1:{args.port}"
    # Bind first: a browser can connect immediately, and port conflicts never open another app.
    with make_server("127.0.0.1", args.port, app, threaded=True) as server:
        print(f"NanoPlaceR GUI: {url}", flush=True)
        try:
            if not args.no_browser:
                webbrowser.open(url)
            server.serve_forever()
        except KeyboardInterrupt:
            # Ctrl+C exits quietly; the finally block stops any active worker.
            pass
        finally:
            app.extensions["nanoplacer_runs"].close()


if __name__ == "__main__":
    main()
