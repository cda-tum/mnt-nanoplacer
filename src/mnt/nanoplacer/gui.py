"""Optional, local-only browser interface for isolated NanoPlaceR runs."""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import webbrowser
import zipfile
from contextlib import suppress
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

try:
    from flask import Flask, jsonify, render_template, request, send_file
    from werkzeug.exceptions import BadRequest, Conflict, HTTPException, NotFound
    from werkzeug.serving import make_server
except ImportError as exc:
    msg = 'Install the browser interface with: python -m pip install "mnt.nanoplacer[gui]"'
    raise SystemExit(msg) from exc

from mnt.nanoplacer.placement_envs.utils import layout_dimensions
from mnt.nanoplacer.placement_envs.utils.placement_utils import (
    MAX_TIMESTEPS,
    placement_node_count,
    recommended_timesteps,
)

CLOCKS = ("2DDWave", "USE", "RES", "ESR")
TECHNOLOGIES = ("Gate-level", "QCA", "SiDB")
MODEL_KEYS = ("benchmark", "function", "clocking_scheme", "technology", "layout_width", "layout_height")
MAX_DIMENSION = 128
FINISHED = {"completed", "cancelled", "failed"}


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _package_versions() -> dict[str, str | None]:
    packages = {}
    for package in (
        "mnt.nanoplacer",
        "mnt.pyfiction",
        "sb3-contrib",
        "stable-baselines3",
        "gymnasium",
        "numpy",
        "torch",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return packages


def _checkpoint_warnings(model: Path, config: dict[str, Any]) -> list[str]:
    manifest = _read_json(model.parents[1] / "manifest.json")
    warnings = []
    circuit = Path(__file__).parent / "benchmarks" / config["benchmark"] / f"{config['function']}.v"
    recorded_hash = manifest.get("circuit_sha256")
    if recorded_hash is None:
        warnings.append("This older checkpoint has no circuit hash; its circuit identity cannot be verified.")
    elif recorded_hash != _sha256(circuit):
        warnings.append("The bundled circuit has changed since this checkpoint. Starting fresh is recommended.")
    previous = manifest.get("packages")
    if not isinstance(previous, dict):
        warnings.append("This older checkpoint has no package versions; software compatibility is unknown.")
    elif any(previous.get(name) != value for name, value in _package_versions().items()):
        warnings.append("Package versions differ from this checkpoint. Loading or reproducibility may be affected.")
    if manifest.get("python") and manifest["python"] != platform.python_version():
        warnings.append("The Python version differs from this checkpoint.")
    return warnings


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

    def __init__(self, directory: Path, benchmarks: dict[str, list[str]]) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.closed = False
        self.process: subprocess.Popen | None = None
        self.current: Path | None = None
        self.benchmarks = benchmarks
        self.pending: list[Path] = []
        self.batch_id: str | None = None
        self.batch_stop = threading.Event()

    def run_directory(self, run_id: str) -> Path:
        path = self.directory / run_id
        if not _valid_id(run_id) or path.is_symlink() or not path.is_dir():
            msg = "Run not found."
            raise NotFound(msg)
        try:
            data = _read_json(path / "config.json")
            if not {*MODEL_KEYS, "time_steps", "seed"} <= data.keys():
                msg = "Run configuration not found."
                raise NotFound(msg)
            if type(data.get("minimal_layout_dimension", True)) is not bool:
                msg = "Run configuration is invalid."
                raise BadRequest(msg)
            # Validate recorded dimensions, not today's published minimum for this circuit.
            _configuration({**data, "minimal_layout_dimension": False}, self.benchmarks)
        except BadRequest as exc:
            msg = "Run configuration is invalid."
            raise NotFound(msg) from exc
        return path

    def active(self, run_dir: Path) -> bool:
        return run_dir == self.current and self.process is not None and self.process.poll() is None

    def saved_model(self, config: dict[str, Any] | None = None) -> Path | None:
        source = config.get("resume_run_id") if config else None
        if source is not None:
            try:
                folders = [self.run_directory(source)]
            except NotFound:
                return None
        else:
            folders = self.directory.iterdir()
        models = []
        for path in (model for folder in folders for model in (folder / "models").glob("*.zip")):
            if (
                path.is_symlink()
                or path.parent.is_symlink()
                or path.parents[1].is_symlink()
                or not path.is_file()
                or not _valid_id(path.parents[1].name)
                or ".tmp." in path.name
                or (path.parents[1] == self.current and self.process is not None and self.process.poll() is None)
                or not zipfile.is_zipfile(path)
            ):
                continue
            try:
                self.run_directory(path.parents[1].name)
            except NotFound:
                continue
            previous = _read_json(path.parents[1] / "config.json")
            if config is None or all(previous.get(key) == config[key] for key in MODEL_KEYS):
                models.append(path)
        return max(models, key=lambda path: path.stat().st_mtime, default=None)

    def state(self, run_dir: Path, *, include_log: bool = True) -> dict[str, Any]:
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
        # Older runs used solution_found for complete placement, including failed equivalence checks.
        state.setdefault("complete_candidate", bool(state["solution_found"]))
        state["verified_solution"] = bool(
            state.get("verified_solution", state["solution_found"] and state["equivalent"] in {"STRONG", "WEAK"})
        )
        state["solution_found"] = state["verified_solution"]
        alive = self.active(run_dir)
        cancelling = (run_dir / "cancel").exists()
        if alive and cancelling:
            state["status"] = "cancelling"
        elif not alive and run_dir not in self.pending and state["status"] not in FINISHED:
            state["status"] = "cancelled" if cancelling or state["status"] == "queued" else "failed"
            if state["status"] == "failed":
                state["error"] = "The training process stopped unexpectedly. Check the run log."
        # A worker may publish its final state just before exiting; keep starts locked until it exits.
        if alive and state["status"] in FINISHED:
            state["status"] = "cancelling" if cancelling else "running"
        state.update(
            id=run_dir.name,
            config=config,
            manifest=_read_json(run_dir / "manifest.json"),
            checkpoint_available=not alive and self.saved_model({**config, "resume_run_id": run_dir.name}) is not None,
            files=[
                {"name": path.name, "size": path.stat().st_size, "url": f"/api/files/{run_dir.name}/{path.name}"}
                for path in _files(run_dir)
                if path.stat().st_size > 0 and not alive
            ],
        )
        if include_log:
            state["log"] = ""
            log_path = run_dir / "worker.log"
            if log_path.is_file() and not log_path.is_symlink():
                with log_path.open("rb") as stream:
                    stream.seek(0, 2)
                    stream.seek(max(0, stream.tell() - 8000))
                    state["log"] = stream.read().decode("utf-8", errors="replace")
        return state

    def history(self) -> list[dict[str, Any]]:
        history = []
        for path in self.directory.iterdir():
            try:
                history.append(self.state(self.run_directory(path.name), include_log=False))
            except (NotFound, OSError):
                continue
        return sorted(history, key=lambda state: str(state["manifest"].get("created_at", "")), reverse=True)

    def batch(self, history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        if self.batch_id is None:
            return None
        members = [
            run
            for run in (history if history is not None else self.history())
            if run["manifest"].get("batch_id") == self.batch_id
        ]
        finished = sum(run["status"] in FINISHED for run in members)
        return {
            "id": self.batch_id,
            "total": len(members),
            "completed": finished,
            "remaining": len(members) - finished,
            "verified": sum(bool(run.get("verified_solution", run["solution_found"])) for run in members),
            "cancelled": sum(run["status"] == "cancelled" for run in members),
        }

    def snapshot(self) -> dict[str, Any]:
        result = {
            "run": self.state(self.current) if self.current else None,
            "can_resume": self.saved_model() is not None,
        }
        if self.batch_id:
            result["batch"] = self.batch()
        return result

    def start(self, config: dict[str, Any]) -> None:
        if self.closed:
            msg = "The GUI is shutting down. Restart it before starting another run."
            raise Conflict(msg)
        if self.pending or (self.process is not None and self.process.poll() is None):
            msg = "A run is already active. Cancel it or wait for it to finish."
            raise Conflict(msg)
        model = self.saved_model(config) if config["resume"] else None
        if config["resume"] and model is None:
            msg = "No saved model matches this benchmark, technology, clocking scheme and grid size. Start a fresh run."
            raise BadRequest(msg)
        self.batch_stop = threading.Event()
        count = config["seed_count"]
        self.batch_id = uuid.uuid4().hex if count > 1 else None
        circuit = Path(__file__).parent / "benchmarks" / config["benchmark"] / f"{config['function']}.v"
        manifest = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "circuit_sha256": _sha256(circuit),
            "source_checkpoint": {"run_id": model.parents[1].name, "filename": model.name, "sha256": _sha256(model)}
            if model
            else None,
            "checkpoint_warnings": _checkpoint_warnings(model, config) if model else [],
            "batch_id": self.batch_id,
            "batch_count": count,
        }
        try:
            for index in range(count):
                run_dir = self.directory / uuid.uuid4().hex
                run_dir.mkdir()
                _write_json(run_dir / "config.json", {**config, "seed": config["seed"] + index, "seed_count": 1})
                _write_json(
                    run_dir / "manifest.json",
                    {**manifest, "batch_index": index + 1, "created_at": datetime.now(UTC).isoformat()},
                )
                _write_json(run_dir / "status.json", {"status": "queued"})
                self.pending.append(run_dir)
            run_dir = self.pending.pop(0)
            if model is not None:
                (run_dir / "models").mkdir()
                # Recovery and legacy checkpoint names must load through the same canonical main.py path.
                clock = "ROW" if config["technology"] == "SiDB" else config["clocking_scheme"]
                filename = (
                    f"ppo_{config['technology']}_{config['benchmark']}_{config['function']}_"
                    f"{clock}_{config['layout_width']}x{config['layout_height']}.zip"
                )
                shutil.copy2(model, run_dir / "models" / filename)
            self.launch(run_dir)
        except Exception:
            self.cancel_pending()
            if run_dir.is_dir():
                _write_json(
                    run_dir / "status.json", {"status": "failed", "error": "The training process could not be started."}
                )
            raise
        if self.pending:
            threading.Thread(target=self.advance_batch, args=(self.batch_stop,), daemon=True).start()

    def launch(self, run_dir: Path) -> None:
        _write_json(run_dir / "status.json", {"status": "starting"})
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

    def advance_batch(self, stop: threading.Event) -> None:
        # ponytail: one sequential worker; add scheduling only if parallel experiments are needed.
        while not stop.wait(0.25):
            with self.lock:
                if self.closed or stop.is_set() or not self.pending:
                    return
                if self.process is not None and self.process.poll() is None:
                    continue
                next_run = self.pending.pop(0)
                try:
                    self.launch(next_run)
                except OSError:
                    _write_json(
                        next_run / "status.json",
                        {"status": "failed", "error": "The training process could not be started."},
                    )

    def cancel_pending(self) -> None:
        self.batch_stop.set()
        for run_dir in self.pending:
            # A queued run outside self.pending is also reported cancelled after a restart.
            with suppress(OSError):
                _write_json(run_dir / "status.json", {"status": "cancelled"})
        self.pending.clear()

    def cancel(self) -> None:
        self.cancel_pending()
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
            self.cancel_pending()
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
        "time_steps": None,
        "automatic_time_steps": data.get("time_steps") is None,
        "seed": 42,
        "optimize": True,
        "minimal_layout_dimension": True,
        "resume": False,
        "resume_run_id": None,
        "stop_on_solution": False,
        "seed_count": 1,
    }
    if data.keys() - config.keys():
        msg = "Unknown configuration parameter."
        raise BadRequest(msg)
    config.update(data)
    source = config["resume_run_id"]
    if source is not None and (not isinstance(source, str) or not _valid_id(source)):
        msg = "Select a saved run as the checkpoint source."
        raise BadRequest(msg)
    benchmark, function = config["benchmark"], config["function"]
    if not isinstance(benchmark, str) or not isinstance(function, str) or function not in benchmarks.get(benchmark, []):
        msg = "Select a bundled benchmark and circuit."
        raise BadRequest(msg)
    if config["clocking_scheme"] not in CLOCKS or config["technology"] not in TECHNOLOGIES:
        msg = "Select a supported technology and clocking scheme."
        raise BadRequest(msg)
    for key in ("optimize", "minimal_layout_dimension", "resume", "stop_on_solution", "automatic_time_steps"):
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
    # Keep explicit/resolved budgets unchanged, including those restored from old runs.
    if config["automatic_time_steps"] and config["time_steps"] is None:
        config["time_steps"] = recommended_timesteps(placement_node_count(benchmark, function))
    for key, lower, upper in (
        ("layout_width", 1, MAX_DIMENSION),
        ("layout_height", 1, MAX_DIMENSION),
        ("time_steps", 1, MAX_TIMESTEPS),
        ("seed", 0, 2**32 - 1),
        ("seed_count", 1, 10),
    ):
        if type(config[key]) is not int or not lower <= config[key] <= upper:
            msg = f"{key} must be an integer between {lower} and {upper}."
            raise BadRequest(msg)
    if config["seed"] + config["seed_count"] - 1 >= 2**32:
        msg = "The last seed must be at most 4294967295."
        raise BadRequest(msg)
    if config["resume"] and config["seed_count"] > 1:
        msg = "Seed sweeps must start fresh. Select one seed to resume a checkpoint."
        raise BadRequest(msg)
    return config


def _rewards_csv(state: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["timesteps", "mean_reward"])
    for sample in state.get("reward_history", []):
        writer.writerow(sample)
    return output.getvalue()


def create_app(runs_dir: Path | str = "nanoplacer-runs") -> Flask:
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=64 * 1024, TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"])
    token = secrets.token_urlsafe(32)
    benchmark_dir = Path(__file__).parent / "benchmarks"
    benchmarks = {
        folder.name: sorted(path.stem for path in folder.glob("*.v"))
        for folder in sorted(benchmark_dir.iterdir())
        if folder.is_dir()
    }
    runs = Runs(Path(runs_dir).expanduser(), benchmarks)
    app.extensions["nanoplacer_runs"] = runs

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
            limits={
                "max_dimension": MAX_DIMENSION,
                "max_tiles": MAX_DIMENSION**2,
                "max_timesteps": MAX_TIMESTEPS,
                "max_seeds": 10,
            },
            version=version("mnt.nanoplacer"),
        )

    @app.get("/api/circuit")
    def circuit():
        benchmark, function = request.args.get("benchmark"), request.args.get("function")
        if function not in benchmarks.get(benchmark, []):
            msg = "Select a bundled benchmark and circuit."
            raise BadRequest(msg)
        nodes = placement_node_count(benchmark, function)
        return jsonify(placement_nodes=nodes, recommended_timesteps=recommended_timesteps(nodes))

    @app.get("/api/status")
    def status():
        with runs.lock:
            return jsonify(runs.snapshot())

    @app.get("/api/runs")
    def history():
        with runs.lock:
            items = runs.history()
            return jsonify(
                runs=items,
                active_run_id=runs.current.name if runs.current and runs.active(runs.current) else None,
                batch=runs.batch(items),
            )

    @app.get("/api/runs/<run_id>")
    def run_details(run_id: str):
        with runs.lock:
            return jsonify(run=runs.state(runs.run_directory(run_id)))

    @app.post("/api/resume-check")
    def resume_check():
        data = request.get_json(silent=True)
        config = _configuration({**data, "resume": False} if isinstance(data, dict) else data, benchmarks)
        with runs.lock:
            model = runs.saved_model(config) if config["seed_count"] == 1 else None
            return jsonify(
                can_resume=model is not None,
                source_run_id=model.parents[1].name if model else None,
                filename=model.name if model else None,
                checkpoint_kind=("recovery" if model.name == "recovery.zip" else "final") if model else None,
                warnings=_checkpoint_warnings(model, config) if model else [],
                reason=None
                if model
                else "No checkpoint matches this circuit, technology, clocking scheme and grid. Seed sweeps must start fresh.",
            )

    @app.post("/api/start")
    def start_run():
        try:
            config = _configuration(request.get_json(silent=True), benchmarks)
            nodes = placement_node_count(config["benchmark"], config["function"])
            if config["layout_width"] * config["layout_height"] < nodes:
                msg = (
                    f"The grid needs at least {nodes} tiles for this circuit's placement nodes, plus room for routing."
                )
                raise BadRequest(msg)
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
            run_id = request.args.get("run_id")
            run_dir = runs.run_directory(run_id) if run_id is not None else runs.current
            frame = request.args.get("frame")
            if frame is not None and (not frame.isdecimal() or len(frame) > 3 or not 0 <= int(frame) < 128):
                return jsonify(error="Select a recorded replay frame between 0 and 127."), 400
            if run_dir is None:
                data = {}
            elif frame is not None:
                data = (
                    {} if (run_dir / "replay").is_symlink() else _read_json(run_dir / "replay" / f"{int(frame)}.json")
                )
            else:
                data = _read_json(run_dir / "preview.json")
            return jsonify(data) if data else (jsonify(error="No placement preview yet."), 404)

    @app.get("/api/export/<run_id>/<kind>")
    def export(run_id: str, kind: str):
        with runs.lock:
            run_dir = runs.run_directory(run_id)
            if runs.active(run_dir) or run_dir in runs.pending:
                return jsonify(error="Wait for this run to finish before exporting it."), 409
            if kind not in {"experiment.zip", "rewards.csv", "metadata.json"}:
                return jsonify(error="Export not found."), 404
            state = runs.state(run_dir, include_log=False)
        # Finished folders are immutable to the GUI; compression must not block cancelling another run.
        state.pop("files")
        metadata = {"config": state.pop("config"), "manifest": state.pop("manifest"), "status": state}
        if kind == "metadata.json":
            return send_file(
                io.BytesIO(json.dumps(metadata, indent=2).encode()),
                mimetype="application/json",
                as_attachment=True,
                download_name=f"{run_id}-metadata.json",
            )
        rewards = _rewards_csv(state)
        if kind == "rewards.csv":
            return send_file(
                io.BytesIO(rewards.encode()),
                mimetype="text/csv",
                as_attachment=True,
                download_name=f"{run_id}-rewards.csv",
            )
        # The response owns this stream until streaming finishes, so a with block would close it too soon.
        archive_stream = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)  # noqa: SIM115
        try:
            with zipfile.ZipFile(archive_stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, value in metadata.items():
                    archive.writestr(f"{name}.json", json.dumps(value, indent=2))
                archive.writestr("rewards.csv", rewards)
                preview_data = _read_json(run_dir / "preview.json")
                if preview_data:
                    archive.writestr("preview.json", json.dumps(preview_data))
                for path in _files(run_dir):
                    archive.write(path, path.relative_to(run_dir))
            archive_stream.seek(0)
            response = send_file(
                archive_stream,
                mimetype="application/zip",
                as_attachment=True,
                download_name=f"{run_id}-experiment.zip",
            )
            response.call_on_close(archive_stream.close)
            return response
        except Exception:
            archive_stream.close()
            raise

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
