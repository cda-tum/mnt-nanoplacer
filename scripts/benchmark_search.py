"""Compare a checked-out environment with a Git baseline, without changing training settings.

Example: python scripts/benchmark_search.py --baseline-ref 80892fd --seconds 30 --output /tmp/search-benchmark
The timer includes PPO updates, but excludes model initialization. A native call or PPO
update can overrun the deadline; actual elapsed time is always reported. Baselines
80892fd and 0511ba1 retain only their first completed candidate, not every episode's best.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from contextlib import chdir
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from types import ModuleType

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

import mnt

# Prefer this checkout, including when pyfiction's regular namespace package is installed.
ROOT = Path(__file__).resolve().parents[1]
mnt.__path__.insert(0, str(ROOT / "src" / "mnt"))

from mnt.nanoplacer.placement_envs.nano_placement_env import NanoPlacementEnv  # noqa: E402
from mnt.nanoplacer.placement_envs.utils import layout_dimensions  # noqa: E402

ENV_PATH = "src/mnt/nanoplacer/placement_envs/nano_placement_env.py"


def baseline_environment(ref: str) -> type:
    source = subprocess.check_output(["git", "show", f"{ref}:{ENV_PATH}"], cwd=ROOT, text=True)
    module = ModuleType("benchmark_baseline")
    exec(compile(source, f"{ref}:{ENV_PATH}", "exec"), module.__dict__)  # Trusted code from the requested Git revision.
    return module.NanoPlacementEnv


def environment(env_class: type, circuit: str, **kwargs: object) -> NanoPlacementEnv:
    benchmark, function = circuit.split("/", 1)
    width, height = layout_dimensions["2DDWave"][benchmark][function]
    return env_class(
        benchmark=benchmark,
        function=function,
        layout_width=width,
        layout_height=height,
        technology="Gate-level",
        clocking_scheme="2DDWave",
        optimize=True,
        verbose=0,
        **kwargs,
    )


def layout_state(env: NanoPlacementEnv) -> tuple:
    """Include routing topology, not just occupied tiles or a rendered preview."""
    layout = env.layout
    cells = []
    for y in range(layout.y() + 1):
        for x in range(layout.x() + 1):
            for z in range(layout.z() + 1):
                tile = (x, y, z)
                if not layout.is_empty_tile(tile):
                    cells.append((tile, layout.get_node(tile), tuple((f.x, f.y, f.z) for f in layout.fanins(tile))))
    return (layout.x(), layout.y(), tuple(cells))


def verify_trace(baseline: type, candidate: type, circuit: str, seed: int, steps: int, output: Path) -> dict:
    """Feed both environments identical legal actions and compare their complete state."""
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    envs = [environment(baseline, circuit), environment(candidate, circuit)]
    for env in envs:
        env.reset(seed=seed)
    digest = hashlib.sha256()
    episodes = 0
    with chdir(output):
        for step in range(steps):
            masks = [env.action_masks() for env in envs]
            assert masks[0] == masks[1], (circuit, seed, step, "mask")
            assert all(env.action_masks() == masks[0] for env in envs), (circuit, seed, step, "repeated mask")
            action = int(rng.choice(np.flatnonzero(masks[0])))
            transitions = [env.step(action) for env in envs]
            assert transitions[0] == transitions[1], (circuit, seed, step, "reward/termination")
            states = [
                (
                    layout_state(env),
                    env.current_node,
                    env.current_pi,
                    env.current_po,
                    env.current_tries,
                    env.max_tries,
                    env.placement_possible,
                    env.node_dict,
                    env.tried_positions,
                    env.occupied_tiles.tobytes(),
                    env.max_placed_nodes,
                    env.equivalent,
                )
                for env in envs
            ]
            assert states[0] == states[1], (circuit, seed, step, "layout/state")
            digest.update(repr((action, transitions[0], states[0][0])).encode())
            if transitions[0][2] or transitions[0][3]:
                episodes += 1
                for env in envs:
                    env.reset()
    for env in envs:
        env.close()
    return {"circuit": circuit, "seed": seed, "steps": steps, "episodes": episodes, "sha256": digest.hexdigest()}


class Budget(BaseCallback):
    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds
        self.started = 0.0
        self.episodes = 0
        self.first_solution_seconds = None
        self.quality = None

    def _on_training_start(self) -> None:
        self.started = monotonic()

    def _on_step(self) -> bool:
        self.episodes += int(sum(self.locals["dones"]))
        return monotonic() - self.started < self.seconds

    def best(self, env: NanoPlacementEnv) -> None:
        if env.current_node == len(env.actions) and env.equivalent in {"STRONG", "WEAK"}:
            if self.first_solution_seconds is None:
                self.first_solution_seconds = monotonic() - self.started
            layout = env.layout
            quality = {
                "area": (layout.x() + 1) * (layout.y() + 1),
                "width": layout.x() + 1,
                "height": layout.y() + 1,
                "wires": layout.num_wires(),
                "crossings": layout.num_crossings(),
                "equivalent": env.equivalent,
            }
            if self.quality is None or tuple(quality[k] for k in ("area", "wires", "crossings")) < tuple(
                self.quality[k] for k in ("area", "wires", "crossings")
            ):
                self.quality = quality


def run(env_class: type, circuit: str, seed: int, seconds: float, output: Path) -> dict:
    """Run one fresh, single-CPU-thread policy using NanoPlaceR's existing PPO settings."""
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    progress = Budget(seconds)
    setup_started = monotonic()
    env = environment(env_class, circuit, on_best=progress.best)
    model = MaskablePPO("MlpPolicy", env, batch_size=512, gamma=0.995, learning_rate=0.001, seed=seed, device="cpu")
    setup_seconds = monotonic() - setup_started
    with chdir(output):
        model.learn(total_timesteps=1_000_000_000, callback=progress)
    elapsed = monotonic() - progress.started
    result = {
        "circuit": circuit,
        "seed": seed,
        "budget_seconds": seconds,
        "elapsed_seconds": elapsed,
        "overrun_seconds": max(0.0, elapsed - seconds),
        "setup_seconds": setup_seconds,
        "steps": model.num_timesteps,
        "steps_per_second": model.num_timesteps / elapsed,
        "ppo_epochs": model._n_updates,
        "episodes": progress.episodes,
        "max_placed_nodes": env.max_placed_nodes,
        "total_nodes": len(env.actions),
        "first_solution_seconds": progress.first_solution_seconds,
        "verified_quality": progress.quality,
        "quality_scope": "verified complete candidates reported by on_best",
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="main")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--circuits", nargs="+", default=["trindade16/mux21", "fontes18/cm82a_5"])
    parser.add_argument("--trace-steps", type=int, default=2000)
    parser.add_argument("--variant", choices=["baseline", "candidate", "both"], default="both")
    parser.add_argument(
        "--routing-fallback", action="store_true", help="Enable reverse routing only for the candidate."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.trace_steps < 0:
        parser.error("--seconds must be finite and positive; --trace-steps must be nonnegative")
    if any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seeds must be between 0 and 4294967295")
    if args.routing_fallback and args.trace_steps:
        parser.error("--routing-fallback changes search behavior; use --trace-steps 0")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = baseline_environment(args.baseline_ref)
    candidate = partial(NanoPlacementEnv, routing_fallback=True) if args.routing_fallback else NanoPlacementEnv
    result = {
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True
        ).strip(),
        "candidate_environment_sha256": hashlib.sha256((ROOT / ENV_PATH).read_bytes()).hexdigest(),
        "candidate_routing_fallback": args.routing_fallback,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {package: version(package) for package in ("mnt.pyfiction", "sb3-contrib", "torch", "numpy")},
        "traces": [],
        "runs": [],
    }
    for circuit in args.circuits:
        for seed in args.seeds:
            prefix = f"{circuit.replace('/', '-')}-{seed}"
            if args.trace_steps:
                trace = verify_trace(
                    baseline, NanoPlacementEnv, circuit, seed, args.trace_steps, output / f"trace-{prefix}"
                )
                result["traces"].append(trace)
                print(json.dumps({"trace": trace}), flush=True)
            variants = [("baseline", baseline), ("candidate", candidate)]
            # Alternate pair order between seeds to reduce consistently favoring warm/cool starts.
            if seed % 2:
                variants.reverse()
            for name, env_class in variants:
                if args.variant not in ("both", name):
                    continue
                record = {"variant": name, **run(env_class, circuit, seed, args.seconds, output / f"{name}-{prefix}")}
                result["runs"].append(record)
                (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
