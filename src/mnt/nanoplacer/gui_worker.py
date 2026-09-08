"""Isolated training process for the optional local GUI."""

import argparse
import json
import os
import traceback
from collections import deque
from math import isfinite
from pathlib import Path
from time import monotonic

import torch
from stable_baselines3.common.callbacks import BaseCallback

from mnt import pyfiction
from mnt.nanoplacer.main import create_layout
from mnt.nanoplacer.placement_envs.nano_placement_env import NanoPlacementEnv


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def layout_preview(env: NanoPlacementEnv) -> dict:
    """Keep native coordinates, including upper-only wires and both crossing layers."""
    layout = env.layout
    width, height = layout.x() + 1, layout.y() + 1
    cells, edges = [], []
    for y in range(height):
        for x in range(width):
            for z in range(layout.z() + 1):
                tile = (x, y, z)
                if layout.is_empty_tile(tile):
                    continue
                node = layout.get_node(tile)
                gate_type = "gate"
                for kind in ("pi", "po", "inv", "and", "nand", "or", "nor", "xor", "xnor", "maj"):
                    if getattr(layout, f"is_{kind}")(node):
                        gate_type = kind
                        break
                if gate_type == "gate" and layout.is_wire(node):
                    gate_type = "fanout" if layout.fanout_size(node) > 1 else "buf"
                cells.append(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "type": gate_type,
                        "name": layout.get_name(layout.make_signal(node)) if gate_type in {"pi", "po"} else "",
                        "phase": int(layout.get_clock_number(tile)) + 1,
                    }
                )
                for fanin in layout.fanins(tile):
                    edges.append({"source": [fanin.x, fanin.y, fanin.z], "target": [x, y, z]})
    return {
        "width": width,
        "height": height,
        "clocking_scheme": env.clocking_scheme,
        "cells": cells,
        "edges": edges,
        "phases": [[int(layout.get_clock_number((x, y, 0))) + 1 for x in range(width)] for y in range(height)],
    }


class _Progress(BaseCallback):
    def __init__(self, run_dir: Path) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.started = monotonic()
        self.last_write = 0.0
        self.baseline = 0
        self.cancelled = False
        self.episode_rewards: deque[float] = deque(maxlen=100)
        self.reward_stride = 1
        self.status = {
            "status": "starting",
            "timesteps": 0,
            "total_timesteps": 0,
            "episodes": 0,
            "mean_reward": None,
            "reward_window": 0,
            "reward_history": [],
            "elapsed": 0.0,
            "best_placed": 0,
            "total_nodes": 0,
            "solution_found": False,
            "equivalent": None,
            "preview_revision": 0,
            "error": None,
        }

    def publish(self) -> None:
        self.last_write = monotonic()
        self.status["elapsed"] = round(self.last_write - self.started, 3)
        _write_json(self.run_dir / "status.json", self.status)

    def best(self, env: NanoPlacementEnv) -> None:
        _write_json(self.run_dir / "preview.json", layout_preview(env))
        complete = env.max_placed_nodes == len(env.actions)
        if complete:
            output = self.run_dir / "layouts"
            output.mkdir(exist_ok=True)
            temporary = output / "layout.tmp.fgl"
            pyfiction.write_fgl_layout(env.layout, str(temporary))
            temporary.replace(output / "layout.fgl")
        self.status.update(
            best_placed=env.max_placed_nodes,
            total_nodes=len(env.actions),
            solution_found=complete,
            equivalent=env.equivalent,
            preview_revision=self.status["preview_revision"] + 1,
        )
        self.publish()

    def _on_training_start(self) -> None:
        # Resumed PPO models retain their lifetime timestep count; the GUI reports this run.
        self.baseline = self.num_timesteps
        self.reward_stride = max(1, (self.status["total_timesteps"] + 239) // 240)
        self.status.update(status="running", total_nodes=len(self.training_env.get_attr("actions")[0]))
        self.publish()

    def _on_step(self) -> bool:
        self.status["timesteps"] = self.num_timesteps - self.baseline
        self.status["episodes"] += int(sum(self.locals["dones"]))
        completed_reward = False
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None and isfinite(episode["r"]):
                self.episode_rewards.append(float(episode["r"]))
                completed_reward = True
        if completed_reward:
            mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            self.status.update(mean_reward=mean_reward, reward_window=len(self.episode_rewards))
            history = self.status["reward_history"]
            point = [self.status["timesteps"], mean_reward]
            if len(history) < 2 or point[0] // self.reward_stride > history[-1][0] // self.reward_stride:
                history.append(point)
            else:
                history[-1] = point  # Last measured mean in this bucket; never replace the first point.
            if len(history) > 256:
                # ponytail: bounded samples, not a full episode trace. Coarsen future buckets too.
                history[:] = history[::2]  # 257 points: preserve first and latest, without changing means.
                self.reward_stride *= 2
        self.cancelled = (self.run_dir / "cancel").exists()
        if self.cancelled or monotonic() - self.last_write >= 0.25:
            self.publish()
        return not self.cancelled


def run_worker(run_dir: Path) -> int:
    """Run in a dedicated working directory; never expose tracebacks through GUI status."""
    progress = _Progress(run_dir)
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        progress.status["total_timesteps"] = config["time_steps"]
        progress.publish()
        torch.set_num_threads(1)
        create_layout(
            benchmark=config["benchmark"],
            function=config["function"],
            clocking_scheme=config["clocking_scheme"],
            technology=config["technology"],
            layout_width=config["layout_width"],
            layout_height=config["layout_height"],
            time_steps=config["time_steps"],
            seed=config["seed"],
            optimize=config["optimize"],
            reset_model=not config["resume"],
            minimal_layout_dimension=False,
            verbose=0,
            on_best=progress.best,
            callback=progress,
        )
        # create_layout saves the checkpoint after learn() returns, including cancellation.
        progress.status["status"] = "cancelled" if progress.cancelled else "completed"
        progress.publish()
    except Exception:
        traceback.print_exc()
        progress.status.update(status="failed", error="Training failed. See worker.log for details.")
        progress.publish()
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    os.chdir(run_dir)
    raise SystemExit(run_worker(run_dir))


if __name__ == "__main__":
    main()
