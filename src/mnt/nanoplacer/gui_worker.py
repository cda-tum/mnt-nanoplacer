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
from mnt.nanoplacer.main import _save_checkpoint, create_layout
from mnt.nanoplacer.placement_envs.nano_placement_env import NanoPlacementEnv

MAX_REPLAY_FRAMES = 128
CHECKPOINT_INTERVAL = 60.0


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
        self.last_checkpoint = self.started
        self.baseline = 0
        self.epoch_baseline = 0
        self.cancelled = False
        self.episode_rewards: deque[float] = deque(maxlen=100)
        self.reward_stride = 1
        self.replay_stride = 1
        self.last_replay_placed = 0
        self.status = {
            "status": "starting",
            "timesteps": 0,
            "total_timesteps": 0,
            "episodes": 0,
            "mean_reward": None,
            "reward_window": 0,
            "reward_history": [],
            "placement_history": [],
            "ppo_epochs": 0,
            "ppo_epochs_total": 0,
            "elapsed": 0.0,
            "best_placed": 0,
            "total_nodes": 0,
            "solution_found": False,
            "verified_solution": False,
            "complete_candidate": False,
            "equivalent": None,
            "best_metrics": {},
            "first_solution_time": None,
            "first_solution_timestep": None,
            "first_solution_ppo_epochs": None,
            "first_solution_ppo_epochs_total": None,
            "reported_dimensions": None,
            "target_reproduced": False,
            "target_equivalent": None,
            "successful_episodes": 0,
            "complete_episodes": 0,
            "routing_failures": 0,
            "preview_revision": 0,
            "replay_count": 0,
            "replay_truncated": False,
            "stop_reason": None,
            "checkpoint": None,
            "checkpoint_error": None,
            "error": None,
        }

    def training_epochs(self) -> None:
        # SB3's _n_updates counts PPO epochs, not optimizer steps or rollout iterations.
        total = getattr(getattr(self, "model", None), "_n_updates", 0)
        self.status.update(ppo_epochs=max(0, total - self.epoch_baseline), ppo_epochs_total=total)

    def checkpoint_saved(self, path: Path, kind: str) -> None:
        self.training_epochs()
        self.status.update(
            checkpoint={
                "filename": path.name,
                "kind": kind,
                "timestep": self.num_timesteps - self.baseline,
                "total_timesteps": self.num_timesteps,
                "ppo_epochs_total": self.status["ppo_epochs_total"],
                "elapsed": round(monotonic() - self.started, 3),
            },
            checkpoint_error=None,
        )

    def save_recovery(self) -> None:
        now = monotonic()
        if self.num_timesteps <= self.baseline or now - self.last_checkpoint < CHECKPOINT_INTERVAL:
            return
        self.last_checkpoint = now
        try:
            checkpoint = self.run_dir / "models" / "recovery.zip"
            checkpoint.parent.mkdir(exist_ok=True)
            _save_checkpoint(self.model, checkpoint)
            self.checkpoint_saved(checkpoint, "recovery")
        except Exception:
            traceback.print_exc()
            self.status["checkpoint_error"] = "Recovery checkpoint could not be saved. See worker.log for details."
        self.publish()

    def placement_sample(self, timestep: int) -> None:
        history = self.status["placement_history"]
        point = [timestep, self.status["best_placed"], self.status["total_nodes"]]
        if history and history[-1][0] == timestep:
            history[-1] = point
        else:
            history.append(point)
        if len(history) > 256:
            # ponytail: bounded best-placement milestones, not a per-step trajectory.
            history[:] = history[::2]

    def publish(self) -> None:
        self.last_write = monotonic()
        self.status["elapsed"] = round(self.last_write - self.started, 3)
        _write_json(self.run_dir / "status.json", self.status)

    def best(self, env: NanoPlacementEnv) -> None:
        complete = env.max_placed_nodes == len(env.actions)
        verified = env.verified_solution
        self.training_epochs()
        preview = layout_preview(env)
        preview["metadata"] = {
            # This hook runs inside env.step(), before SB3 increments its timestep counter.
            "timestep": max(0, self.num_timesteps - self.baseline) + 1,
            "elapsed": round(monotonic() - self.started, 3),
            "placed": env.max_placed_nodes,
            "verified": verified,
        }
        _write_json(self.run_dir / "preview.json", preview)
        if (
            not self.status["replay_count"]
            or complete
            or env.max_placed_nodes - self.last_replay_placed >= self.replay_stride
        ):
            if self.status["replay_count"] < MAX_REPLAY_FRAMES:
                replay = self.run_dir / "replay"
                replay.mkdir(exist_ok=True)
                _write_json(replay / f"{self.status['replay_count']}.json", preview)
                self.status["replay_count"] += 1
                self.last_replay_placed = env.max_placed_nodes
            else:
                # ponytail: immutable bounded milestones; the live preview still retains every new best.
                self.status["replay_truncated"] = True
        if verified:
            output = self.run_dir / "layouts"
            output.mkdir(exist_ok=True)
            temporary = output / "layout.tmp.fgl"
            pyfiction.write_fgl_layout(env.layout, str(temporary))
            temporary.replace(output / "layout.fgl")
        self.status.update(
            best_placed=env.max_placed_nodes,
            total_nodes=len(env.actions),
            solution_found=verified,
            verified_solution=verified,
            complete_candidate=complete,
            equivalent=env.equivalent,
            best_metrics=env.best_metrics,
            first_solution_time=env.first_solution_time,
            reported_dimensions=env.reported_dimensions,
            target_reproduced=env.target_reproduced,
            target_equivalent=env.target_equivalent,
            preview_revision=self.status["preview_revision"] + 1,
        )
        if verified and self.status["first_solution_timestep"] is None:
            self.status.update(
                first_solution_timestep=preview["metadata"]["timestep"],
                first_solution_ppo_epochs=self.status["ppo_epochs"],
                first_solution_ppo_epochs_total=self.status["ppo_epochs_total"],
            )
        self.placement_sample(preview["metadata"]["timestep"])
        self.publish()

    def _on_training_start(self) -> None:
        # Resumed PPO models retain their lifetime timestep count; the GUI reports this run.
        self.baseline = self.num_timesteps
        self.epoch_baseline = getattr(self.model, "_n_updates", 0)
        self.training_epochs()
        self.last_checkpoint = monotonic()
        self.reward_stride = max(1, (self.status["total_timesteps"] + 239) // 240)
        self.status.update(status="running", total_nodes=len(self.training_env.get_attr("actions")[0]))
        self.status["reported_dimensions"] = self.training_env.get_attr("reported_dimensions")[0]
        self.placement_sample(0)
        self.replay_stride = max(1, (self.status["total_nodes"] + 119) // 120)
        self.publish()

    def _on_rollout_start(self) -> None:
        self.training_epochs()
        self.save_recovery()

    def _on_training_end(self) -> None:
        self.training_epochs()
        self.placement_sample(self.num_timesteps - self.baseline)
        self.publish()

    def _on_step(self) -> bool:
        self.status["timesteps"] = self.num_timesteps - self.baseline
        self.training_epochs()
        self.status["episodes"] += int(sum(self.locals["dones"]))
        for done, info in zip(self.locals["dones"], self.locals.get("infos", []), strict=False):
            if done:
                self.status["complete_episodes"] += int(bool(info.get("complete_candidate")))
                self.status["successful_episodes"] += int(bool(info.get("verified")))
                self.status["routing_failures"] += int(bool(info.get("routing_failed")))
                if info.get("target_reproduced"):
                    self.status.update(target_reproduced=True, target_equivalent=info["target_equivalent"])
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
        if not self.cancelled:
            self.save_recovery()
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
            stop_on_solution=config.get("stop_on_solution", False),
            reset_model=not config["resume"],
            minimal_layout_dimension=False,
            verbose=0,
            on_best=progress.best,
            callback=progress,
        )
        # create_layout saves the checkpoint after learn() returns, including cancellation.
        checkpoint = next((run_dir / "models").glob("ppo_*.zip"), None)
        if checkpoint is not None:
            progress.checkpoint_saved(checkpoint, "final")
        progress.status["status"] = "cancelled" if progress.cancelled else "completed"
        progress.status["stop_reason"] = (
            "cancelled"
            if progress.cancelled
            else "solution"
            if config.get("stop_on_solution") and progress.status["verified_solution"]
            else "budget"
        )
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
