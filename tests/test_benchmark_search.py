import runpy
from pathlib import Path

import pytest

import mnt
from mnt.nanoplacer.placement_envs.nano_placement_env import NanoPlacementEnv


@pytest.mark.parametrize("change_reward", [False, True])
def test_trace_ignores_diagnostics_but_checks_behavior(tmp_path, monkeypatch, change_reward) -> None:
    # The script prefers its checkout; do not leave its path insertion in other package tests.
    monkeypatch.setattr(mnt, "__path__", list(mnt.__path__))
    benchmark = runpy.run_path(str(Path(__file__).parents[1] / "scripts/benchmark_search.py"))

    class LegacyInfoEnv(NanoPlacementEnv):
        def step(self, action):
            observation, reward, terminated, truncated, _ = super().step(action)
            return observation, reward + int(change_reward), terminated, truncated, {}

    args = (LegacyInfoEnv, NanoPlacementEnv, "trindade16/mux21", 42, 20, tmp_path / "trace")
    if change_reward:
        with pytest.raises(AssertionError, match="reward/termination"):
            benchmark["verify_trace"](*args)
    else:
        result = benchmark["verify_trace"](*args)
        assert result["steps"] == 20
        assert result["episodes"] > 0
