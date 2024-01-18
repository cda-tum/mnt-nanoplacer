import unittest
from unittest.mock import patch

import numpy as np

from src.mnt.nanoplacer.placement_envs.nano_placement_env import NanoPlacementEnv


class TestNanoPlacementEnv(unittest.TestCase):
    def setUp(self):
        self.env = NanoPlacementEnv(
            clocking_scheme="2DDWave",
            technology="Gate-level",
            layout_width=3,
            layout_height=4,
            benchmark="trindade16",
            function="mux21",
            verbose=1,
            optimize=True,
        )

    def test_reset(self):
        observation, _ = self.env.reset()
        assert observation == 0

    def test_step_valid_action(self):
        self.env.reset()
        action = np.random.randint(self.env.action_space.n)
        observation, reward, done, _, _ = self.env.step(action)
        assert isinstance(observation, int)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_action(self):
        self.env.reset()
        action = 0
        observation, reward, done, _, _ = self.env.step(action)
        assert isinstance(observation, int)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_save_layout(self):
        with patch("mnt.pyfiction.apply_qca_one_library"), patch("mnt.pyfiction.write_qca_layout_svg"):
            self.env.save_layout()

        with patch("mnt.pyfiction.hexagonalization"), patch("mnt.pyfiction.write_dot_layout"):
            self.env.technology = "SiDB"
            self.env.save_layout()

        with patch("mnt.pyfiction.write_fgl_layout"):
            self.env.technology = "Gate-level"
            self.env.save_layout()

        with self.assertRaises(Exception):  # noqa: B017
            self.env.technology = "InvalidTech"
            self.env.save_layout()

    def test_place_mux(self):
        self.env.step(3)
        self.env.step(6)
        self.env.step(0)
        self.env.step(1)
        self.env.step(7)
        self.env.step(2)
        self.env.step(5)
        self.env.step(8)
        _, reward, done, _, _ = self.env.step(11)
        assert reward > 1000
        assert isinstance(done, bool)

    def test_action_masks(self):
        masks = self.env.action_masks()
        assert isinstance(masks, list)
        assert all(isinstance(mask, np.bool_) for mask in masks)

    def test_calculate_reward(self):
        x, y = np.random.randint(self.env.layout_width), np.random.randint(self.env.layout_height)
        placed_node = True
        reward, done = self.env.calculate_reward(x, y, placed_node)
        assert isinstance(reward, float)
        assert isinstance(done, bool)


if __name__ == "__main__":
    unittest.main()
