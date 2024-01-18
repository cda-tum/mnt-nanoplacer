import unittest
from unittest.mock import patch

from src.mnt.nanoplacer.main import create_layout, start


class TestNanoPlacementEnv(unittest.TestCase):
    def test_create_layout(self):
        create_layout(
            benchmark="trindade16",
            function="mux21",
            clocking_scheme="2DDWave",
            technology="QCA",
            minimal_layout_dimension=False,
            layout_width=3,
            layout_height=4,
            time_steps=10000,
            reset_model=True,
            verbose=1,
            optimize=True,
        )

    def test_start(self):
        with patch("builtins.input", side_effect=["yes"]):
            start()


if __name__ == "__main__":
    unittest.main()
