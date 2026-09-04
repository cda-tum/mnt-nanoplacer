[![PyPI](https://img.shields.io/pypi/v/mnt.nanoplacer?logo=pypi&style=flat-square)](https://pypi.org/project/mnt.nanoplacer/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/github/actions/workflow/status/cda-tum/mnt-nanoplacer/deploy.yml?branch=main&style=flat-square&logo=github&label=python)](https://github.com/cda-tum/mnt-nanoplacer/actions/workflows/deploy.yml)
[![Ruff](https://img.shields.io/badge/lint%20%26%20format-Ruff-D7FF64?style=flat-square&logo=ruff)](https://docs.astral.sh/ruff/)

# NanoPlaceR: Reinforcement-learning placement and routing for FCN

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mnt-nanoplacer/main/docs/_static/mnt_light.svg" width="60%">
    <img src="https://raw.githubusercontent.com/cda-tum/mnt-nanoplacer/main/docs/_static/mnt_dark.svg" width="60%" alt="Munich Nanotech Toolkit logo">
  </picture>
</p>

NanoPlaceR is an open-source physical-design tool for field-coupled nanocomputing (FCN). It uses masked Proximal Policy Optimization to place logic elements and routes each partial placement with A\* search. It can generate layouts for logic networks of roughly 200 gates while using about 50% less area than the state-of-the-art heuristic approach described in the publications below.

![Overview of the NanoPlaceR methodology](https://raw.githubusercontent.com/cda-tum/mnt-nanoplacer/main/docs/_static/lbr.png)

## Installation

NanoPlaceR requires Python 3.10 or newer and `mnt.pyfiction` 0.8 or newer. The CI suite covers Python 3.10 and 3.13.

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install mnt.nanoplacer
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Usage

Use NanoPlaceR from Python:

```python
from mnt.nanoplacer import create_layout

create_layout(
    benchmark="trindade16",
    function="mux21",
    clocking_scheme="2DDWave",
    technology="QCA",
    minimal_layout_dimension=False,
    layout_width=3,
    layout_height=4,
    time_steps=10_000,
    reset_model=True,
    verbose=1,
    optimize=True,
)
```

Or use the command-line interface:

```console
mnt.nanoplacer --help
mnt.nanoplacer --benchmark trindade16 --function mux21 \
  --clocking-scheme 2DDWave --technology Gate-level \
  --minimal-layout-dimension --time-steps 10000 --verbose 1
```

Runs store generated layouts in `layouts/`, trained agents in `models/`, and TensorBoard data in `tensorboard/`. By default, the CLI resumes a matching saved model when one exists; pass `--reset-model` to train from scratch.

## Repository structure

```text
src/mnt/nanoplacer/
├── benchmarks/                  Verilog benchmark circuits
├── main.py                      Python and command-line entry point
└── placement_envs/
    ├── nano_placement_env.py    Gymnasium placement environment
    └── utils/
        ├── layout_dimensions.py Predefined minimal dimensions
        └── placement_utils.py   Network and action helpers
tests/                           Unit and integration tests
```

## References

If you use NanoPlaceR in your work, please cite the following publications:

- [Late Breaking Results From Hybrid Design Automation for Field-coupled Nanotechnologies (DAC 2023)](https://www.cda.cit.tum.de/files/eda/2023_dac_late_breaking_results_from_hybrid_design_automation_for_field_coupled_nanotechnologies.pdf)
- [Thinking Outside the Clock: Physical Design for Field-coupled Nanocomputing with Deep Reinforcement Learning (ISQED 2024)](https://www.cda.cit.tum.de/files/eda/2024_isqed_thinking_outside_the_clock_physical_design_for_field-coupled_nanocomputing_with_deep_reinforcement_learning.pdf)

```bibtex
@INPROCEEDINGS{hofmann2023nanoplacer,
  author        = {S. Hofmann and M. Walter and L. Servadei and R. Wille},
  title         = {{Late Breaking Results From Hybrid Design Automation for Field-coupled Nanotechnologies}},
  booktitle     = {{2023 60th ACM/IEEE Design Automation Conference (DAC)}},
  year          = {2023},
}
```

```bibtex
@INPROCEEDINGS{hofmann2024nanoplacer,
  author        = {S. Hofmann and M. Walter and L. Servadei and R. Wille},
  title         = {{Thinking Outside the Clock: Physical Design for Field-coupled Nanocomputing with Deep Reinforcement Learning}},
  booktitle     = {{2024 25th International Symposium on Quality Electronic Design (ISQED)}},
  year          = {2024},
}
```
