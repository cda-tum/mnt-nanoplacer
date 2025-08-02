[![PyPI](https://img.shields.io/pypi/v/mnt.nanoplacer?logo=pypi&style=flat-square)](https://pypi.org/project/mnt.nanoplacer/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Bindings](https://img.shields.io/github/actions/workflow/status/cda-tum/mnt-nanoplacer/deploy.yml?branch=main&style=flat-square&logo=github&label=python)](https://github.com/cda-tum/mnt-nanoplacer/actions/workflows/deploy.yml)
[![Code style: black][black-badge]][black-link]

# NanoPlaceR: Placement and Routing for Field-coupled Nanocomputing (FCN) based on Reinforcement Learning

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mnt-nanoplacer/main/docs/_static/mnt_light.svg" width="60%">
    <img src="https://raw.githubusercontent.com/cda-tum/mnt-nanoplacer/main/docs/_static/mnt_dark.svg" width="60%">
  </picture>
</p>

NanoPlaceR is a tool for the physical design of FCN circuitry based on Reinforcement Learning.
It can generate layouts for logic networks up to ~200 gates, while requiring ~50% less area than the state-of-the-art heuristic approach.

Related publications: 
- [DAC23](https://www.cda.cit.tum.de/files/eda/2023_dac_late_breaking_results_from_hybrid_design_automation_for_field_coupled_nanotechnologies.pdf)
- [ISQED24](https://www.cda.cit.tum.de/files/eda/2024_isqed_thinking_outside_the_clock_physical_design_for_field-coupled_nanocomputing_with_deep_reinforcement_learning.pdf)


Inspired by recent developments in the field of machine learning-aided design automation, this tool combines reinforcement learning with efficient path routing algorithms based on established algorithms such as A\* search.
Masked Proximal Policy Optimization (PPO) is used to learn the placement of logic elements, which is further accelerated by incorporating an action mask computed based on the netlist structure and the last partial placement, ensuring valid and compact solutions.
To minimize the occurrence of unpromising partial placements, several checks constantly ensure the early termination of sub-par solutions.
Furthermore, the routing of placed gates is incorporated directly into the placement step using established routing strategies.
The following figure outlines the methodology:

![](https://raw.githubusercontent.com/cda-tum/mnt-nanoplacer/main/docs/_static/lbr.png)

# Usage of NanoPlaceR

Currently, due to the Open-AI gym dependency, only python versions up to 3.10 are supported.

If you do not have a virtual environment set up, the following steps outline one possible way to do so.
First, install virtualenv:

```console
$ pip install virtualenv
```

Then create a new virtual environment in your project folder and activate it:

```console
$ mkdir nano_placement
$ cd nano_placement
$ python -m venv venv
$ source venv/bin/activate
```

NanoPlaceR can be installed via pip:

```console
(venv) $ pip install mnt.nanoplacer
```

You can then create the desired layout based on specified parameters (e.g. logic function, clocking scheme, layout width etc.) directly in your pyhon project:

```
from mnt import nanoplacer

if __name__ == "__main__":
    nanoplacer.create_layout(
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
```

or via the command:

```
(venv) $ mnt.nanoplacer
usage: mnt.nanoplacer [-h] [-b {fontes18,trindade16,EPFL,TOY,ISCAS85}] [-f FUNCTION] [-c {2DDWave,USE, RES, ESR}] [-t {QCA,SiDB, Gate-level}] [-l] [-lw LAYOUT_WIDTH] [-lh LAYOUT_HEIGHT] [-ts TIME_STEPS] [-r] [-v {0,1, 2, 3}]
Optional arguments:
  -h, --help                       Show this help message and exit.
  -b, --benchmark                  Benchmark set.
  -f, --function                   Logic function to generate layout for.
  -c, --clocking_scheme            Underlying clocking scheme.
  -t, --technology                 Underlying technology (QCA, SiDB or technology-independent Gate-level layout).
  -l, --minimal_layout_dimension   If True, experimentally found minimal layout dimensions are used (defautls to False).
  -lw, --layout_width              User defined layout width.
  -lh, --layout_height             User defined layout height.
  -ts, --time_steps                Number of time steps to train the RL agent.
  -r,  --reset_model               If True, reset saved model and train from scratch (defautls to False).
  -v,  --verbosity                 0: No information. 1: Print layout after every new best placement. 2: Print training metrics. 3: 1 and 2 combined.
  -o,  --optimize                  If True, layout will be further optimized after placement.
```

For example to create the gate-level layout for the mux21 function from trindade16 on the 2DDWave clocking scheme using the best found layout dimensions (by training for a maximum of 10000 timesteps):

```
mnt.nanoplacer -b "trindade16" -f "mux21" -c "2DDWave" -t "Gate-level" -l -ts 10000 -v 1
```

# Repository Structure

```
.
├── docs/
├── src/
│ ├── mnt/
│   └── nanoplacer/
│     ├── main.py                         # entry point for the mnt.nanoplacer script
│     ├── benchmarks/                     # common benchmark sets
│     ├── placement_envs/
│     │ └── utils/
│     │   ├── placement_utils/            # placement util functions
│     │   └── layout_dimenions/           # predefined layout dimensions for certain functions
│     └──── nano_placement_env.py           # placement environment
```

# References

In case you are using NanoPlaceR in your work, we would be thankful if you referred to it by citing the following publications:

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

[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]: https://github.com/psf/black
