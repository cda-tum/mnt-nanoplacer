[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
# [![CodeCov](https://github.com/cda-tum/MQTPredictor/actions/workflows/coverage.yml/badge.svg)](https://github.com/cda-tum/MQTPredictor/actions/workflows/coverage.yml)
# [![Deploy to PyPI](https://github.com/cda-tum/MQTPredictor/actions/workflows/deploy.yml/badge.svg)](https://github.com/cda-tum/MQTPredictor/actions/workflows/deploy.yml)
# [![codecov](https://codecov.io/gh/cda-tum/MQTPredictor/branch/main/graph/badge.svg?token=ZL5js1wjrB)](https://codecov.io/gh/cda-tum/MQTPredictor)

# <p align="center">
# <picture>
#   <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mqtpredictor/main/img/mqt_light.png" width="60%">
#   <img src="https://raw.githubusercontent.com/cda-tum/mqtpredictor/main/img/mqt_dark.png" width="60%">
# </picture>
# </p>

# NanoPlaceR: Placement and Routing for Field-coupled Nanocomputing (FCN) based on Reinforcement Learning

NanoPlaceR is a tool for the physical design of FCN circuitry based on Reinforcement Learning.
It can generate layouts for logic networks up to ~200 gates, while requiring ~50% less area than the state-of-the-art heuristic approach.

## Reinforcement Learning Model (referred to as "RL")

Inspired by recent developments in the field of machine learning-aided design automation, this tool combines reinforcement learning with efficient path routing algorithms based on established algorithms such as A* search. 
Masked Proximal Policy Optimization (PPO) is used to learn the placement of logic elements, which is further accelerated by incorporating an action mask computed based on the netlist structure and the last partial placement, ensuring valid and compact solutions. 
To minimize the occurrence of unpromising partial placements, several checks constantly ensure the early termination of sub-par solutions. 
Furthermore, the routing of placed gates is incorporated directly into the placement step using established routing strategies.
The following figure outlines the methodology:

![](https://raw.githubusercontent.com/cda-tum/mqtpredictor/main/img/mdp.png)


# Usage of NanoPlaceR

First, the package must be installed:

```console
(venv) $ pip install mqt.predictor
```

```python
from mqt.predictor import ml, rl

compiled_qc_ML, compilation_info_ML = ml.qcompile("qasm_file_path_or_QuantumCircuit")
compiled_qc_RL, compilation_info_RL = rl.qcompile(
    "qasm_file_path_or_QuantumCircuit", opt_objective="fidelity"
)
```


# Repository Structure

```
.
├── benchmarks/                       # common benchmark sets
├── images/                           # generated layouts in .svg format are saved here
├── models/                           # ppo models
├── placement_envs/
│ ├── envs/
│   └── nano_placement_env/           # placement environment
├── ppo_masked/                       # ppo mask implementation adapted from sb3-contrib
├── tensorboard/                      # tensorboard log directory
```

# References

In case you are using NanoPlaceR in your work, we would be thankful if you referred to it by citing the following publication:

```bibtex
@INPROCEEDINGS{hofmann2023nanoplacer,
  author        = {S. Hofmann and M. Walter and L. Servadei and R. Wille},
  title         = {{Late Breaking Results From Hybrid Design Automation for Field-coupled Nanotechnologies}},
  booktitle     = {{Design Automation Conference (DAC)}},
  year          = {2023},
}
```
