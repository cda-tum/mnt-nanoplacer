[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)

# NanoPlaceR: Placement and Routing for Field-coupled Nanocomputing (FCN) based on Reinforcement Learning

NanoPlaceR is a tool for the physical design of FCN circuitry based on Reinforcement Learning.
It can generate layouts for logic networks up to ~200 gates, while requiring ~50% less area than the state-of-the-art heuristic approach.

## Reinforcement Learning Model (referred to as "RL")

Inspired by recent developments in the field of machine learning-aided design automation, this tool combines reinforcement learning with efficient path routing algorithms based on established algorithms such as A* search. 
Masked Proximal Policy Optimization (PPO) is used to learn the placement of logic elements, which is further accelerated by incorporating an action mask computed based on the netlist structure and the last partial placement, ensuring valid and compact solutions. 
To minimize the occurrence of unpromising partial placements, several checks constantly ensure the early termination of sub-par solutions. 
Furthermore, the routing of placed gates is incorporated directly into the placement step using established routing strategies.
The following figure outlines the methodology:

![](images/lbr.png)


# Usage of NanoPlaceR

Currently, due to the Open-AI gym dependency, only python versions up to 3.10 are supported.

If you do not have a virtual environment set up, the following steps outline one possible way to do so.
First, install virtualenv:

```console
$ pip install virtualenv
```

Then create a new virtual environment in your your project folder and acitvate it:

```console
$ mkdir nano_placement
$ cd nano_placement
$ python -m venv venv
$ source env/bin/activate
```

NanoPlaceR heavily depends on fiction, whose python binding currently has to be installed from source (will be available as a python package in the future).
To install fiction and build it, run the following commands:
```console
(venv) $ git clone -b pyml --single-branch --recursive https://github.com/marcelwa/fiction.git
(venv) $ git submodule update --init --recursive
(venv) $ cd fiction
(venv) $ pip install .
(venv) $ cd ..
```

After succesfully installing pyfiction, clone this repository and install the dependencies:
```console
(venv) $ git clone https://github.com/cda-tum/NanoPlaceR.git
(venv) $ cd NanoPlaceR
(venv) $ pip install -r requirements.txt
```

To register the environment, install the package locally:
```console
(venv) $ pip install --e .
```

You can either change the parameters (e.g. logic function, clocking scheme, layout width etc.) in ``main.py``or simply use the tool in the command line.

```
$ main.py -h
usage: main.py [-h] [-b] [-f] [-c] [-t] [-lw] [-lh] [-ts] [-m] [-s] [-v]

Optional arguments:
  -h, --help         Show this help message and exit.
  -b, --benchmark    Benchmark set.
  -f, --function     Logic function to generate layout for.
  -c, --clocking_scheme  Underlying clocking scheme.
  -t, --technology          Underlying technology (QCA or SiDB).
  -lw, --layout_width   Layout width.
  -lh, --layout_height  Layout height.
  -ts, --time_steps     Number of time steps to train the RL agent.
  -m,  --mode           "INIT" for instantiating new model, "TRAIN" for using exisiting model.
  -s,  --save           Save trained model.
  -v,  --verbosity      
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
