# MoCapAct
[![Code License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Dataset License](https://licensebuttons.net/l/by-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-sa/4.0/)

This is the codebase for the MoCapAct project, which contains all code to train the clip snippet experts, collect expert rollouts into a dataset, perform policy distillation, and perform RL on downstream tasks.
For more information and to access the dataset, please visit the [project website](https://mhauskn.github.io/mocapact.github.io/).

## Setup
We recommend that you use a virtual environment.
For example, using conda:
```bash
conda create -n mocapact pip
conda activate mocapact
```

To install the package, you can run `pip install` on the GitHub repo:
```bash
pip install git+https://github.com/nolanwagener/MoCapAct.git
```

Alternatively, to have an editable version, clone the repo and install the local copy:
```bash
git clone https://github.com/nolanwagener/MoCapAct.git
cd mocapact
pip install -e .
```

**Note:** All included policies only work with MuJoCo 2.1.5 or earlier.
MuJoCo 2.2.0 uses analytic derivatives in place of finite-difference derivatives to determine actuator forces, which effectively changes the transition function of the simulator.
Accordingly, MoCapAct installs MuJoCo 2.1.5 and `dm_control` 1.0.2.

## Licenses
The dataset is licensed under the [Creative Commons Attribution-ShareAlike 4.0 License (CC BY-SA)](https://creativecommons.org/licenses/by-sa/4.0/).
The code is licensed under the [MIT License](https://opensource.org/licenses/MIT).
