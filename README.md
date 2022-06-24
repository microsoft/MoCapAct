# MoCapAct
[![Code License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &nbsp;&nbsp;&nbsp;&nbsp; [![Dataset License](https://i.creativecommons.org/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

This is the codebase for the MoCapAct project, which contains all code to train the clip snippet experts, collect expert rollouts into a dataset, perform policy distillation, and perform RL on downstream tasks.
For more information on the project and to access the dataset, please visit the [project website](https://mhauskn.github.io/mocapact.github.io/).

## Setup
MoCapAct requires Python 3.7+.
We recommend that you use a virtual environment.
For example, using conda:
```bash
conda create -n MoCapAct pip
conda activate MoCapAct
```

To install the package, you can run `pip install` on the GitHub repo:
```bash
pip install git+https://github.com/nolanwagener/MoCapAct@main#egg=MoCapAct
```

Alternatively, to have an editable version, clone the repo and install the local copy:
```bash
git clone https://github.com/nolanwagener/MoCapAct.git
cd MoCapAct
pip install -e .
```

**Note:** All included policies (experts, multi-clip, etc.) will only work with MuJoCo 2.1.5 or earlier.
MuJoCo 2.2.0 uses analytic derivatives in place of finite-difference derivatives to determine actuator forces, which effectively changes the transition function of the simulator.
**Accordingly, MoCapAct installs MuJoCo 2.1.5 and `dm_control` 1.0.2.**
The rollout datasets were also generated under MuJoCo 2.1.5.
## Licenses
The code is licensed under the [MIT License](https://opensource.org/licenses/MIT).
The dataset is licensed under the [Creative Commons Attribution 4.0 International License (CC BY)](https://creativecommons.org/licenses/by/4.0/).
