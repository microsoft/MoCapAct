# MoCapAct

## Getting Started
We recommend that you use a virtual environment.
For example, using conda:
```bash
conda create -n mocapact pip
conda activate mocapact
```

To install the package, you can run `pip install` on the GitHub repo:
```bash
pip install git+https://github.com/nolanwagener/mocapact.git
```

Alternatively, to have an editable version, clone the repo and install the local copy:
```bash
git clone https://github.com/nolanwagener/mocapact.git
cd mocapact
pip install -e .
```

**Note:** All included policies only work with MuJoCo 2.1.5 or earlier.
MuJoCo 2.2.0 uses analytic derivatives in place of finite-difference derivatives to determine actuator forces, which effectively changes the transition function of the simulator.
Accordingly, MoCapAct installs MuJoCo 2.1.5 and `dm_control` 1.0.2.
