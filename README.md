# HumanLocoMocap code

This repo intends to track several experiments, tools and datasets for expert rollouts based on the CMU Mocap dataset and DM Control Humanoid Environment.

## Getting Started
```console
conda create -n humcontrol pip python=3.7
conda activate humcontrol
pip install -e .
```

## MuJoCo dependencies
Please follow instrunctions in [mujoco-py](https://github.com/openai/mujoco-py) and/or [dm_control](https://github.com/deepmind/dm_control)

- Useful env variables to have in .bashrc:

```console
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
export LIBGL_ALWAYS_INDIRECT=0
export DISPLAY=:0
```
