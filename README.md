# MoCapAct

<p align="center">
  <img src="https://raw.githubusercontent.com/mhauskn/mocapact.github.io/master/assets/MoCapAct.gif" alt="montage" width="70%">
</p>

[![Code License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &nbsp;&nbsp;&nbsp;&nbsp; [<img src="https://cdla.dev/wp-content/uploads/sites/52/2017/10/cdla_logo.png" alt="Dataset License" width="150"/>](https://cdla.dev/permissive-2-0/)


This is the codebase for the MoCapAct project, which uses motion capture (MoCap) clips to learn low-level motor skills for the "CMU Humanoid" from the <tt>dm_control</tt> package.
This repo contains all code to:
- train the clip snippet experts,
- collect expert rollouts into a dataset,
- download our experts and rollouts from the command line,
- perform policy distillation,
- perform reinforcement learning on downstream tasks, and
- perform motion completion.

For more information on the project and to download the entire dataset, please visit the [project website](https://microsoft.github.io/MoCapAct/).

## Setup
MoCapAct requires Python 3.7+.
We recommend that you use a virtual environment.
For example, using conda:
```bash
conda create -n MoCapAct pip python==3.8
conda activate MoCapAct
```

To install the package, we recommend cloning the repo and installing the local copy:
```bash
git clone https://github.com/microsoft/MoCapAct.git
cd MoCapAct
pip install -e .
```

**<span style="color:red">Note:</span>** All included policies (experts, multi-clip, etc.) will only work with MuJoCo 2.1.5 or earlier.
MuJoCo 2.2.0 uses analytic derivatives in place of finite-difference derivatives to determine actuator forces, which effectively changes the transition function of the simulator.
**Accordingly, MoCapAct installs MuJoCo 2.1.5 and <tt>dm_control</tt> 1.0.2.**
The rollout datasets were also generated under MuJoCo 2.1.5.

## Downloading the Dataset
We provide a Python script to download portions of the dataset.
The script takes the following flags:
- `-t`: a type from `<experts | small_dataset | large_dataset>`,
- `-c`: a comma-separated list of clips (e.g., `CMU_001_01,CMU_002_01`) or a specific subset from <tt>dm_control</tt>'s [MoCap subsets](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/tasks/reference_pose/cmu_subsets.py) `<get_up | walk_tiny | run_jump_tiny | locomotion_small | all>`, and
- `-d`: a destination path.

For example:
```bash
python -m mocapact.download_dataset -t small_dataset -c CMU_001_01,CMU_002_01 -d ./data
```

## Examples

Below are Python commands we used for our paper.

<details>
<summary>Clip snippet experts</summary>

Training a clip snippet expert:
```bash
python -m mocapact.clip_expert.train \
  --clip_id [CLIP_ID] `# e.g., CMU_016_22` \
  --start_step [START_STEP] `# e.g., 0` \
  --max_steps [MAX_STEPS] `# e.g., 210 (can be larger than clip length)` \
  --n_workers [N_CPU] `# e.g., 8` \
  --log_root experts \
  $(cat cfg/clip_expert/train.txt)
```

Evaluating a clip snippet expert (numerical evaluation and visual evaluation):
```bash
python -m mocapact.clip_expert.evaluate \
  --policy_root [POLICY_ROOT] `# e.g., experts/CMU_016-22-0-82/0/eval_rsi/model` \
  --n_workers [N_CPU] `# e.g., 8` \
  --n_eval_episodes 1000 `# set to 0 to just run the visualizer` \
  $(cat cfg/clip_expert/evaluate.txt)
```

We can also load the experts in Python:
```python
from mocapact import observables
from mocapact.sb3 import utils
expert_path = "experts/CMU_016_22-0-82/0/eval_rsi/model" # example path
expert = utils.load_policy(expert_path, observables.TIME_INDEX_OBSERVABLES)

from mocapact.envs import tracking
from dm_control.locomotion.tasks.reference_pose import types
dataset = types.ClipCollection(ids=['CMU_016_22'])
env = tracking.MocapTrackingGymEnv(dataset)
obs, done = env.reset(), False
while not done:
    action, _ = expert.predict(obs, deterministic=True)
    obs, rew, done, _ = env.step(action)
    print(rew)
```
</details>

<details>
<summary>Creating rollout dataset</summary>

Rolling out a collection of experts and collecting into a dataset:
```bash
python -m mocapact.distillation.rollout_experts \
  --input_dirs [EXPERT_ROOT] `# e.g., experts` \
  --n_workers [N_CPU] `# e.g., 8` \
  --device [DEVICE] `# e.g., cuda` \
  --output_path dataset/file_name_ignored.hdf5 \
  $(cat cfg/rollout.txt)
```

This will result in a collection of HDF5 files (one per clip), which can be read and utilized in Python:
```python
import h5py
dset = h5py.File("dataset/CMU_016_22.hdf5", "r")
print("Expert actions from first rollout episode:")
print(dset["CMU_016_22-0-82/0/actions"][...])
```
</details>

<details>
<summary>Multi-clip policy</summary>

Training a multi-clip policy on the entire MoCapAct dataset:
```bash
source scripts/get_all_clips.sh [PATH_TO_DATASET]
python -m mocapact.distillation.train \
  --train_dataset_paths $train \
  --val_dataset_paths $val \
  --dataset_metrics_path $metrics \
  --extra_clips $clips \
  --output_root multi_clip/all \
  --gpus 0 `# indices of GPUs` \
  $(cat cfg/multi_clip/train.txt) \
  --model.config.embed_size 60 \
  --eval.n_workers [N_CPU] `# e.g., 16`
```

Training a multi-clip policy on the locomotion subset of the MoCapAct dataset:
```bash
source scripts/get_locomotion_clips.sh [PATH_TO_DATASET]
python -m mocapact.distillation.train \
  --train_dataset_paths $train \
  --dataset_metrics_path $metrics \
  --extra_clips $clips \
  --output_root multi_clip/locomotion \
  --gpus 0 `# indices of GPUs` \
  $(cat cfg/multi_clip/train.txt) \
  --model.config.embed_size 20 \
  --eval.n_workers [N_CPU] `# e.g., 16`
```

Evaluating a multi-clip policy on all the snippets within the MoCapAct dataset (numerical evaluation and visual evaluation):
```bash
source scripts/get_all_clips.sh [PATH_TO_DATASET]
python -m mocapact.distillation.evaluate \
  --policy_path [POLICY_PATH] `# e.g., multi_clip/all/eval/train_rsi/best_model.ckpt` \
  --clip_snippets $snippets \
  --n_workers [N_CPU] `# e.g., 8` \
  --device [DEVICE] `# e.g., cuda` \
  --n_eval_episodes 1000 `# set to 0 to just run the visualizer` \
  $(cat cfg/multi_clip/evaluate.txt)
```

The multi-clip policy can be loaded using PyTorch Lightning's functionality to interact with the environment:
```python
from mocapact.distillation import model
model_path = "multi_clip/all/eval/train_rsi/best_model.ckpt"
policy = model.NpmpPolicy.load_from_checkpoint(model_path, map_location="cpu")

from mocapact.envs import tracking
from dm_control.locomotion.tasks.reference_pose import cmu_subsets
dataset = cmu_subsets.ALL
ref_steps = (1, 2, 3, 4, 5)
env = tracking.MocapTrackingGymEnv(dataset, ref_steps)
obs, done = env.reset(), False
embed = policy.initial_state(deterministic=False)
while not done:
    action, embed = expert.predict(obs, state=embed, deterministic=False)
    obs, rew, done, _ = env.step(action)
    print(rew)
```
</details>

<details>
<summary>RL transfer tasks</summary>

Training a task policy (here, with a pre-defined low-level policy):
```bash
python -m mocapact.transfer.train \
  --log_root [LOG_ROOT] `# e.g., transfer/go_to_target` \
  $(cat cfg/transfer/train.txt) \
  $(cat cfg/transfer/go_to_target.txt) `# set to cfg/transfer/velocity_control.txt for velocity control` \
  $(cat cfg/transfer/with_low_level.txt) `# set to cfg/transfer/no_low_level.txt for no low-level policy`
```

Evaluating a task policy:
```bash
python -m mocapact.transfer.evaluate \
  --model_root [MODEL_ROOT] `# e.g., transfer/go_to_target/0/eval/model` \
  --task [TASK] `# e.g., mocapact/transfer/config.py:go_to_target or velocity_control`
```
</details>

<details>
<summary>Motion completion</summary>

Training a GPT policy on the entire MoCapAct dataset:
```bash
source scripts/get_all_clips.sh [PATH_TO_DATASET]
python -m mocapact.distillation.train \
  --train_dataset_paths $train \
  --val_dataset_paths $val \
  --dataset_metrics_path $metrics \
  --output_root motion_completion \
  $(cat cfg/motion_completion/train.txt)
```

Performing motion completion with a trained GPT policy:
```bash
python -m mocapact.distillation.motion_completion \
  --policy_path [POLICY_PATH] `# e.g., motion_completion/model/last.ckpt` \
  --expert_root [EXPERT_ROOT] `# e.g., experts` \
  --clip_snippet [CLIP_SNIPPET] `# e.g., CMU_016_22` \
  --n_workers [N_CPU] `# e.g., 8` \
  --device [DEVICE] `# e.g., cuda` \
  --n_eval_episodes 100 `# Set to 0 to just run the visualizer` \
  $(cat cfg/motion_completion/evaluate.txt)
```
To generate a prompt, we also input a path to the directory of snippet experts.
Alternatively, you can pass a path to a multi-clip policy through `--distillation_path`, though it will likely produce lower-quality prompts than the snippet experts.
</details>

## Licenses
The code is licensed under the [MIT License](https://opensource.org/licenses/MIT).
The dataset is licensed under the [CDLA Permissive v2 License](https://cdla.dev/permissive-2-0/).
