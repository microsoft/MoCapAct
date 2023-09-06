# MoCapAct

<p align="center">
  <img src="https://raw.githubusercontent.com/mhauskn/mocapact.github.io/master/assets/MoCapAct.gif" alt="montage" width="70%">
</p>

[![Code License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &nbsp;&nbsp;&nbsp;&nbsp; [<img src="https://cdla.dev/wp-content/uploads/sites/52/2017/10/cdla_logo.png" alt="Dataset License" width="150"/>](https://cdla.dev/permissive-2-0/)

<b>Paper: [MoCapAct: A Multi-Task Dataset for Simulated Humanoid Control](https://arxiv.org/abs/2208.07363)</b>

This is the codebase for the MoCapAct project, which uses motion capture (MoCap) clips to learn low-level motor skills for the "CMU Humanoid" from the <tt>dm_control</tt> package.
This repo contains all code to:
- train the clip snippet experts,
- collect expert rollouts into a dataset,
- download our experts and rollouts from the command line,
- perform policy distillation,
- perform reinforcement learning on downstream tasks, and
- perform motion completion.

For more information on the project and to download the entire dataset, please visit the [project website](https://microsoft.github.io/MoCapAct/).

For users interested in development, we recommend reading the following documentation on <tt>dm_control</tt>:
- [The <tt>dm_control</tt> whitepaper](https://arxiv.org/abs/2006.12983)
- [The <tt>dm_control</tt> README](https://github.com/deepmind/dm_control/blob/main/README.md)
- [The README for <tt>dm_control</tt>'s locomotion task library](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/README.md)

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

## Dataset
The MoCapAct dataset consists of clip experts trained on the MoCap snippets and the rollouts from those experts.

### Downloading the Dataset
We recommend using Azure Storage Explorer to download the dataset, which we detail below.

<details>
<summary>Azure Storage Explorer</summary>

First, [install Azure Storage Explorer](https://snapcraft.io/install/storage-explorer/ubuntu). This can be done with snap:
```bash
sudo snap install storage-explorer
```

Then, open Azure Storage Explorer.
Click the "Open Connect Dialog" button.
<p align="center">
  <img src="https://raw.githubusercontent.com/mhauskn/mocapact.github.io/master/assets/download/azure_storage_explorer.png" alt="montage" width="70%">
</p>

In the "Connect to Azure Storage" window that opens, click the "Blob container or directory" option.
<p align="center">
  <img src="https://raw.githubusercontent.com/mhauskn/mocapact.github.io/master/assets/download/select_resource.png" alt="montage" width="60%">
</p>

Select "Shared access signature URL (SAS)" and click "Next".
<p align="center">
  <img src="https://raw.githubusercontent.com/mhauskn/mocapact.github.io/master/assets/download/sas.png" alt="montage" width="60%">
</p>

In an Internet browser, go to the [MoCapAct page on Microsoft Research](https://www.microsoft.com/en-us/research/publication/mocapact-a-multi-task-dataset-for-simulated-humanoid-control/) and copy the link given by the "Download Dataset" button.

<p align="center">
  <img src="https://raw.githubusercontent.com/mhauskn/mocapact.github.io/master/assets/download/website.png" alt="montage" width="40%">
</p>

In Azure Storage Explorer, paste the copied URL into the "Blob container or directory SAS URL" box.
Put a desired name in the "Display name" box (e.g., "MoCapAct").
Click "Next" and then "Connect".
<p align="center">
  <img src="https://raw.githubusercontent.com/mhauskn/mocapact.github.io/master/assets/download/connection_info.png" alt="montage" width="60%">
</p>

If successful, you should see the dataset files in the window.
<p align="center">
  <img src="https://raw.githubusercontent.com/mhauskn/mocapact.github.io/master/assets/download/dataset.png" alt="montage" width="70%">
</p>
</details>

Alternatively, the AzCopy program can be used to download files from the command line.

<details>
<summary>AzCopy</summary>

Download the AzCopy tar file from the [product page](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy) ([direct link to x86 Linux tar file](https://aka.ms/downloadazcopy-v10-linux)).
Unzip the file and then go to the AzCopy directory:
```bash
tar -xf /path/to/azcopy.tar.gz
cd /path/to/azcopy
```

Next, use the `azcopy` command to download the desired file from the MoCapAct blob.

```bash
file=README.md # use the empty string "" to download everything in the blob

# Source URL can be copied from "Download Dataset" in Microsoft Research page:
# https://www.microsoft.com/en-us/research/publication/mocapact-a-multi-task-dataset-for-simulated-humanoid-control/
# Note the presence of `${file}$` in the middle of `src` and backslashes `\` before `?`, `=`, and `&`.
src=https://msropendataset01.blob.core.windows.net/motioncapturewithactionsmocapact-u20220731/${file}\?sp\=rl\&st\=2023-08-22T17:20:43Z\&se\=2026-07-02T01:20:43Z\&spr\=https\&sv\=2022-11-02\&sr\=c\&sig\=L9f3Y8jAz3SCoAM5U5g9uCs0pmTI40rDLh2ZGC7OxE8%3D

./azcopy copy $src /path/to/dst --recursive
```

The value for `file` can be any of the files in the MoCapAct blob, shown below.
<p align="center">
  <img src="https://raw.githubusercontent.com/mhauskn/mocapact.github.io/master/assets/download/files.png" alt="montage" width="50%">
</p>

</details>

Finally, we provide an AzCopy-based Python script to download portions of the dataset, though this method has much slower download speeds than the other two methods.

<details>
<summary>Python script</summary>

The script takes the following flags:
- `-t`: a type from `<experts | small_dataset | large_dataset>`,
- `-c`: a comma-separated list of clips (e.g., `CMU_001_01,CMU_009_12`) or a specific subset from <tt>dm_control</tt>'s [MoCap subsets](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/tasks/reference_pose/cmu_subsets.py) `<get_up | walk_tiny | run_jump_tiny | locomotion_small | all>`, and
- `-d`: a destination path.

For example:
```bash
python -m mocapact.download_dataset -t experts -c CMU_009_12 -d ./data
python -m mocapact.download_dataset -t small_dataset -c CMU_001_01,CMU_009_12 -d ./data
```
</details>

### Description
<details>
<summary>Clip snippet experts</summary>
We signify a clip snippet expert by the snippet it is tracking.
Taking <tt>CMU_009_12-165-363</tt> as an example expert, the file hierarchy for the snippet expert is:

```
CMU_009_12-165-363
├── clip_info.json         # Contains clip ID, start step, and end step
└── eval_rsi/model
    ├── best_model.zip     # Contains policy parameters and hyperparameters
    └── vecnormalize.pkl   # Used to get normalizer for observation and reward
```

The expert policy can be loaded using our repository:
```python
from mocapact import observables
from mocapact.sb3 import utils
expert_path = "data/experts/CMU_009_12-165-363/eval_rsi/model"
expert = utils.load_policy(expert_path, observables.TIME_INDEX_OBSERVABLES)

from mocapact.envs import tracking
from dm_control.locomotion.tasks.reference_pose import types
dataset = types.ClipCollection(ids=['CMU_009_12'], start_steps=[165], end_steps=[363])
env = tracking.MocapTrackingGymEnv(dataset)
obs, done = env.reset(), False
while not done:
    action, _ = expert.predict(obs, deterministic=True)
    obs, rew, done, _ = env.step(action)
    print(rew)
```
</details>

<details>
<summary>Expert rollouts</summary>
The expert rollouts consist of a collection of HDF5 files, one per clip.
An HDF5 file contains expert rollouts for each constituent snippet as well as miscellaneous information and statistics.
To facilitate efficient loading of the observations, we concatenate all the proprioceptive observations (joint angles, joint velocities, actuator activations, etc.) from an episode into a single numerical array and provide indices for the constituent observations in the <tt>observable_indices</tt> group.

Taking <tt>CMU_009_12.hdf5</tt> (which contains three snippets) as an example, we have the following HDF5 hierarchy:
```
CMU_009_12.hdf5
├── n_rsi_rollouts                # R, number of rollouts from random time steps in snippet
├── n_start_rollouts              # S, number of rollouts from start of snippet
├── ref_steps                     # Indices of MoCap reference relative to current time step. Here, (1, 2, 3, 4, 5).
├── observable_indices
│   └── walker
│       ├── actuator_activation   # (0, 1, ..., 54, 55)
│       ├── appendages_pos        # (56, 57, ..., 69, 70)
│       ├── body_height           # (71)
│       ├── ...
│       └── world_zaxis           # (2865, 2866, 2867)
│
├── stats                         # Statistics computed over the entire dataset
│   ├── act_mean                  # Mean of the experts' sampled actions
│   ├── act_var                   # Variance of the experts' sampled actions
│   ├── mean_act_mean             # Mean of the experts' mean actions
│   ├── mean_act_var              # Variance of the experts' mean actions
│   ├── proprio_mean              # Mean of the proprioceptive observations
│   ├── proprio_var               # Variance of the proprioceptive observations
│   └── count                     # Number of observations in dataset
│
├── CMU_009_12-0-198              # Rollouts for the snippet CMU_009_12-0-198
├── CMU_009_12-165-363            # Rollouts for the snippet CMU_009_12-165-363
└── CMU_009_12-330-529            # Rollouts for the snippet CMU_009_12-330-529
```

Each snippet group contains $R+S$ snippets.
The first $S$ episodes correspond to episodes initialized from the start of the snippet and the last $R$ episodes to episodes initialized at random points in the snippet.
We now uncollapse the <tt>CMU_009_12-165-363</tt> group within the HDF5 file to reveal the rollout structure:
```
CMU_009_12-165-363
├── early_termination                  # (R+S)-boolean array indicating which episodes terminated early
├── rsi_metrics                        # Metrics for episodes that initialize at random points in snippet
│   ├── episode_returns                # R-array of episode returns
│   ├── episode_lengths                # R-array of episode lengths
│   ├── norm_episode_returns           # R-array of normalized episode rewards
│   └── norm_episode_lengths           # R-array of normalized episode lengths
├── start_metrics                      # Metrics for episodes that initialize at start in snippet
│
├── 0                                  # First episode, of length T
│   ├── observations
│   │   ├── proprioceptive             # (T+1)-array of proprioceptive observations
│   │   ├── walker/body_camera         # (T+1)-array of images from body camera **(not included)**
│   │   └── walker/egocentric_camera   # (T+1)-array of images from egocentric camera **(not included)**
│   ├── actions                        # T-array of sampled actions executed in environment
│   ├── mean_actions                   # T-array of corresponding mean actions
│   ├── rewards                        # T-array of rewards from environment
│   ├── values                         # T-array computed using the policy's value network
│   └── advantages                     # T-array computed using generalized advantage estimation
│
├── 1                                  # Second episode
├── ...
└── R+S-1                              # (R+S)th episode
```
To keep the dataset size manageable, we do *not* include image observations in the dataset.
The camera images can be logged by providing the flags `--log_all_proprios --log_cameras` to the `mocapact/distillation/rollout_experts.py` script.

The HDF5 rollouts can be read and utilized in Python:
```python
import h5py
dset = h5py.File("data/small_dataset/CMU_009_12.hdf5", "r")
print("Expert actions from first rollout episode of second snippet:")
print(dset["CMU_009_12-165-363/0/actions"][...])
```

We provide a "large" dataset where $R = S = 100$ (with size 600 GB) and a "small" dataset where $R = S = 10$ (with size 50 GB).
</details>

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
  $(cat cfg/multi_clip/rwr.txt) `# rwr can be replaced with awr, cwr, or bc` \
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
  $(cat cfg/multi_clip/rwr.txt) `# rwr can be replaced with awr, cwr, or bc` \
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

## Dataset Interface
We provide two datasets in this repo.
The [`ExpertDataset`](https://github.com/microsoft/MoCapAct/blob/main/mocapact/distillation/dataset.py) is used to perform imitation learning, e.g., to train a multi-clip tracking policy or a GPT policy for motion completion.
The [`D4RLDataset`](https://github.com/microsoft/MoCapAct/blob/main/mocapact/offline_rl/d4rl_dataset.py) is used for offline reinforcement learning. 
For small enough instantiations of the datasets that fit into memory, the user can use `D4RLDataset.get_in_memory_rollouts()` to load a batch of transitions into memory.
For instantiations that do not fit into memory (e.g., the entire MoCapAct dataset), the user can use the dataset as a PyTorch `Dataset` by using an iterator over the transitions obtained by using `__getitem__()`.

## Future Plans
We are happy to work with the community to fix bugs and expand functionality of MoCapAct.
This will include incorporating pull requests and allowing for MoCap clips to be added in the form of HDF5 files.

## Citation
If you reference or use MoCapAct in your research, please cite:

```bibtex
@inproceedings{wagener2022mocapact,
  title={{MoCapAct}: A Multi-Task Dataset for Simulated Humanoid Control},
  author={Wagener, Nolan and Kolobov, Andrey and Frujeri, Felipe Vieira and Loynd, Ricky and Cheng, Ching-An and Hausknecht, Matthew},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={35418--35431},
  year={2022}
}
```

## Licenses
The code is licensed under the [MIT License](https://opensource.org/licenses/MIT).
The dataset is licensed under the [CDLA Permissive v2 License](https://cdla.dev/permissive-2-0/).
