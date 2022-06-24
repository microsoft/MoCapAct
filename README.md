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

## Examples

Below are Python commands we used for our paper.

<details>
<summary>Clip snippet experts</summary>

Training a clip snippet expert:
```bash
python mocapact/clip_expert/train.py \
  --clip_id [CLIP_ID] `# e.g., CMU_016_22` \
  --start_step [START_STEP] `# e.g., 0` \
  --max_steps [MAX_STEPS] `# e.g., 210 (can be larger than clip length)` \
  --n_workers [N_CPU] `# e.g., 8` \
  --log_root experts \
  $(cat cfg/clip_expert/train.txt)
```

Evaluating a clip snippet expert (numerical evaluation and visual evaluation):
```bash
python mocapact/clip_expert/evaluate.py \
  --policy_root [POLICY_ROOT] `# e.g., experts/CMU_016-22-0-82/0/eval_rsi/model` \
  --n_workers [N_CPU] `# e.g., 8` \
  --n_eval_episodes 1000 `# set to 0 to just run the visualizer` \
  $(cat cfg/clip_expert/evaluate.txt)
```
</details>

<details>
<summary>Creating rollout dataset</summary>

Rolling out a collection of experts and collecting into a dataset:
```bash
python mocapact/distillation/rollout_experts.py \
  --input_dirs [EXPERT_ROOT] `# e.g., experts` \
  --n_workers [N_CPU] `# e.g., 8` \
  --device [DEVICE] `# e.g., cuda` \
  --output_path dataset/file_name_ignored.hdf5 \
  $(cat cfg/rollout.txt)
```
</details>

<details>
<summary>Multi-clip policy</summary>

Training a multi-clip policy on the entire MoCapAct dataset:
```bash
source scripts/get_all_clips.sh [PATH_TO_DATASET]
python mocapact/distillation/train.py \
  --train_dataset_paths $train \
  --val_dataset_paths $val \
  --dataset_metrics_path $metrics \
  --extra_clips $clips \
  --output_root multi_clip/all \
  --gpus 0 `# indices of GPUs` \
  --model.config.embed_size 60 \
  --eval.n_workers [N_CPU] `# e.g., 16` \
  $(cat cfg/multi_clip/train.txt)
```

Training a multi-clip policy on the locomotion subset of the MoCapAct dataset:
```bash
source scripts/get_locomotion_clips.sh [PATH_TO_DATASET]
python mocapact/distillation/train.py \
  --train_dataset_paths $train \
  --dataset_metrics_path $metrics \
  --extra_clips $clips \
  --output_root multi_clip/locomotion \
  --gpus 0 `# indices of GPUs` \
  --model.config.embed_size 20 \
  --eval.n_workers [N_CPU] `# e.g., 16` \
  $(cat cfg/multi_clip/train.txt)
```

Evaluating a multi-clip policy on all the snippets within the MoCapAct dataset (numerical evaluation and visual evaluation):
```bash
source scripts/get_all_clips.sh [PATH_TO_DATASET]
python mocapact/distillation/evaluate.py \
  --policy_path [POLICY_PATH] `# e.g., multi_clip/all/eval/train_rsi/best_model.ckpt` \
  --clip_snippets $snippets \
  --n_workers [N_CPU] `# e.g., 8` \
  --device [DEVICE] `# e.g., cuda` \
  --n_eval_episodes 1000 `# set to 0 to just run the visualizer` \
  $(cat cfg/multi_clip/evaluate.txt)
```
</details>

<details>
<summary>RL transfer tasks</summary>

Training a task policy (here, with a pre-defined low-level policy):
```bash
python mocapact/transfer/train.py \
  --log_root [LOG_ROOT] `# e.g., transfer/go_to_target` \
  $(cat cfg/transfer/train.txt) \
  $(cat cfg/transfer/go_to_target.txt) `# set to cfg/transfer/velocity_control.txt for velocity control` \
  $(cat cfg/transfer/with_low_level.txt) `# set to cfg/transfer/no_low_level.txt for no low-level policy`
```

Evaluating a task policy:
```bash
python mocapact/transfer/evaluate.py \
  --model_root [MODEL_ROOT] `# e.g., transfer/go_to_target/0/eval/model` \
  --task [TASK] `# e.g., mocapact/transfer/config.py:go_to_target or velocity_control`
```
</details>

<details>
<summary>Motion completion</summary>

Training a GPT policy on the entire MoCapAct dataset:
```bash
source scripts/get_all_clips.sh [PATH_TO_DATASET]
python mocapact/distillation/train.py \
  --train_dataset_paths $train \
  --val_dataset_paths $val \
  --dataset_metrics_path $metrics \
  --output_root motion_completion \
  $(cat cfg/motion_completion/train.txt)
```

Performing motion completion with a trained GPT policy:
```bash
python mocapact/distillation/motion_completion.py \
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
The dataset is licensed under the [Creative Commons Attribution 4.0 International License (CC BY)](https://creativecommons.org/licenses/by/4.0/).
