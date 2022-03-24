"""
Rolls out the expert on a batch of clips and saves the resulting trajectories.
Applies noise to the expert to allow better state-action coverage.

Created dataset has following hierarchy:
    |- clip1
       |- id
       |- early_termination
       |- loaded_metrics
         |- episode_lengths
         |- episode_returns
         |- norm_episode_lengths
         |- norm_episode_returns
       |- start_metrics
         |- episode_lengths
         |- episode_returns
         |- norm_episode_lengths
         |- norm_episode_returns
       |- rsi_metrics
         |- episode_lengths
         |- episode_returns
         |- norm_episode_lengths
         |- norm_episode_returns
       |- 0
         |- actions
         |- observations
         |- rewards
         |- advantages
         |- values
           ...
       |- 1
         |- actions
         |- observations
         |- rewards
         |- advantages
         |- values
           ...
       ...
    |- clip2
       |- id
       |- early_termination
       |- loaded_metrics
       |- start_metrics
       |- rsi_metrics
       |- episode_lengths
       |- episode_rewards
       |- 0
         |- actions
         |- observations
         |- rewards
         |- advantages
         |- values
           ...
       |- 1
         |- actions
         |- observations
         |- rewards
         |- advantages
         |- values
           ...
       ...
    ...
    |- ref_steps
    |- observable_indices
       |- observable1
       |- observable2
       ...
    |- stats
       |- obs_mean
       |- obs_var
       |- act_mean
       |- act_var
       |- count
    |- n_start_rollouts
    |- n_random_rollouts

For each clip, "loaded_metrics" corresponds to the metrics of the loaded policy as
given by the evaluations.npz file. "rsi_metrics" corresponds to the metrics of the loaded
policy when the policy is initialized at random time steps in the clip as given by
the rollout_experts.py script. The "loaded_metrics" and "rsi_metrics" can be compared to
verify that they are similar. The "start_metrics" corresponds to the metrics of the loaded
policy when the policy is initialized at the start of the snippet as given by the
rollout_experts.py script.
"""

import pickle
import zipfile
import glob
import os.path as osp
import h5py
import json
import numpy as np
import torch
from absl import app, flags, logging
from tqdm import tqdm
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from dm_control.locomotion.tasks.reference_pose import types
from humanoid_control import observables
from humanoid_control.sb3 import env_util
from humanoid_control.sb3 import features_extractor
from humanoid_control.sb3 import tracking
from humanoid_control.sb3 import wrappers

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

# Paths
flags.DEFINE_list("input_dirs", None, "List of directories to gather experts from")
flags.DEFINE_string("output_path", None, "Output file for the expert rollouts")
flags.DEFINE_bool("separate_clips", False, "Whether to save different clips to different files")

# Rollout flags
flags.DEFINE_integer("n_start_rollouts", 16, "Number of rollouts per expert at start of clip")
flags.DEFINE_integer("n_random_rollouts", 16, "Number of rollouts per expert from random point in clip")
flags.DEFINE_integer("n_workers", 8, "Number of parallel workers for rolling out")
flags.DEFINE_float("act_noise", 0.1, "Action noise in humanoid")
flags.DEFINE_float("termination_error_threshold", 0.3, "Error for cutting off expert rollout")
flags.DEFINE_list("ref_steps", [1, 2, 3, 4, 5], "Indices for reference observation")
flags.DEFINE_integer("min_steps", 10, "For random rollouts, latest point to start in the clip")

# Miscellaneous
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_string("device", "cpu", "Device to do rollouts on")

flags.mark_flag_as_required("input_dirs")
flags.mark_flag_as_required("output_path")

def get_expert_paths(input_dirs):
    """
    For each clip in the input directories, gets the path of the expert.
    """
    clips = set()
    expert_paths, expert_metrics = {}, {}
    for dir in input_dirs:
        experiment_paths = [osp.dirname(path) for path in glob.iglob(f"{dir}/**/flags.txt", recursive=True)]
        for path in experiment_paths:
            with open(osp.join(path, 'clip_info.json')) as f:
                clip_info = json.load(f)
            clip_id = clip_info['clip_id']
            start_step = clip_info['start_step']
            end_step = clip_info['end_step']
            expert_name = f"{clip_id}-{start_step}-{end_step}"
            if osp.exists(osp.join(path, 'eval_random/evaluations.npz')):
                try:
                    eval_npz = np.load(osp.join(path, 'eval_random/evaluations.npz'))
                except:
                    continue
                clips.add(clip_id)
                idx = eval_npz['results'].mean(1).argmax()
                ret = eval_npz['results'][idx].mean()
                if expert_name not in expert_paths or ret > expert_metrics[expert_name]['ep_return'].mean():
                    expert_paths[expert_name] = path
                    expert_metrics[expert_name] = dict(
                        ep_return=eval_npz['results'][idx],
                        ep_length=eval_npz['ep_lengths'][idx],
                        ep_norm_return=eval_npz['results_norm'][idx],
                        ep_norm_length=eval_npz['ep_lengths_norm'][idx]
                    )
    return expert_paths, expert_metrics, clips

def collect_rollouts(clip_path, always_init_at_clip_start):
    print(clip_path)
    # Make environment
    with open(osp.join(clip_path, 'clip_info.json')) as f:
        clip_info = json.load(f)
    clip_id = clip_info['clip_id']
    start_step = clip_info['start_step']
    end_step = clip_info['end_step']

    dataset = types.ClipCollection(
        ids=[clip_id],
        start_steps=[start_step],
        end_steps=[end_step]
    )
    task_kwargs = dict(
        reward_type='comic',
        min_steps=min(FLAGS.min_steps, end_step-start_step-len(FLAGS.ref_steps)) - 1,
        always_init_at_clip_start=always_init_at_clip_start,
        termination_error_threshold=FLAGS.termination_error_threshold
    )
    env_kwargs = dict(
        dataset=dataset,
        ref_steps=[int(s) for s in FLAGS.ref_steps],
        act_noise=FLAGS.act_noise,
        task_kwargs=task_kwargs
    )
    vec_env = env_util.make_vec_env(
        env_id=tracking.MocapTrackingGymEnv,
        n_envs=FLAGS.n_workers,
        seed=FLAGS.seed,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        vec_monitor_cls=wrappers.MocapTrackingVecMonitor
    )

    # Normalization statistics
    model_path = osp.join(clip_path, "eval_random/model")
    with open(osp.join(model_path, 'vecnormalize.pkl'), 'rb') as f:
        norm_env = pickle.load(f)
        obs_stats = norm_env.obs_rms

    # Set up model
    with zipfile.ZipFile(osp.join(model_path, 'best_model.zip')) as archive:
        json_string = archive.read("data").decode()
        json_dict = json.loads(json_string)
        policy_kwargs = {k: v for k, v in json_dict['policy_kwargs'].items() if not k.startswith(':')}
        if 'Tanh' in policy_kwargs['activation_fn']:
            policy_kwargs['activation_fn'] = torch.nn.Tanh
        elif 'ReLU' in policy_kwargs['activation_fn']:
            policy_kwargs['activation_fn'] = torch.nn.ReLU
        else:
            policy_kwargs['activation_fn'] = torch.nn.ELU
    policy_kwargs['features_extractor_class'] = features_extractor.CmuHumanoidFeaturesExtractor
    policy_kwargs['features_extractor_kwargs'] = dict(
        observable_keys=observables.TIME_INDEX_OBSERVABLES,
        obs_rms=obs_stats
    )
    model = PPO.load(
        osp.join(model_path, 'best_model.zip'),
        custom_objects=dict(policy_kwargs=policy_kwargs, learning_rate=0., clip_range=0.)
    )

    obs = vec_env.reset()

    ctr = 0
    all_observations, all_actions, all_rewards, all_values, all_advs = [], [], [], [], []
    all_normalized_returns, all_normalized_lengths, all_early_terminations = [], [], []
    curr_observations = [{k: [] for k in obs.keys()} for _ in range(FLAGS.n_workers)]
    curr_actions = [[] for _ in range(FLAGS.n_workers)]
    curr_rewards = [[] for _ in range(FLAGS.n_workers)]
    curr_values = [[] for _ in range(FLAGS.n_workers)]
    n_rollouts = FLAGS.n_start_rollouts if always_init_at_clip_start else FLAGS.n_random_rollouts
    while True:
        for i in range(FLAGS.n_workers):
            for k, v in obs.items():
                curr_observations[i][k].append(v[i])

        # Get value estimate
        obs_th, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            val_norm = model.policy.predict_values(obs_th).squeeze(1).cpu().numpy()
        val = norm_env.unnormalize_reward(val_norm)

        act, _ = model.policy.predict(obs, deterministic=True)
        obs, rews, dones, infos = vec_env.step(act)
        for i in range(FLAGS.n_workers):
            curr_actions[i].append(act[i])
            curr_rewards[i].append(rews[i])
            curr_values[i].append(val[i])
            if dones[i]:
                # Add terminal observation
                for k, v in infos[i]['terminal_observation'].items():
                    curr_observations[i][k].append(v)

                # Add episode to buffer
                curr_obs = curr_observations[i]
                all_observations.append({k: np.stack(v) for k, v in curr_obs.items()})
                all_actions.append(np.stack(curr_actions[i]))
                all_rewards.append(np.array(curr_rewards[i]))
                all_values.append(np.array(curr_values[i]))

                # GAE(lambda)
                rew, value = curr_rewards[i], curr_values[i]
                last_gae_lam, adv = 0, []
                for step in reversed(range(len(value))):
                    next_val = value[step+1] if step < len(value)-1 else 0.
                    delta = rew[step] + model.gamma*next_val - value[step]
                    last_gae_lam = delta + model.gamma*model.gae_lambda*last_gae_lam
                    adv.insert(0, last_gae_lam)
                all_advs.append(np.array(adv))

                all_normalized_returns.append(infos[i]['episode']['r_norm'])
                all_normalized_lengths.append(infos[i]['episode']['l_norm'])
                all_early_terminations.append(infos[i]['discount'] == 0.)

                # Reset corresponding episode buffer
                curr_observations[i] = {k: [] for k in obs.keys()}
                curr_actions[i] = []
                curr_rewards[i] = []
                curr_values[i] = []

                ctr += 1
                if ctr >= n_rollouts:
                    return dict(
                        observations=all_observations,
                        actions=all_actions,
                        rewards=all_rewards,
                        values=all_values,
                        advantages=all_advs,
                        normalized_returns=np.array(all_normalized_returns),
                        normalized_lengths=np.array(all_normalized_lengths),
                        early_terminations=np.array(all_early_terminations)
                        )

def create_dataset(expert_paths, expert_metrics, output_path):
    dset = h5py.File(output_path, 'w')

    # For computing dataset statistics
    obs_total, obs_sq_total, act_total, act_sq_total, count = 0, 0, 0, 0, 0

    pbar = tqdm(enumerate(expert_paths.items()))
    for id, (clip, path) in pbar:
        clip_grp = dset.create_group(clip)

        # Add ID
        id_dset = clip_grp.create_dataset("id", (), np.int64)
        id_dset[...] = id

        start_results = collect_rollouts(path, True)
        rsi_results = collect_rollouts(path, False)
        all_observations = start_results['observations'] + rsi_results['observations']
        all_actions = start_results['actions'] + rsi_results['actions']
        all_rewards = start_results['rewards'] + rsi_results['rewards']
        all_advantages = start_results['advantages'] + rsi_results['advantages']
        all_values = start_results['values'] + rsi_results['values']
        all_early_terminations = np.concatenate([start_results['early_terminations'], rsi_results['early_terminations']])

        for i in range(len(all_actions)): # iterate over episodes
            rollout_subgrp = clip_grp.create_group(str(i))
            observations, acts, rews, vals, advs = all_observations[i], all_actions[i], all_rewards[i], all_values[i], all_advantages[i]

            # Actions
            act_dset = rollout_subgrp.create_dataset("actions", acts.shape, np.float32)
            act_dset[...] = acts

            # Observations
            obs = np.concatenate(
                [o for k, o in observations.items() if k != 'walker/clip_id'],
                axis=1
            )
            obs_dset = rollout_subgrp.create_dataset("observations", obs.shape, np.float32)
            obs_dset[...] = obs

            # Rewards
            rew_dset = rollout_subgrp.create_dataset("rewards", rews.shape, np.float32)
            rew_dset[...] = rews

            # Values
            val_dset = rollout_subgrp.create_dataset("values", vals.shape, np.float32)
            val_dset[...] = vals

            # Advantages
            adv_dset = rollout_subgrp.create_dataset("advantages", advs.shape, np.float32)
            adv_dset[...] = advs

            # Accumulate statistics
            obs_total += obs.sum(0)
            obs_sq_total += (obs**2).sum(0)
            act_total += acts.sum(0)
            act_sq_total += (acts**2).sum(0)
            count += obs.shape[0]

        # Early termination dataset
        early_termination_dset = clip_grp.create_dataset("early_termination", all_early_terminations.shape, bool)
        early_termination_dset[...] = all_early_terminations

        # Metrics (episode returns, lengths, normalized return, normalized length)
        episode_returns = np.array([rews.sum() for rews in all_rewards])
        episode_lengths = np.array([acts.shape[0] for acts in all_actions])
        loaded_metrics_grp = clip_grp.create_group("loaded_metrics")
        start_metrics_grp = clip_grp.create_group("start_metrics")
        rsi_metrics_grp = clip_grp.create_group("rsi_metrics")
        for metrics_grp, rets, lens, norm_rets, norm_lens in zip(
            [loaded_metrics_grp, start_metrics_grp, rsi_metrics_grp],
            [expert_metrics[clip]['ep_return'], episode_returns[:FLAGS.n_start_rollouts], episode_returns[FLAGS.n_start_rollouts:]],
            [expert_metrics[clip]['ep_length'], episode_lengths[:FLAGS.n_start_rollouts], episode_lengths[FLAGS.n_start_rollouts:]],
            [expert_metrics[clip]['ep_norm_return'], start_results['normalized_returns'], rsi_results['normalized_returns']],
            [expert_metrics[clip]['ep_norm_length'], start_results['normalized_lengths'], rsi_results['normalized_lengths']]
        ):
            returns_dset = metrics_grp.create_dataset("episode_returns", rets.shape, rets.dtype)
            returns_dset[...] = rets
            lengths_dset = metrics_grp.create_dataset("episode_lengths", lens.shape, lens.dtype)
            lengths_dset[...] = lens
            norm_returns_dset = metrics_grp.create_dataset("norm_episode_returns", norm_rets.shape, norm_rets.dtype)
            norm_returns_dset[...] = norm_rets
            norm_lengths_dset = metrics_grp.create_dataset("norm_episode_lengths", norm_lens.shape, norm_lens.dtype)
            norm_lengths_dset[...] = norm_lens

    # Reference steps
    ref_steps_dset = dset.create_dataset("ref_steps", (len(FLAGS.ref_steps,)), np.int64)
    ref_steps_dset[...] = [int(s) for s in FLAGS.ref_steps]

    # Observable indices group
    idx = 0
    indices_grp = dset.create_group("observable_indices")
    for k in observations.keys():
        if k == 'walker/clip_id':
            continue
        o = observations[k]
        indices_dset = indices_grp.create_dataset(k, (o.shape[1],), np.int64)
        indices_dset[...] = np.arange(o.shape[1]) + idx
        idx += o.shape[1]

    # Dataset statistics
    obs_mean_dset = dset.create_dataset("stats/obs_mean", obs_total.shape, np.float32)
    obs_var_dset = dset.create_dataset("stats/obs_var", obs_total.shape, np.float32)
    act_mean_dset = dset.create_dataset("stats/act_mean", act_total.shape, np.float32)
    act_var_dset = dset.create_dataset("stats/act_var", act_total.shape, np.float32)
    count_dset = dset.create_dataset("stats/count", (), np.int64)
    obs_mean_dset[...] = obs_total / count
    obs_var_dset[...] = np.clip(obs_sq_total / count - obs_mean_dset[...]**2, 0., None)
    act_mean_dset[...] = act_total / count
    act_var_dset[...] = np.clip(act_sq_total / count - act_mean_dset[...]**2, 0., None)
    count_dset[...] = count

    # Number of rollouts
    random_rollouts_dset = dset.create_dataset("n_random_rollouts", (), np.int64)
    start_rollouts_dset = dset.create_dataset("n_start_rollouts", (), np.int64)
    random_rollouts_dset[...] = FLAGS.n_random_rollouts
    start_rollouts_dset[...] = FLAGS.n_start_rollouts

def main(_):
    # Create directory for hdf5 file(s)
    Path(osp.dirname(FLAGS.output_path)).mkdir(parents=True, exist_ok=True)
    paths, metrics, clips = get_expert_paths(FLAGS.input_dirs)

    if FLAGS.separate_clips:
        for clip in clips:
            print(clip)
            output_path = osp.join(osp.dirname(FLAGS.output_path), clip + '.hdf5')
            clip_paths = {k: v for k, v in paths.items() if k.startswith(clip)}
            create_dataset(clip_paths, metrics, output_path)
    else:
        create_dataset(paths, metrics, FLAGS.output_path)

if __name__ == "__main__":
    app.run(main)
