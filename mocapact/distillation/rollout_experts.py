"""
Rolls out the experts on a batch of clips and saves the resulting trajectories.
Applies noise to the expert to allow better state-action coverage.

Created dataset has following hierarchy:
    |- clip1
       |- early_termination
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
         |- mean_actions
         |- observations
           |- proprioceptive
         |- rewards
         |- advantages
         |- values
           ...
       |- 1
         |- actions
         |- mean_actions
         |- observations
         |- rewards
         |- advantages
         |- values
           ...
       ...
    |- clip2
       |- early_termination
       |- start_metrics
       |- rsi_metrics
       |- episode_lengths
       |- episode_rewards
       |- 0
         |- actions
         |- mean_actions
         |- observations
         |- rewards
         |- advantages
         |- values
           ...
       |- 1
         |- actions
         |- mean_actions
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
       |- proprio_mean
       |- proprio_var
       |- act_mean
       |- act_var
       |- mean_act_mean
       |- mean_act_var
       |- count
    |- n_start_rollouts
    |- n_rsi_rollouts

For each clip, "rsi_metrics" corresponds to the metrics of the loaded
policy when the policy is initialized at random time steps in the clip as given by
the rollout_experts.py script. The "start_metrics" corresponds to the metrics of the loaded
policy when the policy is initialized at the start of the snippet as given by the
rollout_experts.py script.
"""

import pickle
import glob
import os.path as osp
import h5py
import json
import numpy as np
import torch
from absl import app, flags, logging
from tqdm import tqdm
from pathlib import Path

from stable_baselines3.common.vec_env import SubprocVecEnv

from dm_control.locomotion.tasks.reference_pose import types
from mocapact import observables
from mocapact.envs import env_util
from mocapact.envs import tracking
from mocapact.sb3 import utils
from mocapact.sb3 import wrappers

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

# Paths
flags.DEFINE_list("input_dirs", None, "List of directories to gather experts from")
flags.DEFINE_string("output_path", None, "Output file for the expert rollouts")
flags.DEFINE_bool("separate_clips", False, "Whether to save different clips to different files")

# Rollout flags
flags.DEFINE_integer("n_start_rollouts", 16, "Number of rollouts per expert at start of clip")
flags.DEFINE_integer("n_rsi_rollouts", 16, "Number of rollouts per expert from random point in clip")
flags.DEFINE_integer("n_workers", 8, "Number of parallel workers for rolling out")
flags.DEFINE_float("act_noise", 0.1, "Action noise in humanoid")
flags.DEFINE_float("termination_error_threshold", 0.3, "Error for cutting off expert rollout")
flags.DEFINE_list("ref_steps", [1, 2, 3, 4, 5], "Indices for reference observation")
flags.DEFINE_integer("min_steps", 10, "For random rollouts, latest point to start in the clip")
flags.DEFINE_bool("log_all_proprios", False, "Log all the low-level observations from the humanoid")
flags.DEFINE_bool("log_cameras", False, "Log all the camera images from the humanoid")

# Miscellaneous
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_string("device", "cpu", "Device to do rollouts on")

flags.mark_flag_as_required("input_dirs")
flags.mark_flag_as_required("output_path")

def mean_and_variance(total, squared_total, count):
    mean = total / count
    variance = squared_total / count - mean**2
    return mean, variance

def get_expert_paths(input_dirs):
    """
    For each clip in the input directories, gets the path of the expert.
    """
    clips = set()
    expert_paths = dict()
    for dir in input_dirs:
        experiment_paths = [osp.dirname(path) for path in glob.iglob(f"{dir}/**/clip_info.json", recursive=True)]
        for path in experiment_paths:
            if not osp.exists(osp.join(path, 'eval_rsi/model/best_model.zip')):
                continue

            with open(osp.join(path, 'clip_info.json')) as f:
                clip_info = json.load(f)
            clip_id = clip_info['clip_id']
            start_step = clip_info['start_step']
            end_step = clip_info['end_step']
            expert_name = f"{clip_id}-{start_step}-{end_step}"

            clips.add(clip_id)
            expert_paths[expert_name] = path
    return expert_paths, clips

def collect_rollouts(clip_path, always_init_at_clip_start):
    print(clip_path)
    # Make environment
    with open(osp.join(clip_path, 'clip_info.json')) as f:
        clip_info = json.load(f)
    clip_id = clip_info['clip_id']
    start_step = clip_info['start_step']
    end_step = clip_info['end_step']

    ref_steps = [int(s) for s in FLAGS.ref_steps]
    dataset = types.ClipCollection(
        ids=[clip_id],
        start_steps=[start_step],
        end_steps=[end_step]
    )
    task_kwargs = dict(
        reward_type='comic',
        min_steps=min(FLAGS.min_steps, end_step-start_step-max(ref_steps)) - 1,
        always_init_at_clip_start=always_init_at_clip_start,
        termination_error_threshold=FLAGS.termination_error_threshold
    )
    env_kwargs = dict(
        dataset=dataset,
        ref_steps=ref_steps,
        act_noise=0.,
        enable_all_proprios=FLAGS.log_all_proprios,
        enable_cameras=FLAGS.log_cameras,
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

    model = utils.load_policy(
        osp.join(clip_path, 'eval_rsi/model'),
        observables.TIME_INDEX_OBSERVABLES,
        device=FLAGS.device,
        seed=FLAGS.seed
    )
    model.policy.observation_space = vec_env.observation_space # env's obs space may differ from policy's
    model.policy.log_std.detach().fill_(np.log(FLAGS.act_noise))
    # Normalization statistics
    with open(osp.join(clip_path, 'eval_rsi/model/vecnormalize.pkl'), 'rb') as f:
        norm_env = pickle.load(f)
    obs = vec_env.reset()

    ctr = 0
    all_observations, all_actions, all_mean_actions, all_rewards, all_values, all_advs = [], [], [], [], [], []
    all_normalized_returns, all_normalized_lengths, all_early_terminations = [], [], []
    curr_observations = [{k: [] for k in obs.keys()} for _ in range(FLAGS.n_workers)]
    curr_actions = [[] for _ in range(FLAGS.n_workers)]
    curr_mean_actions = [[] for _ in range(FLAGS.n_workers)]
    curr_rewards = [[] for _ in range(FLAGS.n_workers)]
    curr_values = [[] for _ in range(FLAGS.n_workers)]
    n_rollouts = FLAGS.n_start_rollouts if always_init_at_clip_start else FLAGS.n_rsi_rollouts
    while True:
        for i in range(FLAGS.n_workers):
            for k, v in obs.items():
                curr_observations[i][k].append(v[i])

        # Get value estimate
        obs_th, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            val_norm = model.policy.predict_values(obs_th).squeeze(1).cpu().numpy()
        val = norm_env.unnormalize_reward(val_norm)

        act, _ = model.policy.predict(obs, deterministic=False)
        mean_act, _ = model.policy.predict(obs, deterministic=True)
        obs, rews, dones, infos = vec_env.step(act)

        for i in range(FLAGS.n_workers):
            curr_actions[i].append(act[i])
            curr_mean_actions[i].append(mean_act[i])
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
                all_mean_actions.append(np.stack(curr_mean_actions[i]))
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
                curr_mean_actions[i] = []
                curr_rewards[i] = []
                curr_values[i] = []

                ctr += 1
                if ctr >= n_rollouts:
                    return dict(
                        observations=all_observations,
                        actions=all_actions,
                        mean_actions=all_mean_actions,
                        rewards=all_rewards,
                        values=all_values,
                        advantages=all_advs,
                        normalized_returns=np.array(all_normalized_returns),
                        normalized_lengths=np.array(all_normalized_lengths),
                        early_terminations=np.array(all_early_terminations)
                        )

def create_dataset(expert_paths, output_path):
    dset = h5py.File(output_path, 'w')

    # For computing dataset statistics
    proprio_total, sq_proprio_total = 0, 0
    act_total, sq_act_total = 0, 0
    mean_act_total, sq_mean_act_total = 0, 0
    count = 0
    snippet_returns, values, advantages = dict(), dict(), dict()

    pbar = tqdm(expert_paths.items())
    for clip, path in pbar:
        clip_grp = dset.create_group(clip)
        values[clip], advantages[clip] = [], []

        start_results = collect_rollouts(path, True)
        rsi_results = collect_rollouts(path, False)
        all_observations = start_results['observations'] + rsi_results['observations']
        all_actions = start_results['actions'] + rsi_results['actions']
        all_mean_actions = start_results['mean_actions'] + rsi_results['mean_actions']
        all_rewards = start_results['rewards'] + rsi_results['rewards']
        all_advantages = start_results['advantages'] + rsi_results['advantages']
        all_values = start_results['values'] + rsi_results['values']
        all_early_terminations = np.concatenate([start_results['early_terminations'], rsi_results['early_terminations']])

        for i in range(len(all_actions)): # iterate over episodes
            rollout_subgrp = clip_grp.create_group(str(i))
            observations, acts, mean_acts = all_observations[i], all_actions[i], all_mean_actions[i]
            rews, vals, advs = all_rewards[i], all_values[i], all_advantages[i]

            # Actions
            act_dset = rollout_subgrp.create_dataset("actions", acts.shape, np.float32)
            act_dset[...] = acts

            # Mean actions
            mean_act_dset = rollout_subgrp.create_dataset("mean_actions", mean_acts.shape, np.float32)
            mean_act_dset[...] = mean_acts

            # Proprioceptive observations
            proprio = np.concatenate(
                [o for o in observations.values() if o.dtype in [np.float32, np.float64]],
                axis=1
            )
            obs_subgrp = rollout_subgrp.create_group("observations")
            proprio_dset = obs_subgrp.create_dataset("proprioceptive", proprio.shape, np.float32)
            proprio_dset[...] = proprio

            # Log images, if included
            for k, o in observations.items():
                if o.dtype == np.uint8:
                    image_dset = obs_subgrp.create_dataset(k, o.shape, np.uint8)
                    image_dset[...] = o

            # Rewards
            rew_dset = rollout_subgrp.create_dataset("rewards", rews.shape, np.float32)
            rew_dset[...] = rews

            # Values
            val_dset = rollout_subgrp.create_dataset("values", vals.shape, np.float32)
            val_dset[...] = vals
            values[clip].append(vals)

            # Advantages
            adv_dset = rollout_subgrp.create_dataset("advantages", advs.shape, np.float32)
            adv_dset[...] = advs
            advantages[clip].append(advs)

            # Accumulate statistics
            proprio_total += proprio.sum(0)
            sq_proprio_total += (proprio**2).sum(0)
            act_total += acts.sum(0)
            sq_act_total += (acts**2).sum(0)
            mean_act_total += mean_acts.sum(0)
            sq_mean_act_total += (mean_acts**2).sum(0)
            count += proprio.shape[0]

        # Early termination dataset
        early_termination_dset = clip_grp.create_dataset("early_termination", all_early_terminations.shape, bool)
        early_termination_dset[...] = all_early_terminations

        # Metrics (episode returns, lengths, normalized return, normalized length)
        episode_returns = np.array([rews.sum() for rews in all_rewards])
        episode_lengths = np.array([acts.shape[0] for acts in all_actions])
        start_metrics_grp = clip_grp.create_group("start_metrics")
        rsi_metrics_grp = clip_grp.create_group("rsi_metrics")
        for metrics_grp, rets, lens, norm_rets, norm_lens in zip(
            [start_metrics_grp, rsi_metrics_grp],
            [episode_returns[:FLAGS.n_start_rollouts], episode_returns[FLAGS.n_start_rollouts:]],
            [episode_lengths[:FLAGS.n_start_rollouts], episode_lengths[FLAGS.n_start_rollouts:]],
            [start_results['normalized_returns'], rsi_results['normalized_returns']],
            [start_results['normalized_lengths'], rsi_results['normalized_lengths']]
        ):
            returns_dset = metrics_grp.create_dataset("episode_returns", rets.shape, rets.dtype)
            returns_dset[...] = rets
            lengths_dset = metrics_grp.create_dataset("episode_lengths", lens.shape, lens.dtype)
            lengths_dset[...] = lens
            norm_returns_dset = metrics_grp.create_dataset("norm_episode_returns", norm_rets.shape, norm_rets.dtype)
            norm_returns_dset[...] = norm_rets
            norm_lengths_dset = metrics_grp.create_dataset("norm_episode_lengths", norm_lens.shape, norm_lens.dtype)
            norm_lengths_dset[...] = norm_lens

        snippet_returns[clip] = np.mean([start_results['normalized_returns'], rsi_results['normalized_returns']])
        values[clip] = np.concatenate(values[clip])
        advantages[clip] = np.concatenate(advantages[clip])

    # Reference steps
    ref_steps_dset = dset.create_dataset("ref_steps", (len(FLAGS.ref_steps,)), np.int64)
    ref_steps_dset[...] = [int(s) for s in FLAGS.ref_steps]

    # Observable indices group
    idx = 0
    indices_grp = dset.create_group("observable_indices")
    for k, o in observations.items():
        if o.dtype in [np.float32, np.float64]:
            indices_dset = indices_grp.create_dataset(k, o.shape[1:], np.int64)
            indices_dset[...] = np.arange(o.shape[1]) + idx
            idx += o.shape[1]

    # Dataset statistics
    proprio_mean_dset = dset.create_dataset("stats/proprio_mean", proprio_total.shape, np.float32)
    proprio_var_dset = dset.create_dataset("stats/proprio_var", proprio_total.shape, np.float32)
    act_mean_dset = dset.create_dataset("stats/act_mean", act_total.shape, np.float32)
    act_var_dset = dset.create_dataset("stats/act_var", act_total.shape, np.float32)
    mean_act_mean_dset = dset.create_dataset("stats/mean_act_mean", mean_act_total.shape, np.float32)
    mean_act_var_dset = dset.create_dataset("stats/mean_act_var", mean_act_total.shape, np.float32)
    count_dset = dset.create_dataset("stats/count", (), np.int64)
    proprio_mean_dset[...] = proprio_total / count
    proprio_var_dset[...] = np.clip(sq_proprio_total / count - proprio_mean_dset[...]**2, 0., None)
    act_mean_dset[...] = act_total / count
    act_var_dset[...] = np.clip(sq_act_total / count - act_mean_dset[...]**2, 0., None)
    mean_act_mean_dset[...] = mean_act_total / count
    mean_act_var_dset[...] = np.clip(sq_mean_act_total / count - mean_act_mean_dset[...]**2, 0., None)
    count_dset[...] = count

    # Number of rollouts
    rsi_rollouts_dset = dset.create_dataset("n_rsi_rollouts", (), np.int64)
    start_rollouts_dset = dset.create_dataset("n_start_rollouts", (), np.int64)
    rsi_rollouts_dset[...] = FLAGS.n_rsi_rollouts
    start_rollouts_dset[...] = FLAGS.n_start_rollouts

    # Metrics to collect over all the datasets
    return (proprio_total, sq_proprio_total, act_total, sq_act_total, mean_act_total, sq_mean_act_total, count,
            snippet_returns, values, advantages)

def main(_):
    # Create directory for hdf5 file(s)
    Path(osp.dirname(FLAGS.output_path)).mkdir(parents=True, exist_ok=True)
    paths, clips = get_expert_paths(FLAGS.input_dirs)

    if FLAGS.separate_clips:
        proprio_total, sq_proprio_total, act_total, sq_act_total, mean_act_total, sq_mean_act_total, count = 0, 0, 0, 0, 0, 0, 0
        snippet_returns, values, advantages = dict(), dict(), dict()
        for clip in clips:
            print(clip)
            output_path = osp.join(osp.dirname(FLAGS.output_path), clip + '.hdf5')
            clip_paths = {k: v for k, v in paths.items() if k.startswith(clip)}
            output = create_dataset(clip_paths, output_path)
            (prop_tot, sq_prop_tot, a_tot, sq_a_tot, mean_a_tot, sq_mean_a_tot, cnt, snip_ret, val, adv) = output
            proprio_total += prop_tot
            sq_proprio_total += sq_prop_tot
            act_total += a_tot
            sq_act_total += sq_a_tot
            mean_act_total += mean_a_tot
            sq_mean_act_total += sq_mean_a_tot
            count += cnt
            snippet_returns.update(snip_ret)
            values.update(val)
            advantages.update(adv)
    else:
        output = create_dataset(paths, FLAGS.output_path)
        (proprio_total, sq_proprio_total, act_total, sq_act_total, mean_act_total, sq_mean_act_total, count,
         snippet_returns, values, advantages) = output

    proprio_mean, proprio_var = mean_and_variance(proprio_total, sq_proprio_total, count)
    act_mean, act_var = mean_and_variance(act_total, sq_act_total, count)
    mean_act_mean, mean_act_var = mean_and_variance(mean_act_total, sq_mean_act_total, count)
    snippet_returns = dict(sorted(snippet_returns.items()))
    values = dict(sorted(values.items()))
    advantages = dict(sorted(advantages.items()))
    np.savez(
        osp.join(osp.dirname(FLAGS.output_path), 'dataset_metrics.npz'),
        proprio_mean=proprio_mean,
        proprio_var=proprio_var,
        act_mean=act_mean,
        act_var=act_var,
        mean_act_mean=mean_act_mean,
        mean_act_var=mean_act_var,
        count=count,
        snippet_returns=snippet_returns,
        values=values,
        advantages=advantages
    )

if __name__ == "__main__":
    app.run(main)
