"""
Rolls out the expert on a batch of clips and saves the resulting trajectories.
Applies noise to the expert to allow better state-action coverage.

Created dataset has following hierarchy:
    |- clip1
       |- id
       |- episode_lengths
       |- episode_rewards
       |- 0
         |- actions
         |- observations
         |- rewards
         |- returns
         |- values
           ...
       |- 1
         |- actions
         |- observations
         |- rewards
         |- returns
         |- values
           ...
       ...
    |- clip2
       |- id
       |- episode_lengths
       |- episode_rewards
       |- 0
         |- actions
         |- observations
         |- rewards
         |- returns
         |- values
           ...
       |- 1
         |- actions
         |- observations
         |- rewards
         |- returns
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
For each clip, "episode_lengths" and "episode_rewards" is an array of
episode lengths and rewards, respectively.
"""

import pickle
import zipfile
import glob
import os.path as osp
import h5py
import numpy as np
import scipy.signal
import torch
from absl import app, flags, logging
from tqdm import tqdm
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from dm_control.locomotion.tasks.reference_pose import types
from humanoid_control import observables
from humanoid_control import utils
from humanoid_control.sb3 import env_util
from humanoid_control.sb3 import features_extractor
from humanoid_control.sb3 import tracking

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
    expert_paths, expert_rews = {}, {}
    for dir in input_dirs:
        experiment_paths = [osp.dirname(path) for path in glob.iglob(f"{dir}/**/flags.txt", recursive=True)]
        for path in experiment_paths:
            flags = utils.load_absl_flags(osp.join(path, "flags.txt"))
            clip_id = flags["clip_id"]
            start_step = int(flags["start_step"])
            clip_length = utils.get_clip_length(flags["clip_id"])
            snippet_length = min(clip_length-start_step, int(flags['max_steps']))
            end_step = start_step + snippet_length
            expert_name = f"{clip_id}-{start_step}-{end_step}"
            if osp.exists(osp.join(path, 'eval_random/evaluations.npz')):
                try:
                    eval_npz = np.load(osp.join(path, 'eval_random/evaluations.npz'))
                except:
                    continue
                clips.add(clip_id)
                expert_rew = eval_npz['results'].mean(1).max()
                if expert_name not in expert_paths or expert_rew > expert_rews[expert_name]:
                    expert_paths[expert_name] = path
                    expert_rews[expert_name] = expert_rew
    return expert_paths, clips


def collect_rollouts(clip_path, always_init_at_clip_start):
    # Get experiment flags
    flags = utils.load_absl_flags(osp.join(clip_path, 'flags.txt'))

    # Make environment
    clip_length = utils.get_clip_length(flags['clip_id'])
    snippet_length = min(clip_length-int(flags['start_step']), int(flags['max_steps']))
    end_step = int(flags['start_step']) + snippet_length
    dataset = types.ClipCollection(
        ids=[flags['clip_id']],
        start_steps=[int(flags['start_step'])],
        end_steps=[end_step]
    )
    task_kwargs = dict(
        reward_type='comic',
        min_steps=min(FLAGS.min_steps, snippet_length-len(FLAGS.ref_steps))-1,
        always_init_at_clip_start=always_init_at_clip_start
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
        vec_env_cls=SubprocVecEnv
    )

    # Extract model zip file
    model_path = osp.join(clip_path, "eval_random/model")
    with zipfile.ZipFile(osp.join(model_path, 'best_model.zip')) as zip_ref:
        zip_ref.extractall(model_path)

    # Normalization statistics
    with open(osp.join(model_path, 'vecnormalize.pkl'), 'rb') as f:
        norm_env = pickle.load(f)

    features_extractor_class = lambda space: features_extractor.CmuHumanoidFeaturesExtractor(
        space,
        observable_keys=observables.TIME_INDEX_OBSERVABLES,
        obs_rms=norm_env.obs_rms
    )
    layer_sizes = int(flags['n_layers']) * [int(flags['layer_size'])]
    activation_fns = dict(relu=torch.nn.ReLU, tanh=torch.nn.Tanh, elu=torch.nn.ELU)
    activation_fn = activation_fns[flags['activation_fn']]
    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        net_arch=[dict(pi=layer_sizes, vf=layer_sizes)],
        activation_fn=activation_fn
    )
    model = PPO("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=0, device=FLAGS.device)
    params = torch.load(osp.join(model_path, 'policy.pth'), map_location=FLAGS.device)
    model.policy.load_state_dict(params)

    obs = vec_env.reset()

    ctr = 0
    all_observations, all_actions, all_rewards, all_returns, all_values = [], [], [], [], []
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
        obs, rews, dones, _ = vec_env.step(act)
        for i in range(FLAGS.n_workers):
            curr_actions[i].append(act[i])
            curr_rewards[i].append(rews[i])
            curr_values[i].append(val[i])
            if dones[i]:
                # Add episode to buffer
                curr_obs = curr_observations[i]
                all_observations.append({k: np.stack(v) for k, v in curr_obs.items()})
                all_actions.append(np.stack(curr_actions[i]))
                all_rewards.append(np.array(curr_rewards[i]))
                all_values.append(np.array(curr_values[i]))

                r = curr_rewards[i]
                rets = scipy.signal.lfilter([1], [1, float(-float(flags['gamma']))], r[::-1], axis=0)[::-1]
                all_returns.append(rets)

                # Reset corresponding episode buffer
                curr_observations[i] = {k: [] for k in obs.keys()}
                curr_actions[i] = []
                curr_rewards[i] = []
                curr_values[i] = []

                ctr += 1
                if ctr >= n_rollouts:
                    return all_observations, all_actions, all_rewards, all_returns, all_values

def create_dataset(expert_paths, output_path):
    dset = h5py.File(output_path, 'w')

    # For computing dataset statistics
    obs_total, obs_sq_total, act_total, act_sq_total, count = 0, 0, 0, 0, 0

    pbar = tqdm(enumerate(expert_paths.items()))
    for id, (clip, path) in pbar:
        clip_grp = dset.create_group(clip)

        # Add ID
        id_dset = clip_grp.create_dataset("id", (), np.int64)
        id_dset[...] = id

        start_observations, start_actions, start_rewards, start_returns, start_values = collect_rollouts(path, True)
        rand_observations, rand_actions, rand_rewards, rand_returns, rand_values = collect_rollouts(path, False)
        all_observations = start_observations + rand_observations
        all_actions = start_actions + rand_actions
        all_rewards = start_rewards + rand_rewards
        all_returns = start_returns + rand_returns
        all_values = start_values + rand_values

        for i in range(len(all_actions)): # iterate over episodes
            rollout_subgrp = clip_grp.create_group(str(i))
            observations, acts, rews, rets, vals = all_observations[i], all_actions[i], all_rewards[i], all_returns[i], all_values[i]

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

            # Returns
            ret_dset = rollout_subgrp.create_dataset("returns", rews.shape, np.float32)
            ret_dset[...] = rets

            # Values
            val_dset = rollout_subgrp.create_dataset("values", vals.shape, np.float32)
            val_dset[...] = vals

            # Accumulate statistics
            obs_total += obs.sum(0)
            obs_sq_total += (obs**2).sum(0)
            act_total += acts.sum(0)
            act_sq_total += (acts**2).sum(0)
            count += obs.shape[0]

        # Episode lengths dataset
        length_dset = clip_grp.create_dataset("episode_lengths", (len(all_actions),), np.int64)
        length_dset[...] = np.array([acts.shape[0] for acts in all_actions])

        # Episode rewards dataset
        episode_rewards = np.array([rews.sum() for rews in all_rewards])
        reward_dset = clip_grp.create_dataset("episode_rewards", episode_rewards.shape, episode_rewards.dtype)
        reward_dset[...] = episode_rewards

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
    paths, clips = get_expert_paths(FLAGS.input_dirs)

    if FLAGS.separate_clips:
        for clip in clips:
            print(clip)
            output_path = osp.join(osp.dirname(FLAGS.output_path), clip + '.hdf5')
            clip_paths = {k: v for k, v in paths.items() if k.startswith(clip)}
            create_dataset(clip_paths, output_path)
    else:
        create_dataset(paths, FLAGS.output_path)

if __name__ == "__main__":
    app.run(main)
