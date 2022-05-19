import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gym

from absl import flags, app
import ml_collections
from ml_collections.config_flags import config_flags
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecVideoRecorder, DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from humanoid_control.sb3 import wrappers

from dm_control.locomotion.tasks.reference_pose import types
from humanoid_control import observables
from humanoid_control.envs import env_util
from humanoid_control.envs import tracking
from humanoid_control.offline_rl.d4rl_dataset import D4RLDataset
from humanoid_control.offline_rl.continuous_bcq.BCQ import BCQ

from aml_experiments import logger


FLAGS = flags.FLAGS

# Experiment flags
flags.DEFINE_bool("use_tensorboard", True, "Whether or not to log tensorboard events.")

# Path flags
flags.DEFINE_string("output_root", None, "Output directory to save the model and logs")
flags.DEFINE_string("dataset_local_path", None, "Path to the dataset")
flags.DEFINE_list("dataset_url", None, "URL to download the dataset")
flags.DEFINE_list("train_dataset_files", None, "HDF5 names for the training")
flags.DEFINE_list("val_dataset_files", None, "HDF5 names for the validation, if desired")
flags.DEFINE_bool("do_validation_loop", False, "Whether to run PyTorch Lightning's loop over the validation set")
flags.DEFINE_bool("randomly_load_hdf5", False, "Whether to randomize the order of hdf5 files before loading")
flags.DEFINE_integer("save_every_n_minutes", 60, "How often to save latest model")
flags.DEFINE_list("clip_ids", None, "List of clips to consider. By default, every clip.")

# Training hyperparameters
flags.DEFINE_float("termination_error_threshold", 0.3, "Error for cutting off rollout")
flags.DEFINE_integer("max_clip_steps", 256, "Maximum steps from start step")
flags.DEFINE_integer("min_clip_steps", 1, "Minimum steps in a rollout")
flags.DEFINE_string("env_name", "MocapTrackingGymEnv", "Environment from which the dataset was generated")
flags.DEFINE_integer("seed", 0, "Sets Gym, PyTorch and Numpy seeds")
flags.DEFINE_string("buffer_name", "Robust", "Prefix for file names")
flags.DEFINE_float("eval_freq", 5e3, "How often to evaluate the policy, in terms of timesteps")
flags.DEFINE_float("max_timesteps", 1e6, "Max time steps to train for (this defines buffer size)")

flags.DEFINE_integer("batch_size", 64, "Batch size used during training")
flags.DEFINE_float("discount", 0.99, "Discount factor")
flags.DEFINE_float("tau", 0.005, "Target network update rate")
flags.DEFINE_float("lmbda", 0.75, "Weighting for clipped double Q-learning in BCQ")
flags.DEFINE_float("phi", 0.05, "Max perturbation hyper-parameter for BCQ")
flags.DEFINE_float("temperature", None, "AWR temperature")

# Model hyperparameters
config_file = "humanoid_control/offline_rl/config.py"
config_flags.DEFINE_config_file("model", f"{config_file}:mlp_time_index", "Model architecture")
flags.DEFINE_string("gpus", "-1", "GPU configuration (https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#select-gpu-devices)")
flags.DEFINE_bool("normalize_obs", False, "Whether to normalize the input observation")
flags.DEFINE_string("load_path", None, "Load path to warm-start")
flags.DEFINE_bool("record_video", False, "Whether to record video for evaluation")

# Rollout evaluation hyperparameters
eval_config = ml_collections.ConfigDict()
eval_config.freq = int(1e5)
eval_config.n_episodes = 2500
eval_config.act_noise = 0.
eval_config.min_steps = 15
eval_config.termination_error_threshold = 0.3
eval_config.n_workers = 1
eval_config.seed = 0
flags.DEFINE_multi_enum("eval_mode", [], ["train_start", "train_random", "val_start", "val_random"], "What dataset and initialization to do evaluation on")
config_flags.DEFINE_config_dict("eval", eval_config)

flags.mark_flag_as_required("output_root")
flags.mark_flag_as_required("dataset_local_path")
flags.mark_flag_as_required("train_dataset_files")


def make_env(
    seed=0,
    clip_ids=[],
    start_steps=[0],
    end_steps=[0],
    min_steps=10,
    training=True,
    act_noise=0.,
    always_init_at_clip_start=False,
    record_video=False,
    video_folder=None,
    n_workers=4,
    termination_error_threshold=float('inf'),
    gamma=0.95,
    normalize_obs=True,
    normalize_rew=True
):
    dataset = types.ClipCollection(
        ids=clip_ids,
        start_steps=start_steps,
        end_steps=end_steps
    )
    task_kwargs = dict(
        reward_type='comic',
        min_steps=min_steps - 1,
        always_init_at_clip_start=always_init_at_clip_start,
        termination_error_threshold=termination_error_threshold
    )
    env_kwargs = dict(
        dataset=dataset,
        ref_steps=(0,),
        act_noise=act_noise,
        task_kwargs=task_kwargs
    )
    env = env_util.make_vec_env(
        env_id=tracking.MocapTrackingGymEnv,
        n_envs=n_workers,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv,  # SubprocVecEnv,
        vec_monitor_cls=wrappers.MocapTrackingVecMonitor
    )
    if record_video and video_folder:
        env = VecVideoRecorder(env, video_folder,
                               record_video_trigger=lambda x: x >= 0,
                               video_length=float('inf'))

    env = VecNormalize(env, training=training, gamma=gamma,
                       norm_obs=normalize_obs,
                       norm_reward=normalize_rew,
                       norm_obs_keys=observables.MULTI_CLIP_OBSERVABLES_SANS_ID)

    return env


def eval_policy(
    policy,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.
    We additionally return the average normalized reward to allow for better per-clip
    comparison.
    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.
    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_norm_rewards = []
    episode_norm_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    frames = []
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        observations = np.concatenate(list({k: observations[k] for k in observables.TIME_INDEX_OBSERVABLES}.values()), axis=-1)
        actions = policy.select_action(observations)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            episode_norm_rewards.append(info["episode"]["r_norm"])
                            episode_norm_lengths.append(info["episode"]["l_norm"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        if render:
            # Custom rendering using the physics object of the first vec env.
            frame = env.get_attr('physics')[0].render(width=640, height=480, camera_id=3).copy()
            frames.append(frame)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_norm_rewards, episode_norm_lengths, frames
    return mean_reward, std_reward


def main(_):
    output_dir = os.path.join(FLAGS.output_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)

    if FLAGS.use_tensorboard:
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))

    # Log some stuff (but only in process 0)
    if os.getenv("LOCAL_RANK", "0") == "0":
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(FLAGS.flags_into_string())
        with open(os.path.join(output_dir, 'flags.txt'), 'w') as f:
            f.write(FLAGS.flags_into_string())
        with open(os.path.join(output_dir, 'model_constructor.txt'), 'w') as f:
            f.write(FLAGS.model.constructor)

    pl.seed_everything(FLAGS.seed, workers=True)

    # If desired, randomize order of dataset paths
    if FLAGS.randomly_load_hdf5:
        print(FLAGS.train_dataset_files)
        random.shuffle(FLAGS.train_dataset_files)
        print(FLAGS.train_dataset_files)
        if FLAGS.val_dataset_files is not None:
            random.shuffle(FLAGS.val_dataset_files)

    # Make OfflineRL dataset
    if hasattr(FLAGS.model.config, 'seq_steps'):
        seq_steps = FLAGS.model.config.seq_steps
    elif hasattr(FLAGS.model.config, 'block_size'):
        seq_steps = FLAGS.model.config.block_size
    else:
        seq_steps = 1

    train_dataset = D4RLDataset(
        observables=observables.TIME_INDEX_OBSERVABLES,
        dataset_local_path=FLAGS.dataset_local_path,
        h5py_fnames=FLAGS.train_dataset_files,
        clip_ids=FLAGS.clip_ids,
        min_seq_steps=seq_steps,
        max_seq_steps=seq_steps,
        normalize_obs=False,  # FLAGS.normalize_obs,
        temperature=FLAGS.temperature,
    )

    if FLAGS.val_dataset_files is not None:
        val_dataset = D4RLDataset(
            observables=observables.TIME_INDEX_OBSERVABLES,
            dataset_local_path=FLAGS.dataset_local_path,
            h5py_fnames=FLAGS.val_dataset_files,
            clip_ids=FLAGS.clip_ids,
            min_seq_steps=seq_steps,
            max_seq_steps=seq_steps,
            normalize_obs=FLAGS.normalize_obs,
            temperature=FLAGS.temperature,
        )

    if FLAGS.normalize_obs:
        obs_rms = {}
        for obs_key, obs_indices in train_dataset.observable_indices.items():
            rms = RunningMeanStd(shape=obs_indices.shape)
            rms.mean = train_dataset.obs_mean[obs_indices]
            rms.var = train_dataset.obs_var[obs_indices]
            obs_rms[obs_key] = rms
    else:
        obs_rms = None

    obs_dim = train_dataset.observation_space.shape[0]
    action_dim = train_dataset.action_space.shape[0]
    max_action = float(train_dataset.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For saving files
    setting = f"{FLAGS.env_name}_{FLAGS.seed}"
    clip_ids, start_steps, end_steps = zip(*[clip_id.split('-') for clip_id in train_dataset.clip_snippets_flat])
    start_steps = [int(s) for s in start_steps]
    end_steps = [int(s) for s in end_steps]
    eval_env = make_env(
        seed=FLAGS.seed,
        clip_ids=clip_ids,
        start_steps=start_steps,
        end_steps=end_steps,
        training=True,
        act_noise=0.,
        always_init_at_clip_start=False,
        record_video=FLAGS.record_video,
        video_folder=output_dir,
        termination_error_threshold=FLAGS.termination_error_threshold
    )

    # Initialize policy
    policy = BCQ(obs_dim, action_dim, max_action, device, FLAGS.discount, FLAGS.tau, FLAGS.lmbda, FLAGS.phi)

    training_iters = 0
    replay_buffer = DataLoader(train_dataset, FLAGS.batch_size, shuffle=True)
    while training_iters < FLAGS.max_timesteps:
        print('Train step:', training_iters)

        KL_loss, vae_loss, critic_loss, actor_loss = policy.train(replay_buffer, iterations=int(FLAGS.eval_freq), batch_size=FLAGS.batch_size)
        tb_writer.add_scalar('KL_loss', KL_loss.tolist(), training_iters)
        tb_writer.add_scalar('vae_loss', vae_loss.tolist(), training_iters)
        tb_writer.add_scalar('critic_loss', critic_loss.tolist(), training_iters)
        tb_writer.add_scalar('actor_loss', actor_loss.tolist(), training_iters)

        logger.log_metric('KL_loss', KL_loss.tolist())
        logger.log_metric('vae_loss', vae_loss.tolist())
        logger.log_metric('critic_loss', critic_loss.tolist())
        logger.log_metric('actor_loss', actor_loss.tolist())

        eval_avg_reward, eval_std_reward = eval_policy(
            policy,
            eval_env,
            # render=True
        )

        tb_writer.add_scalar('eval_avg_reward', eval_avg_reward, training_iters)
        tb_writer.add_scalar('eval_std_reward', eval_std_reward, training_iters)

        logger.log_metric('eval_avg_reward', eval_avg_reward)
        logger.log_metric('eval_std_reward', eval_std_reward)

        training_iters += FLAGS.eval_freq
        print(f"Training iterations: {training_iters}")

    if FLAGS.use_tensorboard:
        tb_writer.close()


if __name__ == '__main__':
    app.run(main)
