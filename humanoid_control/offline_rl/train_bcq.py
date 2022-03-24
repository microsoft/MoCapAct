import os
from datetime import datetime, timedelta
from pathlib import Path
import random

from absl import flags, app
import ml_collections
from ml_collections.config_flags import config_flags
import numpy as np
import torch

from stable_baselines3.common.running_mean_std import RunningMeanStd

from humanoid_control import observables
from humanoid_control.offline_rl.d4rl_dataset import D4RLDataset
from humanoid_control.offline_rl.continuous_bcq import utils as bcq_utils, BCQ

FLAGS = flags.FLAGS

# Path flags
flags.DEFINE_string("output_root", None, "Output directory to save the model and logs")
flags.DEFINE_list("dataset_url", None, "URL to download the dataset")
flags.DEFINE_list("train_dataset_file_names", None, "HDF5 names for the training")
flags.DEFINE_list("val_dataset_file_names", None, "HDF5 names for the validation, if desired")
flags.DEFINE_bool("do_validation_loop", False, "Whether to run PyTorch Lightning's loop over the validation set")
flags.DEFINE_bool("randomly_load_hdf5", False, "Whether to randomize the order of hdf5 files before loading")
flags.DEFINE_integer("save_every_n_minutes", 60, "How often to save latest model")
flags.DEFINE_list("clip_ids", None, "List of clips to consider. By default, every clip.")
flags.DEFINE_float("eval_freq", 5e3, "How often to evaluate the policy")
flags.DEFINE_string("buffer_name", "Robust", "Prefix for file names")
flags.DEFINE_string("env_name", "Humanoid-Control", "Environment from which the dataset was generated")
flags.DEFINE_integer("seed", 0, "Sets Gym, PyTorch and Numpy seeds")
flags.DEFINE_integer("max_timesteps", 1e6, "Max time steps to train for (this defines buffer size)")

# Training hyperparameters
flags.DEFINE_integer("batch_size", 64, "Batch size used during training")
flags.DEFINE_float("discount", 0.99, "Discount factor")
flags.DEFINE_float("tau", 0.005, "Target network update rate")
flags.DEFINE_float("lmbda", 0.75, "Weighting for clipped double Q-learning in BCQ")
flags.DEFINE_float("phi", 0.05, "Max perturbation hyper-parameter for BCQ")

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
eval_config.n_workers = 8
eval_config.seed = 0
flags.DEFINE_multi_enum("eval_mode", [], ["train_start", "train_random", "val_start", "val_random"], "What dataset and initialization to do evaluation on")
config_flags.DEFINE_config_dict("eval", eval_config)

flags.mark_flag_as_required("output_root")
flags.mark_flag_as_required("train_dataset_paths")

def eval_policy(policy, env_name, seed, eval_episodes=10):
    pass

def main(_):
    output_dir = os.path.join(FLAGS.output_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

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
        print(FLAGS.train_dataset_file_names)
        random.shuffle(FLAGS.train_dataset_file_names)
        print(FLAGS.train_dataset_file_names)
        if FLAGS.val_dataset_file_names is not None:
            random.shuffle(FLAGS.val_dataset_file_names)

    # Make supervision dataset
    if hasattr(FLAGS.model.config, 'seq_steps'):
        seq_steps = FLAGS.model.config.seq_steps
    elif hasattr(FLAGS.model.config, 'block_size'):
        seq_steps = FLAGS.model.config.block_size
    else:
        seq_steps = 1

    train_dataset = D4RLDataset(
        FLAGS.train_dataset_file_names,
        observables.TIME_INDEX_OBSERVABLES,
        FLAGS.clip_ids,
        min_seq_steps=seq_steps,
        max_seq_steps=seq_steps,
        normalize_obs=False,  # FLAGS.normalize_obs,
        preload=FLAGS.preload_dataset,
        temperature=FLAGS.temperature,
        concat_observables=False
    )

    if FLAGS.val_dataset_file_names is not None:
        val_dataset = D4RLDataset(
            FLAGS.val_dataset_file_names,
            FLAGS.model.config.observables,
            FLAGS.clip_ids,
            min_seq_steps=seq_steps,
            max_seq_steps=seq_steps,
            normalize_obs=FLAGS.normalize_obs,
            preload=FLAGS.preload_dataset
        )

    if FLAGS.normalize_obs:
        obs_rms = {}
        observable_indices = train_dataset.observable_indices['walker']
        for k in observable_indices:
            rms = RunningMeanStd(shape=(len(observable_indices[k]),))
            rms.mean = train_dataset.obs_mean[observable_indices[k]]
            rms.var = train_dataset.obs_var[observable_indices[k]]
            obs_rms[f"walker/{k}"] = rms
    else:
        obs_rms = None

    obs, act, rew, weight, terminal, timeout = train_dataset[0]
    obs_dim = obs.shape[0]
    action_dim = act.shape[0]
    max_action = float(train_dataset.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For saving files
    setting = f"{FLAGS.env_name}_{FLAGS.seed}"
    buffer_name = f"{FLAGS.buffer_name}_{setting}"

    # Initialize policy
    policy = BCQ(obs_dim, action_dim, max_action, device, FLAGS.discount, FLAGS.tau, FLAGS.lmbda, FLAGS.phi)

    # Load buffer
    replay_buffer = bcq_utils.ReplayBuffer(obs_dim, action_dim, device)
    N = FLAGS.max_timesteps
    print('Loading buffer!')

    for i in range(1, N - 1):
        done = terminal or timeout
        new_obs, new_act, new_rew, new_weight, new_terminal, new_timeout = train_dataset[i + 1]
        replay_buffer.add(obs, act, new_obs, rew, done)
        obs, act, rew, weight, terminal, timeout = new_obs, new_act, new_rew, new_weight, new_terminal, new_timeout

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < FLAGS.max_timesteps:
        print('Train step:', training_iters)
        pol_vals = policy.train(replay_buffer, iterations=int(FLAGS.eval_freq), batch_size=FLAGS.batch_size)

        evaluations.append(eval_policy(policy, FLAGS.env_name, FLAGS.seed))
        np.save(os.path.join(output_dir, f"BCQ_{setting}"), evaluations)

        training_iters += FLAGS.eval_freq
        print(f"Training iterations: {training_iters}")


if __name__ == '__main__':
    app.run(main)
