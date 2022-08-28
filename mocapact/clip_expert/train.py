"""
Main script for training the clip experts.
"""
import os
import os.path as osp
import json
from pathlib import Path
import pickle
import zipfile
import numpy as np
import torch
from absl import app, flags
import ml_collections
from ml_collections import config_flags
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import get_device

from mocapact import observables
from mocapact import utils
from mocapact.envs import env_util
from mocapact.envs import tracking
from mocapact.sb3 import callbacks as sb3_callbacks
from mocapact.sb3 import features_extractor
from mocapact.sb3 import utils as sb3_utils
from mocapact.sb3 import wrappers
from mocapact.clip_expert import callbacks
from mocapact.clip_expert import utils as clip_expert_utils

FLAGS = flags.FLAGS
flags.DEFINE_string("clip_id", None, "Name of reference clip. See cmu_subsets.py")
flags.DEFINE_string("log_root", None, "Directory where logs are stored")
flags.DEFINE_integer("start_step", 0, "Start step in clip")
flags.DEFINE_integer("max_steps", 256, "Maximum steps from start step")
flags.DEFINE_string("mocap_path", None, "Path to MoCap HDF5. If None, uses dm_control's MoCap data")
flags.DEFINE_float("act_noise", 0.1, "Action noise to apply")
flags.DEFINE_integer("min_steps", 1, "Minimum steps in a rollout")
flags.DEFINE_float("termination_error_threshold", 0.3, "Error for cutting off rollout")

# Training hyperparameters
flags.DEFINE_integer("total_timesteps", int(1e8), "Total number of env steps")
flags.DEFINE_integer("n_workers", 16, "Number of workers used for rollouts")
flags.DEFINE_integer("n_steps", 4096, "Number of steps per policy optimization")
flags.DEFINE_integer("n_epochs", 10, "Number of epochs when optimizing the PPO loss")
flags.DEFINE_integer("batch_size", 256, "Minibatch size for PPO")
flags.DEFINE_float("clip_range", 0.2, "Clipping parameter for PPO")
flags.DEFINE_float("target_kl", float('inf'), "Limits KL divergence in updating policy")
flags.DEFINE_float("max_grad_norm", 1., "Clipping value for gradient norm")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda parameter")
lr_config = ml_collections.ConfigDict()
lr_config.start_val = 1e-4  # Initial step size
lr_config.decay = 1.73 # Decay rate for learning rate
config_flags.DEFINE_config_dict("learning_rate", lr_config)

# Network hyperparameters
flags.DEFINE_integer("n_layers", 3, "Number of hidden layers")
flags.DEFINE_integer("layer_size", 1024, "Number of neurons in hidden layer")
flags.DEFINE_enum("activation_fn", "torch.nn.Tanh", ["torch.nn.ReLU", "torch.nn.Tanh", "torch.nn.ELU"], "Activation for the hidden layers")

# Evaluation hyperparameters
eval_config = ml_collections.ConfigDict()
eval_config.seed = 0                                    # RNG seed for evaluation
eval_config.min_steps = 10                              # Minimum steps in an evaluation episodes
eval_config.freq = int(1e5)                             # After how many total environment steps to evaluate policy
eval_config.n_rsi_episodes = 32                         # Number of episodes to evaluate the policy from random initial states
eval_config.rsi_eval_act_noise = 0.1                    # Action noise to apply for random initial states
eval_config.n_start_episodes = 32                       # Number of episodes to evaluate the policy from the start of snippet
eval_config.start_eval_act_noise = 0.1                  # Action noise to apply for start of snippet
eval_config.early_stop = ml_collections.ConfigDict()
eval_config.early_stop.ep_length_threshold = 1.         # Episode length threshold for early stopping
eval_config.early_stop.min_reward_delta = 0.            # Minimum change in normalized reward to qualify as improvement
eval_config.early_stop.patience = 10                    # Number of queries with no improvement after which training is stopped
config_flags.DEFINE_config_dict("eval", eval_config)

# Misc. hyperparameters
flags.DEFINE_float("gamma", 0.95, "Discount factor")
flags.DEFINE_integer("seed", 0, "RNG seed for training")
flags.DEFINE_enum("device", "auto", ["auto", "cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"], "Device to do training on")
flags.DEFINE_bool("do_logging", True, "Whether to log")
flags.DEFINE_bool("record_video", False, "Whether to record video for evaluation")
flags.DEFINE_bool("include_timestamp", True, "Whether to include timestamp in log directory")
flags.DEFINE_string("warm_start_path", None, "If desired, path to warm-start parameters")

flags.mark_flag_as_required('clip_id')
flags.mark_flag_as_required('log_root')

def make_env(seed=0, start_step=0, end_step=0, min_steps=10, training=True,
             act_noise=0., always_init_at_clip_start=False,
             termination_error_threshold=float('inf')):
    env_kwargs = clip_expert_utils.make_env_kwargs(
        FLAGS.clip_id,
        mocap_path=FLAGS.mocap_path,
        start_step=start_step,
        end_step=end_step,
        min_steps=min_steps,
        always_init_at_clip_start=always_init_at_clip_start,
        termination_error_threshold=termination_error_threshold,
        act_noise=act_noise
    )
    env = env_util.make_vec_env(
        env_id=tracking.MocapTrackingGymEnv,
        n_envs=FLAGS.n_workers,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        vec_monitor_cls=wrappers.MocapTrackingVecMonitor
    )
    env = VecNormalize(env, training=training, gamma=FLAGS.gamma,
                       norm_obs=True,
                       norm_reward=True,
                       norm_obs_keys=observables.MULTI_CLIP_OBSERVABLES_SANS_ID)
    return env

def main(_):
    clip_length = utils.get_clip_length(FLAGS.clip_id, FLAGS.mocap_path)
    snippet_length = min(clip_length - FLAGS.start_step, FLAGS.max_steps)
    end_step = FLAGS.start_step + snippet_length

    # Log directory
    log_dir = osp.join(FLAGS.log_root, f"{FLAGS.clip_id}-{FLAGS.start_step}-{end_step}", str(FLAGS.seed))
    if FLAGS.include_timestamp:
        now = datetime.now()
        log_dir = osp.join(log_dir, now.strftime("%Y-%m-%d_%H-%M-%S"))

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    clip_info = dict(
        clip_id=FLAGS.clip_id,
        start_step=int(FLAGS.start_step),
        end_step=int(end_step)
    )
    with open(osp.join(log_dir, 'clip_info.json'), 'w') as f:
        json.dump(clip_info, f)

    rsi_eval_path = osp.join(log_dir, 'eval_rsi')
    start_eval_path = osp.join(log_dir, 'eval_start')
    Path(osp.join(rsi_eval_path, 'model')).mkdir(parents=True)
    Path(osp.join(start_eval_path, 'model')).mkdir(parents=True)

    # Logger configuration
    print(FLAGS.flags_into_string())
    with open(osp.join(log_dir, 'flags.txt'), 'w') as f:
        f.write(FLAGS.flags_into_string())
    if FLAGS.do_logging:
        format_strings = ['csv', 'tensorboard', 'stdout']
        logger = configure(log_dir, format_strings)

    # Rollout environment
    env = make_env(
        seed=FLAGS.seed,
        start_step=FLAGS.start_step,
        end_step=end_step,
        min_steps=FLAGS.min_steps,
        training=True,
        act_noise=0.,
        always_init_at_clip_start=False,
        termination_error_threshold=FLAGS.termination_error_threshold
    )

    # Evaluation environment where start point is selected at random
    rsi_eval_env_ctor = lambda: make_env(seed=FLAGS.eval.seed, start_step=FLAGS.start_step,
                                         end_step=end_step, min_steps=FLAGS.eval.min_steps,
                                         act_noise=FLAGS.eval.rsi_eval_act_noise,
                                         training=False, always_init_at_clip_start=False,
                                         termination_error_threshold=FLAGS.termination_error_threshold)
    eval_freq = int(FLAGS.eval.freq / FLAGS.n_workers)
    rsi_eval_model_path = osp.join(rsi_eval_path, 'model')
    callback_on_new_best = sb3_callbacks.SaveVecNormalizeCallback(
        save_freq=1,
        save_path=rsi_eval_model_path
    )
    early_stopping_callback = callbacks.EarlyStoppingCallback(
        FLAGS.eval.early_stop.ep_length_threshold,
        FLAGS.eval.early_stop.min_reward_delta,
        patience=FLAGS.eval.early_stop.patience
    )
    rsi_eval_callback = callbacks.MocapTrackingEvalCallback(
        rsi_eval_env_ctor,
        best_model_save_path=rsi_eval_model_path,
        log_path=rsi_eval_path,
        eval_freq=eval_freq,
        callback_on_new_best=callback_on_new_best,
        callback_after_eval=early_stopping_callback,
        n_eval_episodes=FLAGS.eval.n_rsi_episodes,
        deterministic=True,
        record_video=FLAGS.record_video,
        name="eval_rsi"
    )

    # Evaluation environment where start point is beginning of snippet
    start_eval_env_ctor = lambda: make_env(seed=FLAGS.eval.seed, start_step=FLAGS.start_step,
                                           act_noise=FLAGS.eval.start_eval_act_noise, end_step=end_step,
                                           min_steps=FLAGS.eval.min_steps, training=False,
                                           always_init_at_clip_start=True,
                                           termination_error_threshold=FLAGS.termination_error_threshold)
    start_eval_model_path = osp.join(start_eval_path, 'model')
    callback_on_new_best = sb3_callbacks.SaveVecNormalizeCallback(
        save_freq=1,
        save_path=start_eval_model_path
    )
    start_eval_callback = callbacks.MocapTrackingEvalCallback(
        start_eval_env_ctor,
        best_model_save_path=start_eval_model_path,
        log_path=start_eval_path,
        eval_freq=eval_freq,
        callback_on_new_best=callback_on_new_best,
        n_eval_episodes=FLAGS.eval.n_start_episodes,
        deterministic=True,
        record_video=FLAGS.record_video,
        name="eval_start"
    )

    # Set up model (policy + value)
    lr_schedule = sb3_utils.get_piecewise_constant_fn(FLAGS.learning_rate.start_val, FLAGS.learning_rate.decay)

    # Load a prior policy, if available
    if FLAGS.warm_start_path:
        print("Loading prior model from:", FLAGS.warm_start_path)
        with zipfile.ZipFile(osp.join(FLAGS.warm_start_path, 'best_model.zip')) as archive:
            json_string = archive.read("data").decode()
            json_dict = json.loads(json_string)
            policy_kwargs = {k: v for k, v in json_dict['policy_kwargs'].items() if not k.startswith(":")}
            if 'Tanh' in policy_kwargs['activation_fn']:
                policy_kwargs['activation_fn'] = torch.nn.Tanh
            elif 'ReLU' in policy_kwargs['activation_fn']:
                policy_kwargs['activation_fn'] = torch.nn.ReLU
            else:
                policy_kwargs['activation_fn'] = torch.nn.ELU
        policy_kwargs['features_extractor_class'] = features_extractor.CmuHumanoidFeaturesExtractor
        policy_kwargs['features_extractor_kwargs'] = dict(observable_keys=observables.TIME_INDEX_OBSERVABLES)
        model = PPO.load(
            osp.join(FLAGS.warm_start_path, 'best_model.zip'),
            env,
            device=get_device(FLAGS.device),
            custom_objects=dict(
                policy_kwargs=policy_kwargs,
                n_steps=int(FLAGS.n_steps / FLAGS.n_workers),
                n_envs=FLAGS.n_workers,
                clip_range=FLAGS.clip_range,
                learning_rate=lr_schedule
            )
        )
        with open(osp.join(FLAGS.warm_start_path, 'vecnormalize.pkl'), 'rb') as f:
            prior_norm_env = pickle.load(f)
            env.obs_rms = prior_norm_env.obs_rms
            env.ret_rms = prior_norm_env.ret_rms
    else:
        print("Training from scratch!")
        layer_sizes = FLAGS.n_layers * [FLAGS.layer_size]
        policy_kwargs = dict(
            net_arch=[dict(pi=layer_sizes, vf=layer_sizes)],
            activation_fn=utils.str_to_callable(FLAGS.activation_fn),
            log_std_init=np.log(FLAGS.act_noise),
            features_extractor_class=features_extractor.CmuHumanoidFeaturesExtractor,
            features_extractor_kwargs=dict(observable_keys=observables.TIME_INDEX_OBSERVABLES)
        )
        model = PPO("MultiInputPolicy", env, n_steps=int(FLAGS.n_steps / FLAGS.n_workers),
                    gamma=FLAGS.gamma, clip_range=FLAGS.clip_range,
                    batch_size=FLAGS.batch_size, n_epochs=FLAGS.n_epochs,
                    gae_lambda=FLAGS.gae_lambda, max_grad_norm=FLAGS.max_grad_norm,
                    learning_rate=lr_schedule, target_kl=FLAGS.target_kl,
                    policy_kwargs=policy_kwargs, seed=FLAGS.seed, verbose=1,
                    device=FLAGS.device)

    if FLAGS.do_logging:
        model.set_logger(logger)
    model.policy.log_std.requires_grad = False  # keep policy covariance fixed throughout training

    # Train the model
    callback = [
        rsi_eval_callback,
        start_eval_callback,
        callbacks.NormalizedRolloutCallback(),
        sb3_callbacks.LogOnRolloutEndCallback(log_dir)
    ]
    model.learn(FLAGS.total_timesteps, callback=callback)

if __name__ == '__main__':
    app.run(main)
