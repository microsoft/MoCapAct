"""
Main script for training a policy for a RL transfer task.
"""
import os.path as osp
import gym
import shutil
from pathlib import Path
import numpy as np
from absl import app, flags
import ml_collections
from ml_collections import config_flags
from ml_collections.config_flags import config_flags
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from dm_control.locomotion.tasks.reference_pose import tracking

from mocapact import utils
from mocapact.envs import env_util
from mocapact.envs import dm_control_wrapper
from mocapact.sb3 import features_extractor
from mocapact.sb3 import wrappers
from mocapact.sb3 import callbacks
from mocapact.distillation import model

FLAGS = flags.FLAGS
# Environment hyperparameters
flags.DEFINE_integer("episode_steps", 833, "Number of time steps in an episode")
task_file = "mocapact/transfer/config.py"
config_flags.DEFINE_config_file("task", f"{task_file}:go_to_target", "Task")

# Training hyperparameters
flags.DEFINE_string("log_root", None, "Directory where logs are stored")
flags.DEFINE_integer("total_timesteps", int(1e9), "Total number of env steps")
flags.DEFINE_integer("n_workers", 16, "Number of workers used for rollouts")
flags.DEFINE_integer("n_steps", 4096, "Number of steps per policy optimization")
flags.DEFINE_integer("n_epochs", 10, "Number of epochs when optimizing the PPO loss")
flags.DEFINE_integer("batch_size", 256, "Minibatch size for PPO")
flags.DEFINE_float("clip_range", 0.2, "Clipping parameter for PPO")
flags.DEFINE_float("target_kl", 0.15, "Limits KL divergence in updating policy")
flags.DEFINE_float("max_grad_norm", 1., "Clipping value for gradient norm")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda parameter")
flags.DEFINE_float("ent_coef", 0., "Entropy coefficient for PPO")
flags.DEFINE_bool("normalize_observation", True, "Whether to normalize the observations")
flags.DEFINE_bool("normalize_reward", True, "Whether to normalize the rewards")
flags.DEFINE_float("learning_rate", 1e-4, "Step size for PPO")

# Low-level policy hyperparameters
flags.DEFINE_string("low_level_policy_path", None, "Path to low-level policy, if desired")
flags.DEFINE_float("max_embed", 3., "If there's a low-level policy, the element-wise maximum embedding")

# Network hyperparameters
flags.DEFINE_integer("n_layers", 3, "Number of hidden layers")
flags.DEFINE_integer("layer_size", 1024, "Number of neurons in hidden layer")
flags.DEFINE_enum("activation_fn", "torch.nn.Tanh", ["torch.nn.ReLU", "torch.nn.Tanh", "torch.nn.ELU"], "Activation for the hidden layers")
flags.DEFINE_float("std_init", 1., "Initial standard deviation for policy")

# Evaluation hyperparameters
eval_config = ml_collections.ConfigDict()
eval_config.seed = 0                                    # RNG seed for evaluation
eval_config.freq = int(1e5)                             # After how many total environment steps to evaluate policy
eval_config.n_episodes = 100
config_flags.DEFINE_config_dict("eval", eval_config)

# Misc. hyperparameters
flags.DEFINE_float("gamma", 0.99, "Discount factor")
flags.DEFINE_integer("seed", 0, "RNG seed for training")
flags.DEFINE_enum("device", "auto", ["auto", "cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"], "Device to do training on")
flags.DEFINE_bool("check_other_runs", False, "Whether to check if preceding runs were finished")
flags.DEFINE_bool("do_logging", True, "Whether to log")
flags.DEFINE_bool("record_video", False, "Whether to record video for evaluation")
flags.DEFINE_bool("include_timestamp", True, "Whether to include timestamp in log directory")

flags.mark_flag_as_required('log_root')

def make_env(seed=0, training=True):
    env_id = dm_control_wrapper.DmControlWrapper.make_env_constructor(FLAGS.task.constructor)
    task_kwargs = dict(
        physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
        control_timestep=0.03,
        **FLAGS.task.config
    )
    env_kwargs = dict(task_kwargs=task_kwargs)
    env = env_util.make_vec_env(
        env_id=env_id,
        n_envs=FLAGS.n_workers,
        seed=seed,
        wrapper_class=gym.wrappers.TimeLimit,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        vec_monitor_cls=wrappers.VecMonitor,
        wrapper_kwargs=dict(max_episode_steps=FLAGS.episode_steps)
    )
    if FLAGS.low_level_policy_path:
        distilled_model = model.NpmpPolicy.load_from_checkpoint(
            FLAGS.low_level_policy_path,
            map_location='cpu'
        )
        env = wrappers.EmbedToActionVecWrapper(
            env,
            distilled_model.embed_size,
            max_embed=FLAGS.max_embed,
            embed_to_action=distilled_model.low_level_policy
        )
    env = VecNormalize(env, training=training, gamma=FLAGS.gamma,
                       norm_obs=FLAGS.normalize_observation,
                       norm_reward=FLAGS.normalize_reward)
    return env

def main(_):
    # Log directory
    log_dir = osp.join(FLAGS.log_root, str(FLAGS.seed))
    if FLAGS.include_timestamp:
        now = datetime.now()
        log_dir = osp.join(log_dir, now.strftime("%Y-%m-%d_%H-%M-%S"))

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    eval_path = osp.join(log_dir, 'eval')
    Path(osp.join(eval_path, 'model')).mkdir(parents=True)

    # Logger configuration
    print(FLAGS.flags_into_string())
    with open(osp.join(log_dir, 'flags.txt'), 'w') as f:
        f.write(FLAGS.flags_into_string())
    if FLAGS.do_logging:
        format_strings = ['csv', 'tensorboard', 'stdout']
        logger = configure(log_dir, format_strings)

    # If given a low-level policy, copy it to log directory to ease evaluation
    if FLAGS.low_level_policy_path:
        shutil.copyfile(
            FLAGS.low_level_policy_path,
            osp.join(eval_path, 'model/low_level_policy.ckpt')
        )

    # Rollout environment
    env = make_env(
        seed=FLAGS.seed,
        training=True,
    )

    # Evaluation environment where start point is selected at random
    eval_env = make_env(seed=FLAGS.eval.seed, training=False)
    eval_freq = int(FLAGS.eval.freq / FLAGS.n_workers)
    eval_model_path = osp.join(eval_path, 'model')
    callback_on_new_best = callbacks.SaveVecNormalizeCallback(
        save_freq=1,
        save_path=eval_model_path
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_model_path,
        log_path=eval_path,
        eval_freq=eval_freq,
        callback_on_new_best=callback_on_new_best,
        callback_after_eval=callbacks.SeedEnvCallback(FLAGS.eval.seed),
        n_eval_episodes=FLAGS.eval.n_episodes,
        deterministic=True,
    )

    layer_sizes = FLAGS.n_layers * [FLAGS.layer_size]
    policy_kwargs = dict(
        net_arch=[dict(pi=layer_sizes, vf=layer_sizes)],
        activation_fn=utils.str_to_callable(FLAGS.activation_fn),
        log_std_init=np.log(FLAGS.std_init),
        features_extractor_class=features_extractor.CmuHumanoidFeaturesExtractor,
        features_extractor_kwargs=dict(observable_keys=list(env.observation_space.keys()))
    )
    model = PPO("MultiInputPolicy", env, n_steps=int(FLAGS.n_steps / FLAGS.n_workers),
                gamma=FLAGS.gamma, clip_range=FLAGS.clip_range, ent_coef=FLAGS.ent_coef,
                batch_size=FLAGS.batch_size, n_epochs=FLAGS.n_epochs,
                gae_lambda=FLAGS.gae_lambda, max_grad_norm=FLAGS.max_grad_norm,
                learning_rate=FLAGS.learning_rate, target_kl=FLAGS.target_kl,
                policy_kwargs=policy_kwargs, seed=FLAGS.seed, verbose=1,
                device=FLAGS.device)

    if FLAGS.do_logging:
        model.set_logger(logger)

    # Train the model
    callback = [
        eval_callback,
    ]
    model.learn(FLAGS.total_timesteps, callback=callback)

if __name__ == '__main__':
    app.run(main)
