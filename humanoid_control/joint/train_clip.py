import os
import os.path as osp
from pathlib import Path
import glob
from datetime import datetime
import pickle
import zipfile
import gym
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from absl import app
from absl import flags
import ml_collections
from ml_collections import config_flags
from datetime import datetime

from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.utils import get_device

from dm_control.locomotion.tasks.reference_pose import types
from humanoid_control import observables
from humanoid_control import utils
from humanoid_control.clip_expert import callbacks
from humanoid_control.distillation import dataset
# from stable_baselines3.common import env_util
from humanoid_control.envs import env_util
from humanoid_control.sb3 import features_extractor
from humanoid_control.tasks import stand
from humanoid_control.sb3 import utils as sb3_utils
from humanoid_control.envs import wrappers
from humanoid_control.joint.ppo import PPOBC
from humanoid_control.joint.a2c import A2CBC
from humanoid_control.sb3 import tracking
from dm_control.locomotion.tasks.reference_pose import types


# min steps for rollouts and evaluations
ROLLOUT_MIN_STEPS = 1
EVAL_MIN_STEPS = 10

FLAGS = flags.FLAGS
flags.DEFINE_string("log_root", None, "Directory where logs are stored")
flags.DEFINE_float("act_noise", 0.1, "Action noise to apply")
flags.DEFINE_list("train_dataset_paths", None, "Path(s) to training dataset(s)")

# Training hyperparameters
flags.DEFINE_integer("total_timesteps", int(1e8), "Total number of env steps")
flags.DEFINE_integer("n_workers", 16, "Number of workers used for rollouts")
flags.DEFINE_integer("n_steps", 4096, "Number of steps per policy optimization")
flags.DEFINE_integer("n_epochs", 10, "Number of epochs when optimizing the PPO loss")
flags.DEFINE_integer("batch_size", 256, "Minibatch size for PPO")
flags.DEFINE_float("gamma", 0.95, "Discount factor")
flags.DEFINE_float("clip_range", 0.2, "Clipping parameter for PPO")
flags.DEFINE_float("target_kl", 0.15, "Limits KL divergence in updating policy")
flags.DEFINE_float("max_grad_norm", 1., "Clipping value for gradient norm")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda parameter")
flags.DEFINE_float("kl_coef", 1e-5, "")
flags.DEFINE_float("bc_coef", 1., "")
flags.DEFINE_bool("normalize_observation", True, "Whether to normalize the observations")
flags.DEFINE_bool("normalize_reward", True, "Whether to normalize the rewards")
lr_config = ml_collections.ConfigDict()
lr_config.start_val = 1e-4 # Initial step size
lr_config.decay_half_life = 0.2 # Half-life for decay rate of learning rate
lr_config.min_val = 1e-6 # Minimum step size
config_flags.DEFINE_config_dict("learning_rate", lr_config)

# Network hyperparameters
config_file = "humanoid_control/joint/config.py"
config_flags.DEFINE_config_file("model", f"{config_file}:mlp", "Model architecture")

# Evaluation hyperparameters
eval_config = ml_collections.ConfigDict()
eval_config.seed = 0 # RNG seed for evaluation
eval_config.freq = int(1e5) # After how many total environment steps to evaluate policy
eval_config.n_episodes = 32 # Number of episodes to evaluate the policy from random initial states
eval_config.act_noise = 0.1 # Action noise to apply for start of snippet
#eval_config.early_stop = ml_collections.ConfigDict()
#eval_config.early_stop.ep_length_threshold = 1. # Episode length threshold for early stopping
#eval_config.early_stop.min_reward_delta = float('inf') # Minimum change in normalized reward to qualify as improvement
#eval_config.early_stop.patience = 10 # Number of queries with no improvement after which training is stopped
config_flags.DEFINE_config_dict("eval", eval_config)

# Misc. hyperparameters
flags.DEFINE_integer("seed", 0, "RNG seed for training")
flags.DEFINE_enum("device", "auto", ["auto", "cpu", "cuda", "cuda:0", "cuda:1"], "Device to do training on")
flags.DEFINE_string("warm_start_path", None, "")

flags.mark_flag_as_required('log_root')

#def make_env(seed=0, training=True):
def make_env(seed=0, training=True, min_steps=10, always_init_at_clip_start=False):
    embed_dim = FLAGS.model.config.embed_size if hasattr(FLAGS.model.config, 'embed_size') else 0
    #env = env_util.make_vec_env(
    #    env_id=stand.StandUpGymEnv,
    #    n_envs=FLAGS.n_workers,
    #    seed=seed,
    #    wrapper_class=wrappers.Embedding,
    #    vec_env_cls=SubprocVecEnv,
    #    wrapper_kwargs=dict(embed_dim=embed_dim, embed_max=3.)
    #)
    dataset = types.ClipCollection(
        ids=['CMU_016_22'],
        start_steps=[0],
        end_steps=None
    )
    task_kwargs = dict(
        reward_type='comic',
        min_steps=min_steps,
        always_init_at_clip_start=always_init_at_clip_start,
        termination_error_threshold=0.3
    )
    env_kwargs = dict(
        dataset=dataset,
        ref_steps=(1,2,3,4,5),
        act_noise=0.,
        task_kwargs=task_kwargs
    )
    env = env_util.make_vec_env(
        env_id=tracking.MocapTrackingGymEnv,
        n_envs=FLAGS.n_workers,
        seed=seed,
        wrapper_class=wrappers.Embedding,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        vec_monitor_cls=wrappers.MocapTrackingVecMonitor,
        wrapper_kwargs=dict(embed_dim=embed_dim, embed_max=3.)
    )
    env = VecNormalize(env, training=training, gamma=FLAGS.gamma,
                        norm_obs=FLAGS.normalize_observation,
                        norm_reward=FLAGS.normalize_reward,
                        #norm_obs_keys=observables.CMU_HUMANOID_OBSERVABLES + ('embedding',))
                        norm_obs_keys=observables.MULTI_CLIP_OBSERVABLES_SANS_ID + ('embedding',))
    return env

def main(_):
    # Log directory
    now = datetime.now()
    log_dir = osp.join(FLAGS.log_root,
                       str(FLAGS.seed), now.strftime("%Y-%m-%d_%H-%M-%S"))

    eval_path = osp.join(log_dir, 'eval')
    Path(osp.join(eval_path, 'model')).mkdir(parents=True, exist_ok=True)

    # Logger configuration
    print(FLAGS.flags_into_string())
    format_strings = ['csv', 'tensorboard', 'stdout']
    logger = configure(log_dir, format_strings)

    if hasattr(FLAGS.model.config, 'seq_steps'):
        seq_steps = FLAGS.model.config.seq_steps
    elif hasattr(FLAGS.model.config, 'block_size'):
        seq_steps = FLAGS.model.config.block_size
    else:
        seq_steps = 1

    dataloader = None
    if FLAGS.train_dataset_paths:
        train_dataset = dataset.ExpertDataset(
            FLAGS.train_dataset_paths,
            observables.HIERARCHICAL_OBSERVABLES,
            #observables.MULTI_CLIP_OBSERVABLES_SANS_ID,
            min_seq_steps=seq_steps,
            max_seq_steps=seq_steps,
            concat_observables=True,
            normalize_obs=True
        )
        dataloader = DataLoader(train_dataset, FLAGS.batch_size, num_workers=4, shuffle=True)

    # Rollout environment
    if FLAGS.train_dataset_paths:
        env = make_env(seed=FLAGS.seed, training=True, min_steps=1, always_init_at_clip_start=False)
        #env.obs_rms = deepcopy(train_dataset.obs_rms)
        for k in env.obs_rms.keys():
            if k == 'embedding':
                continue
            env.obs_rms[k] = train_dataset.obs_rms[k]
    else:
        env = make_env(seed=FLAGS.seed, training=True)

    # Evaluation environment where start point is selected at random
    eval_freq = int(FLAGS.eval.freq / FLAGS.n_workers)
    eval_model_path = osp.join(eval_path, 'model')
    callback_on_new_best = callbacks.SaveVecNormalizeCallback(
        save_freq=1,
        save_path=eval_model_path
    )
    random_eval_env_ctor = lambda: make_env(seed=FLAGS.eval.seed, training=False, min_steps=10, always_init_at_clip_start=False)
    eval_callback = callbacks.MocapTrackingEvalCallback(
        random_eval_env_ctor,
        best_model_save_path=eval_model_path,
        log_path=eval_path,
        eval_freq=eval_freq,
        callback_on_new_best=callback_on_new_best,
        n_eval_episodes=FLAGS.eval.n_episodes,
        deterministic=False,
        render=False,
        name='eval'
    )

    # Set up model (policy + value)
    decay = np.log(2) / FLAGS.learning_rate.decay_half_life
    lr_schedule = sb3_utils.get_exponential_fn(FLAGS.learning_rate.start_val, decay, FLAGS.learning_rate.min_val)
    policy = utils.str_to_callable(FLAGS.model.constructor)
    space = gym.spaces.Dict({**env.observation_space, **deepcopy(train_dataset.full_observation_space)})
    model_observables = FLAGS.model.config.observables

    features_extractor_class = lambda _: features_extractor.CmuHumanoidFeaturesExtractor(
        observation_space=space,
        observable_keys=model_observables
    )
    policy_kwargs = dict(
        #ref_steps=train_dataset.ref_steps,
        ref_steps=(1,2,3,4,5),
        features_extractor_class=features_extractor_class,
        **FLAGS.model.config
    )
    #layer_sizes = FLAGS.model.config.n_layers * [FLAGS.model.config.layer_size]
    #policy_kwargs = dict(
    #    net_arch=[dict(pi=layer_sizes, vf=layer_sizes)],
    #    activation_fn=utils.str_to_callable(FLAGS.model.config.activation_fn),
    #    log_std_init=np.log(0.1),
    #    full_std=False,
    #    features_extractor_class=features_extractor.CmuHumanoidFeaturesExtractor,
    #    features_extractor_kwargs=dict(observable_keys=FLAGS.model.config.observables)
    #)
    model = PPOBC(policy, env, n_steps=int(FLAGS.n_steps/FLAGS.n_workers),
                gamma=FLAGS.gamma, clip_range=FLAGS.clip_range,
                batch_size=FLAGS.batch_size, n_epochs=FLAGS.n_epochs,
                gae_lambda=FLAGS.gae_lambda, max_grad_norm=FLAGS.max_grad_norm,
                learning_rate=lr_schedule, target_kl=FLAGS.target_kl,
                seed=FLAGS.seed, verbose=1, kl_coef=FLAGS.kl_coef, policy_kwargs=policy_kwargs,
                device=FLAGS.device, bc_coef=FLAGS.bc_coef, bc_dataloader=dataloader)
    #model = A2CBC(policy, env, learning_rate=lr_schedule, n_steps=int(FLAGS.n_steps/FLAGS.n_workers),
    #              gamma=FLAGS.gamma, gae_lambda=FLAGS.gae_lambda, kl_coef=FLAGS.kl_coef,
    #              bc_coef=FLAGS.bc_coef, max_grad_norm=FLAGS.max_grad_norm, bc_dataloader=dataloader,
    #              use_rms_prop=False, policy_kwargs=policy_kwargs, seed=FLAGS.seed, device=FLAGS.device)
    model.set_logger(logger)
    if FLAGS.warm_start_path:
        params = torch.load(FLAGS.warm_start_path)
        #model.policy.load_state_dict(params['state_dict'], strict=False)
        model.policy.load_state_dict(params, strict=False)
    #model.policy.log_std.requires_grad = False

    # Train the model
    callback = [
        eval_callback,
        callbacks.NormalizedRolloutCallback(),
        callbacks.LogOnRolloutEndCallback(log_dir)
    ]
    model.learn(FLAGS.total_timesteps, callback=callback)

    print("Finished!")
    Path(osp.join(log_dir, "FINISHED")).touch()
    Path(osp.join(osp.dirname(log_dir), "SUCCEEDED")).touch()

if __name__ == '__main__':
    app.run(main)
