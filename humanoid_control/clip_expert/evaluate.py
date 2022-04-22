import os.path as osp
import pickle
import zipfile
import numpy as np
import torch
import json
from absl import app
from absl import flags
from absl import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from dm_control.viewer import application
from dm_control.locomotion.tasks.reference_pose import types
from humanoid_control import observables
from humanoid_control.envs import env_util
from humanoid_control.sb3 import evaluation
from humanoid_control.sb3 import features_extractor
from humanoid_control.sb3 import wrappers
from humanoid_control.envs import tracking

MIN_STEPS = 10

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_root", None, "Directory where experiment results are stored")
flags.DEFINE_float("act_noise", 0.1, "Action noise in humanoid")

# Visualization hyperparameters
flags.DEFINE_bool("visualize", True, "Whether to visualize via GUI")
flags.DEFINE_float("ghost_offset", 0., "Offset for reference ghost")

# Evaluation hyperparameters
flags.DEFINE_integer("n_eval_episodes", 0, "Number of episodes to numerically evaluate policy")
flags.DEFINE_integer("n_workers", 1, "Number of parallel workers")
flags.DEFINE_bool("always_init_at_clip_start", False, "Whether to initialize at beginning or random point in clip")
flags.DEFINE_float("termination_error_threshold", 0.3, "Error for cutting off expert rollout")
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_string("save_path", None, "If desired, the path to save the evaluation results")

flags.mark_flag_as_required("exp_root")

logging.set_verbosity(logging.WARNING)

def main(_):
    model_path = osp.join(
        FLAGS.exp_root,
        'eval_start' if FLAGS.always_init_at_clip_start else 'eval_random',
        'model'
    )

    # Make environment
    with open(osp.join(FLAGS.exp_root, 'clip_info.json')) as f:
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
        min_steps=MIN_STEPS-1,
        ghost_offset=np.array([FLAGS.ghost_offset, 0., 0.]),
        always_init_at_clip_start=FLAGS.always_init_at_clip_start,
        termination_error_threshold=FLAGS.termination_error_threshold
    )
    env_kwargs = dict(
        dataset=dataset,
        ref_steps=(0,),
        act_noise=FLAGS.act_noise,
        task_kwargs=task_kwargs
    )

    # VecEnv for evaluation
    vec_env = env_util.make_vec_env(
        env_id=tracking.MocapTrackingGymEnv,
        n_envs=FLAGS.n_workers,
        seed=FLAGS.seed,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        vec_monitor_cls=wrappers.MocapTrackingVecMonitor
    )

    # env for visualization
    env = tracking.MocapTrackingGymEnv(**env_kwargs)

    # Normalization statistics
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

    if FLAGS.n_eval_episodes > 0:
        ep_rews, ep_lens, ep_norm_rews, ep_norm_lens, _ = evaluation.evaluate_locomotion_policy(
            model,
            vec_env,
            n_eval_episodes=FLAGS.n_eval_episodes,
            deterministic=True,
            return_episode_rewards=True
        )
        print(f"Mean return: {np.mean(ep_rews):.1f} +/- {np.std(ep_lens):.1f}")
        print(f"Mean episode length: {np.mean(ep_lens):.1f} +/- {np.std(ep_lens):.1f}")
        print(f"Mean normalized return: {np.mean(ep_norm_rews):.3f} +/- {np.std(ep_norm_rews):.3f}")
        print(f"Mean normalized episode length: {np.mean(ep_norm_lens):.3f} +/- {np.std(ep_norm_lens):.3f}")

        if FLAGS.save_path is not None:
            np.savez(
                osp.join(FLAGS.save_path, flags['clip_id']),
                ep_rews=ep_rews,
                ep_lens=ep_lens,
                ep_norm_rews=ep_norm_rews,
                ep_norm_lens=ep_norm_lens
            )

    @torch.no_grad()
    def policy_fn(time_step):
        action, _ = model.predict(env._get_obs(time_step), deterministic=True)
        return action

    if FLAGS.visualize:
        viewer_app = application.Application(title='Explorer', width=1024, height=768)
        viewer_app.launch(environment_loader=env._env, policy=policy_fn)

if __name__ == '__main__':
    app.run(main)
