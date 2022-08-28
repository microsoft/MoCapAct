"""
Script used for evaluating a clip expert. Can do parallel evaluation,
saving statistics, saving videos, and visualizing the expert.
"""
import os
import os.path as osp
import numpy as np
import torch
import json
import imageio
from absl import app
from absl import flags
from absl import logging
from pathlib import Path
from stable_baselines3.common.vec_env import SubprocVecEnv

from dm_control.viewer import application
from mocapact import observables
from mocapact.envs import env_util
from mocapact.sb3 import evaluation
from mocapact.sb3 import utils
from mocapact.sb3 import wrappers
from mocapact.envs import tracking
from mocapact.clip_expert import utils as clip_expert_utils

FLAGS = flags.FLAGS
flags.DEFINE_string("policy_root", None, "Path to the policy")
flags.DEFINE_integer("min_steps", 10, "Minimum number of steps (used for RSI)")
flags.DEFINE_float("act_noise", 0.1, "Action noise in humanoid")
flags.DEFINE_string("mocap_path", None, "Path to MoCap HDF5. If None, uses dm_control's MoCap data")

# Visualization hyperparameters
flags.DEFINE_bool("visualize", True, "Whether to visualize via GUI")
flags.DEFINE_float("ghost_offset", 0., "Offset for reference ghost")

# Evaluation hyperparameters
flags.DEFINE_integer("n_eval_episodes", 0, "Number of episodes to numerically evaluate policy")
flags.DEFINE_integer("n_workers", 1, "Number of parallel workers")
flags.DEFINE_bool("always_init_at_clip_start", False, "Whether to initialize at beginning or random point in clip")
flags.DEFINE_float("termination_error_threshold", 0.3, "Error for cutting off expert rollout")
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_string("eval_save_path", None, "If desired, the path to save the evaluation results")
flags.DEFINE_string("video_save_path", None, "If desired, path to save videos of rollouts")
flags.DEFINE_string("device", "auto", "Device to run evaluation on")

flags.mark_flag_as_required("policy_root")

logging.set_verbosity(logging.WARNING)

def main(_):
    # Make environment
    with open(osp.join(FLAGS.policy_root, osp.pardir, osp.pardir, 'clip_info.json')) as f:
        clip_info = json.load(f)
    clip_id = clip_info['clip_id']
    start_step = clip_info['start_step']
    end_step = clip_info['end_step']
    env_kwargs = clip_expert_utils.make_env_kwargs(
        clip_id,
        FLAGS.mocap_path,
        start_step,
        end_step,
        FLAGS.min_steps-1,
        FLAGS.ghost_offset,
        FLAGS.always_init_at_clip_start,
        FLAGS.termination_error_threshold,
        FLAGS.act_noise
    )

    model = utils.load_policy(FLAGS.policy_root, observables.TIME_INDEX_OBSERVABLES, device=FLAGS.device)

    if FLAGS.n_eval_episodes > 0:
        # VecEnv for evaluation
        vec_env = env_util.make_vec_env(
            env_id=tracking.MocapTrackingGymEnv,
            n_envs=FLAGS.n_workers,
            seed=FLAGS.seed,
            env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv,
            vec_monitor_cls=wrappers.MocapTrackingVecMonitor
        )
        record_video = FLAGS.video_save_path is not None
        ep_rews, ep_lens, ep_norm_rews, ep_norm_lens, frames = evaluation.evaluate_locomotion_policy(
            model,
            vec_env,
            n_eval_episodes=FLAGS.n_eval_episodes,
            deterministic=True,
            render=record_video,
            return_episode_rewards=True
        )
        rews_per_step = np.array(ep_rews) / np.array(ep_lens)
        print(f"Mean return:                    {np.mean(ep_rews):.1f} +/- {np.std(ep_lens):.1f}")
        print(f"Mean episode length:            {np.mean(ep_lens):.1f} +/- {np.std(ep_lens):.1f}")
        print(f"Mean normalized return:         {np.mean(ep_norm_rews):.3f} +/- {np.std(ep_norm_rews):.3f}")
        print(f"Mean normalized episode length: {np.mean(ep_norm_lens):.3f} +/- {np.std(ep_norm_lens):.3f}")
        print(f"Mean reward per step:           {np.mean(rews_per_step):.3f} +/- {np.std(rews_per_step):.3f}")

        if FLAGS.eval_save_path is not None:
            Path(osp.dirname(FLAGS.eval_save_path)).mkdir(parents=True, exist_ok=True)
            np.savez(
                FLAGS.eval_save_path,
                ep_rews=ep_rews,
                ep_lens=ep_lens,
                ep_norm_rews=ep_norm_rews,
                ep_norm_lens=ep_norm_lens,
                rews_per_step=rews_per_step
            )
        if record_video:
            Path(osp.dirname(FLAGS.video_save_path)).mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(FLAGS.video_save_path, frames, fps=1/0.03)

    if FLAGS.visualize:
        # env for visualization
        env = tracking.MocapTrackingGymEnv(**env_kwargs)

        @torch.no_grad()
        def policy_fn(time_step):
            action, _ = model.predict(env.get_observation(time_step), deterministic=True)
            return action

        viewer_app = application.Application(title='Clip Expert', width=1024, height=768)
        viewer_app.launch(environment_loader=env.dm_env, policy=policy_fn)

if __name__ == '__main__':
    app.run(main)
