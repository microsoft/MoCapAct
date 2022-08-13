"""
Script used for evaluating a PyTorch Lightning policy. Can do parallel evaluation,
saving statistics, saving videos, and visualizing the policy.
"""
import os.path as osp
import numpy as np
from absl import app, flags, logging
from pathlib import Path
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import torch
import imageio

from dm_control.viewer import application

from mocapact import utils
from mocapact.envs import env_util
from mocapact.envs import tracking
from mocapact.sb3 import evaluation
from mocapact.sb3 import wrappers

FLAGS = flags.FLAGS
flags.DEFINE_string("policy_path", None, "Path to the saved policy")
flags.DEFINE_list("clip_snippets", None, "A list of clip snippets, of form CMU_{clip id}[-{start step}-{end step}")
flags.DEFINE_float("act_noise", 0.1, "Action noise in humanoid")
flags.DEFINE_string("device", "cpu", "Device to run evaluation on")

# Evaluation hyperparameters
flags.DEFINE_integer("n_eval_episodes", 0, "Number of episodes to numerically evaluate policy")
flags.DEFINE_integer("n_workers", 1, "Number of parallel workers")
flags.DEFINE_bool("always_init_at_clip_start", False, "Whether to initialize at beginning or random point in clip")
flags.DEFINE_bool("deterministic", False, "Whether the policy is deterministic")
flags.DEFINE_float("termination_error_threshold", 0.3, "Error for cutting off rollout")
flags.DEFINE_integer("min_steps", 10, "Minimum steps left at end of episode")
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_string("eval_save_path", None, "If desired, the path to save the evaluation results")
flags.DEFINE_string("video_save_path", None, "Path to save a video recording of the evaluation")

# Visualization hyperparameters
flags.DEFINE_float("ghost_offset", 0., "Offset for reference ghost")
flags.DEFINE_bool("visualize", True, "Whether to visualize via GUI")

flags.mark_flag_as_required("policy_path")
flags.mark_flag_as_required("clip_snippets")
logging.set_verbosity(logging.WARNING)

def main(_):
    clips = utils.make_clip_collection(FLAGS.clip_snippets)

    # set up model
    is_in_eval_dir = osp.exists(osp.join(osp.dirname(osp.dirname(osp.dirname(FLAGS.policy_path))), 'model_constructor.txt'))
    model_constructor_path = (osp.join(osp.dirname(osp.dirname(osp.dirname(FLAGS.policy_path))), 'model_constructor.txt')
                              if is_in_eval_dir
                              else osp.join(osp.dirname(osp.dirname(FLAGS.policy_path)), 'model_constructor.txt'))
    with open(model_constructor_path, 'r') as f:
        model_cls = utils.str_to_callable(f.readline())
    policy = model_cls.load_from_checkpoint(FLAGS.policy_path, map_location=FLAGS.device)

    # set up environment
    task_kwargs = dict(
        reward_type='comic',
        min_steps=FLAGS.min_steps-1,
        ghost_offset=np.array([FLAGS.ghost_offset, 0., 0.]),
        always_init_at_clip_start=FLAGS.always_init_at_clip_start,
        termination_error_threshold=FLAGS.termination_error_threshold
    )
    env_kwargs = dict(
        dataset=clips,
        ref_steps=policy.ref_steps,
        act_noise=FLAGS.act_noise,
        task_kwargs=task_kwargs
    )

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
        ep_rews, ep_lens, ep_norm_rews, ep_norm_lens, ep_frames = evaluation.evaluate_locomotion_policy(
            policy,
            vec_env,
            n_eval_episodes=FLAGS.n_eval_episodes,
            deterministic=FLAGS.deterministic,
            render=record_video,
            return_episode_rewards=True,
        )
        rews_per_step = np.array(ep_rews) / np.array(ep_lens)
        print(f"Mean return:                    {np.mean(ep_rews):.1f} +/- {np.std(ep_rews):.1f}")
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
                rews_per_steps=rews_per_step
            )
        if record_video:
            Path(osp.dirname(FLAGS.video_save_path)).mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(FLAGS.video_save_path, ep_frames, fps=1/0.03)

    if FLAGS.visualize:
        # env for visualization
        env = tracking.MocapTrackingGymEnv(**env_kwargs)

        state = None
        @torch.no_grad()
        def policy_fn(time_step):
            nonlocal state
            if time_step.step_type == 0: # first time step
                state = None
            action, state = policy.predict(env.get_observation(time_step), state, deterministic=FLAGS.deterministic)
            return action

        viewer_app = application.Application(title='Distillation', width=1024, height=768)
        viewer_app.launch(environment_loader=env.dm_env, policy=policy_fn)

if __name__ == '__main__':
    app.run(main)
