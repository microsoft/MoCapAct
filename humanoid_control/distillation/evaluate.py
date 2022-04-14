import os.path as osp
import numpy as np
from absl import app, flags, logging
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import torch

from dm_control.locomotion.tasks.reference_pose import types
from dm_control.viewer import application

from humanoid_control import utils
from humanoid_control.envs import env_util
from humanoid_control.sb3 import evaluation
from humanoid_control.sb3 import tracking
from humanoid_control.envs import wrappers

FLAGS = flags.FLAGS
flags.DEFINE_string("policy_root", None, "Directory where saved policy is stored")
flags.DEFINE_list("clip_snippets", None, "A list of clip snippets, of form CMU_{clip id}[-{start step}-{end step}")
flags.DEFINE_float("act_noise", 0.1, "Action noise in humanoid")

# Evaluation hyperparameters
flags.DEFINE_integer("n_eval_episodes", 0, "Number of episodes to numerically evaluate policy")
flags.DEFINE_integer("n_workers", 8, "Number of parallel workers")
flags.DEFINE_bool("always_init_at_clip_start", False, "Whether to initialize at beginning or random point in clip")
flags.DEFINE_bool("deterministic", False, "Whether the policy is deterministic")
flags.DEFINE_float("termination_error_threshold", 0.3, "Error for cutting off rollout")
flags.DEFINE_integer("min_steps", 10, "Minimum steps left at end of episode")
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_string("save_path", None, "If desired, the path to save the evaluation results")

# Visualization hyperparameters
flags.DEFINE_float("ghost_offset", 0., "Offset for reference ghost")
flags.DEFINE_bool("visualize", True, "Whether to visualize via GUI")

flags.mark_flag_as_required("policy_root")
flags.mark_flag_as_required("clip_snippets")
logging.set_verbosity(logging.WARNING)

def make_clips():
    ids, start_steps, end_steps = [], [], []
    for clip_snippet in FLAGS.clip_snippets:
        substrings = clip_snippet.split('-')
        ids.append(substrings[0])
        if len(substrings) >= 2:
            start_steps.append(int(substrings[1]))
        else:
            start_steps.append(0)

        if len(substrings) >= 3:
            end_steps.append(int(substrings[2]))
        else:
            clip_length = utils.get_clip_length(substrings[0])
            end_steps.append(clip_length)

    return types.ClipCollection(ids=ids, start_steps=start_steps, end_steps=end_steps)

def main(_):
    clips = make_clips()

    # set up model
    with open(osp.join(osp.dirname(osp.dirname(osp.dirname(FLAGS.policy_root))), 'model_constructor.txt'), 'r') as f:
    #with open(osp.join(osp.dirname(FLAGS.policy_root), 'model_constructor.txt'), 'r') as f:
        model_cls = utils.str_to_callable(f.readline())
    policy = model_cls.load_from_checkpoint(FLAGS.policy_root, map_location='cpu')
    #policy.to('cuda:1')
    policy.observation_space['walker/clip_id'].n = int(1e6)

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

    if FLAGS.n_eval_episodes > 0:
        ep_rews, ep_lens, ep_norm_rews, ep_norm_lens = evaluation.evaluate_locomotion_policy(
            policy,
            vec_env,
            n_eval_episodes=FLAGS.n_eval_episodes,
            deterministic=FLAGS.deterministic,
            return_episode_rewards=True
        )
        print(f"Mean return: {np.mean(ep_rews):.1f} +/- {np.std(ep_rews):.1f}")
        print(f"Mean episode length: {np.mean(ep_lens):.1f} +/- {np.std(ep_lens):.1f}")
        print(f"Mean normalized return: {np.mean(ep_norm_rews):.3f} +/- {np.std(ep_norm_rews):.3f}")
        print(f"Mean normalized episode length: {np.mean(ep_norm_lens):.3f} +/- {np.std(ep_norm_lens):.3f}")

        if FLAGS.save_path is not None:
            np.savez(
                osp.join(FLAGS.save_path, 'results.npz'),
                ep_rews=ep_rews,
                ep_lens=ep_lens,
                ep_norm_rews=ep_norm_rews,
                ep_norm_lens=ep_norm_lens
            )

    state = None
    @torch.no_grad()
    def policy_fn(time_step):
        nonlocal state
        if time_step.step_type == 0: # first time step
            state = None
        action, state = policy.predict(env._get_obs(time_step), state, deterministic=FLAGS.deterministic)
        return action

    if FLAGS.visualize:
        viewer_app = application.Application(title='Explorer', width=1024, height=768)
        viewer_app.launch(environment_loader=env._env, policy=policy_fn)

if __name__ == '__main__':
    app.run(main)
