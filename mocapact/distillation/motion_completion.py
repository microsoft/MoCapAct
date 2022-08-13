"""
Script used for performing motion completion with a trained GPT policy.
"""
import os
import os.path as osp
import numpy as np
import torch
import json
from absl import app, flags, logging
from pathlib import Path
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from stable_baselines3.common import evaluation

from dm_control.viewer import application
from dm_control.locomotion.tasks.reference_pose import types

from mocapact import observables
from mocapact import utils
from mocapact.envs import env_util
from mocapact.envs import motion_completion
from mocapact.sb3 import utils as sb3_utils
from mocapact.distillation.model import NpmpPolicy

FLAGS = flags.FLAGS
flags.DEFINE_string("policy_path", None, "Path where saved policy is stored")
flags.DEFINE_string("expert_root", None, "Path to expert used to generate context")
flags.DEFINE_string("distillation_path", None, "Path where distilled expert is stored")
flags.DEFINE_string("clip_snippet", None, "A clip snippets, of form CMU_{clip id}[-{start step}-{end step}")
flags.DEFINE_string("device", "cpu", "Device to do rollouts on")

# Environment hyperparameters
flags.DEFINE_bool("always_init_at_clip_start", False, "Whether to initialize at beginning or random point in clip")
flags.DEFINE_bool("deterministic", False, "Whether the policy is deterministic")
flags.DEFINE_float("termination_error_threshold", 0.3, "Error for cutting off rollout")
flags.DEFINE_integer("min_steps", 10, "Minimum steps left at end of episode")
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_integer("prompt_length", 16, "Number of time steps to build up context")
flags.DEFINE_integer("max_steps", None, "Maximum number of steps in episode")

# Evaluation hyerparameters
flags.DEFINE_integer("n_eval_episodes", 0, "Number of episodes to numerically evaluate policy")
flags.DEFINE_integer("n_workers", 1, "Number of parallel workers")
flags.DEFINE_string("eval_save_path", None, "If desired, the path to save the evaluation results")

# Visualization hyperparameters
flags.DEFINE_float("ghost_offset", 0., "Offset for reference ghost")
flags.DEFINE_bool("visualize", True, "Whether to visualize via GUI")

flags.mark_flag_as_required("policy_path")
flags.mark_flag_as_required("clip_snippet")
logging.set_verbosity(logging.WARNING)

class PromptWrapper(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        prompt_policy: base_class.BaseAlgorithm,
        prompt_len: int,
        deterministic_prompt: bool
    ):
        super().__init__(
            venv,
            venv.observation_space,
            venv.action_space
        )
        self._prompt_policy = prompt_policy
        self._prompt_len = prompt_len
        self._deterministic_prompt = deterministic_prompt

    def reset(self):
        self._prompt_counter = np.zeros(self.num_envs, dtype=np.int64)
        self._prompt_state = None
        self._obs = self.venv.reset()
        return self._obs

    def step_async(self, actions: np.ndarray) -> None:
        prompt_actions, self._prompt_state = self._prompt_policy.predict(
            self._obs,
            self._prompt_state,
            self._prompt_counter == 0,
            deterministic=self._deterministic_prompt
        )
        executed_actions = actions.copy()
        prompt_mask = (self._prompt_counter < self._prompt_len)
        executed_actions[prompt_mask] = prompt_actions[prompt_mask]
        self.venv.step_async(executed_actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rew, done, info = self.venv.step_wait()
        self._obs = obs
        self._prompt_counter += 1
        self._prompt_counter[done] = 0
        return obs, rew, done, info


def get_expert(expert_names, clip_id, start_step, end_step):
    expert_name = None
    for name in expert_names:
        id, start, end = name.split('-')
        if id == clip_id and int(start) <= start_step and end_step <= int(end):
            expert_name = name
            break
    assert expert_name is not None

    # Overwrite the start and end steps to ensure proper time step alignment with expert.
    with open(osp.join(FLAGS.expert_root, expert_name, 'clip_info.json')) as f:
        clip_info = json.load(f)
    clip_id, start_step, end_step = [clip_info[s] for s in ['clip_id', 'start_step', 'end_step']]
    dataset = types.ClipCollection(ids=[clip_id], start_steps=[start_step], end_steps=[end_step])

    model_path = osp.join(FLAGS.expert_root, expert_name, 'eval_rsi/model')
    model = sb3_utils.load_policy(model_path, observables.TIME_INDEX_OBSERVABLES)

    return model, dataset

def main(_):
    # set up GPT
    is_in_eval_dir = osp.exists(osp.join(osp.dirname(osp.dirname(osp.dirname(FLAGS.policy_path))), 'model_constructor.txt'))
    model_constructor_path = (osp.join(osp.dirname(osp.dirname(osp.dirname(FLAGS.policy_path))), 'model_constructor.txt')
                              if is_in_eval_dir
                              else osp.join(osp.dirname(osp.dirname(FLAGS.policy_path)), 'model_constructor.txt'))
    with open(model_constructor_path, 'r') as f:
        model_cls = utils.str_to_callable(f.readline())
    policy = model_cls.load_from_checkpoint(FLAGS.policy_path, map_location=FLAGS.device)
    policy.to(FLAGS.device)

    # set up prompt policy
    dataset = utils.make_clip_collection([FLAGS.clip_snippet])
    if FLAGS.expert_root:
        expert_names = os.listdir(FLAGS.expert_root)
        prompt_policy, dataset = get_expert(expert_names, dataset.ids[0], dataset.start_steps[0], dataset.end_steps[0])
        prompt_policy.policy.observation_space = policy.observation_space
    elif FLAGS.distillation_path:
        prompt_policy = NpmpPolicy.load_from_checkpoint(FLAGS.distillation_path, map_location=FLAGS.device)
    else:
        raise ValueError()

    # set up environment
    task_kwargs = dict(
        ghost_offset=np.array([FLAGS.ghost_offset, 0., 0.]),
        always_init_at_clip_start=FLAGS.always_init_at_clip_start,
        termination_error_threshold=FLAGS.termination_error_threshold,
        min_steps=min(FLAGS.min_steps, dataset.end_steps[0]-max(policy.ref_steps))-1,
        max_steps=FLAGS.max_steps,
        steps_before_color_change=FLAGS.prompt_length
    )
    env_kwargs = dict(
        dataset=dataset,
        ref_steps=policy.ref_steps,
        task_kwargs=task_kwargs
    )

    if FLAGS.n_eval_episodes > 0:
        # VecEnv for evaluation
        vec_env = env_util.make_vec_env(
            env_id=motion_completion.MotionCompletionGymEnv,
            n_envs=FLAGS.n_workers,
            seed=FLAGS.seed,
            env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv,
        )
        deterministic_prompt = FLAGS.expert_root is not None # clip expert is deterministic
        vec_env = PromptWrapper(
            vec_env,
            prompt_policy=prompt_policy,
            prompt_len=min(FLAGS.prompt_length, dataset.end_steps[0]),
            deterministic_prompt=deterministic_prompt
        )
        vec_env = VecMonitor(vec_env)

        _, ep_lens = evaluation.evaluate_policy(
            policy,
            vec_env,
            n_eval_episodes=FLAGS.n_eval_episodes,
            deterministic=FLAGS.deterministic,
            return_episode_rewards=True
        )
        print(f"Mean episode length: {np.mean(ep_lens):.1f} +/- {np.std(ep_lens):.1f}")

        if FLAGS.eval_save_path is not None:
            Path(osp.dirname(FLAGS.eval_save_path)).mkdir(parents=True, exist_ok=True)
            np.save(
                FLAGS.eval_save_path,
                ep_lens
            )

    if FLAGS.visualize:
        # env for visualization
        env = motion_completion.MotionCompletionGymEnv(**env_kwargs)

        t, gpt_state, prompt_state = 0, None, None
        @torch.no_grad()
        def policy_fn(time_step):
            nonlocal t, gpt_state, prompt_state
            if time_step.step_type == 0: # first time step
                t, gpt_state, prompt_state = 0, None, None

            obs = env.get_observation(time_step)
            gpt_action, gpt_state = policy.predict(obs, gpt_state, deterministic=FLAGS.deterministic)
            if t < FLAGS.prompt_length:
                deterministic_prompt = FLAGS.expert_root is not None
                prompt_action, prompt_state = prompt_policy.predict(obs, prompt_state, deterministic=deterministic_prompt)
                action = prompt_action
            else:
                action = gpt_action
            t += 1
            return action

        viewer_app = application.Application(title='Motion Completion', width=1024, height=768)
        viewer_app.launch(environment_loader=env.dm_env, policy=policy_fn)

if __name__ == '__main__':
    app.run(main)
