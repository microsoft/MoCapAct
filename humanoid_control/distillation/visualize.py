import os.path as osp
import glob
import zipfile
import pickle
import json
import numpy as np
from absl import app, flags, logging
from stable_baselines3 import PPO
import torch

from dm_control.locomotion.tasks.reference_pose import types
from dm_control.viewer import application

from humanoid_control import observables
from humanoid_control import utils
from humanoid_control.sb3 import env_util
from humanoid_control.sb3 import evaluation
from humanoid_control.sb3 import wrappers
from humanoid_control.sb3 import features_extractor
from humanoid_control.sb3 import tracking

FLAGS = flags.FLAGS
flags.DEFINE_string("policy_path", None, "Path where saved policy is stored")
flags.DEFINE_list("clip_snippets", None, "A list of clip snippets, of form CMU_{clip id}[-{start step}-{end step}")
flags.DEFINE_float("act_noise", 0.1, "Action noise in humanoid")

# Evaluation hyperparameters
flags.DEFINE_bool("always_init_at_clip_start", False, "Whether to initialize at beginning or random point in clip")
flags.DEFINE_bool("deterministic", False, "Whether the policy is deterministic")
flags.DEFINE_float("termination_error_threshold", 0.3, "Error for cutting off rollout")
flags.DEFINE_integer("min_steps", 10, "Minimum steps left at end of episode")
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_integer("warmup_steps", 16, "Number of time steps to build up context")
flags.DEFINE_integer("max_steps", None, "Maximum number of steps in episode")
flags.DEFINE_string("expert_root", None, "Path to expert used to generate context")
flags.DEFINE_string("device", "cpu", "Device to do rollouts on")

# Visualization hyperparameters
flags.DEFINE_float("ghost_offset", 0., "Offset for reference ghost")

flags.mark_flag_as_required("policy_path")
flags.mark_flag_as_required("clip_snippets")
logging.set_verbosity(logging.WARNING)

def make_clips(clip_snippets):
    ids, start_steps, end_steps = [], [], []
    for clip_snippet in clip_snippets:
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

def get_expert_paths(input_dir):
    """
    For each clip in the input directories, gets the path of the expert.
    """
    clips = set()
    expert_paths, expert_metrics = {}, {}
    experiment_paths = [osp.dirname(path) for path in glob.iglob(f"{input_dir}/**/flags.txt", recursive=True)]
    for path in experiment_paths:
        with open(osp.join(path, 'clip_info.json')) as f:
            clip_info = json.load(f)
        clip_id = clip_info['clip_id']
        start_step = clip_info['start_step']
        end_step = clip_info['end_step']
        expert_name = f"{clip_id}-{start_step}-{end_step}"
        if osp.exists(osp.join(path, 'eval_random/evaluations.npz')):
            try:
                eval_npz = np.load(osp.join(path, 'eval_random/evaluations.npz'))
            except:
                continue
            clips.add(clip_id)
            idx = eval_npz['results'].mean(1).argmax()
            ret = eval_npz['results'][idx].mean()
            if expert_name not in expert_paths or ret > expert_metrics[expert_name]['ep_return'].mean():
                expert_paths[expert_name] = path
                expert_metrics[expert_name] = dict(
                    ep_return=eval_npz['results'][idx],
                    ep_length=eval_npz['ep_lengths'][idx],
                    ep_norm_return=eval_npz['results_norm'][idx],
                    ep_norm_length=eval_npz['ep_lengths_norm'][idx]
                    )
    return expert_paths, expert_metrics, clips


def get_expert(expert_paths, clip_id, start_step):
    expert_path = None
    for k in expert_paths.keys():
        id, start, end = k.split('-')
        if id == clip_id and int(start) <= start_step < int(end):
            expert_path = expert_paths[k]
    assert expert_path is not None

    model_path = osp.join(expert_path, 'eval_start/model')
    # Normalization statistics
    with open(osp.join(model_path, 'vecnormalize.pkl'), 'rb') as f:
        norm_env = pickle.load(f)
        obs_stats = norm_env.obs_rms

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
        custom_objects=dict(policy_kwargs=policy_kwargs, learning_rate=0., clip_range=0.),
        device=FLAGS.device
    )

    return model


def main(_):
    clips = make_clips(FLAGS.clip_snippets)
    expert_paths, *_ = get_expert_paths(FLAGS.expert_root)

    # set up model
    with open(osp.join(osp.dirname(osp.dirname(osp.dirname(FLAGS.policy_path))), 'model_constructor.txt'), 'r') as f:
    #with open(osp.join(osp.dirname(FLAGS.policy_path), 'model_constructor.txt'), 'r') as f:
        model_cls = utils.str_to_callable(f.readline())
    policy = model_cls.load_from_checkpoint(FLAGS.policy_path, map_location='cpu')
    policy.to(FLAGS.device)
    policy.observation_space['walker/clip_id'].n = int(1e6)

    # set up environment
    task_kwargs = dict(
        reward_type='comic',
        min_steps=FLAGS.min_steps-1,
        ghost_offset=np.array([FLAGS.ghost_offset, 0., 0.]),
        always_init_at_clip_start=FLAGS.always_init_at_clip_start,
        termination_error_threshold=FLAGS.termination_error_threshold,
        max_steps_override=FLAGS.max_steps
    )
    env_kwargs = dict(
        dataset=clips,
        ref_steps=policy.ref_steps,
        act_noise=FLAGS.act_noise,
        task_kwargs=task_kwargs
    )

    # env for visualization
    env = tracking.MocapTrackingGymEnv(**env_kwargs)
    t, expert, state = 0, None, None
    @torch.no_grad()
    def policy_fn(time_step):
        nonlocal t, expert, state
        if time_step.step_type == 0: # first time step
            state, t = None, 0
            clip_id = env.task._dataset.ids[env.task._current_clip_index]
            start_step = env.task._dataset.start_steps[env.task._current_clip_index]
            expert = get_expert(expert_paths, clip_id, start_step)
        action, state = policy.predict(env._get_obs(time_step), state, deterministic=FLAGS.deterministic)
        if t < FLAGS.warmup_steps:
            action, _ = expert.predict(env._get_obs(time_step), None, deterministic=FLAGS.deterministic)
        t += 1
        return action

    viewer_app = application.Application(title='Explorer', width=1024, height=768)
    viewer_app.launch(environment_loader=env._env, policy=policy_fn)

if __name__ == '__main__':
    app.run(main)
