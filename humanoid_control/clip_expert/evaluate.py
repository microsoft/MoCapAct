import os.path as osp
import pickle
import zipfile
import numpy as np
import torch
from absl import app, flags, logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from dm_control.viewer import application
from dm_control.locomotion.tasks.reference_pose import types
from humanoid_control import observables
from humanoid_control import utils
from humanoid_control.sb3 import env_util
from humanoid_control.sb3 import evaluation
from humanoid_control.sb3 import features_extractor
from humanoid_control.sb3 import tracking
from humanoid_control.sb3 import wrappers

MIN_STEPS = 10

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_root", None, "Directory where experiment results are stored")
flags.DEFINE_float("act_noise", 0.1, "Action noise in humanoid")

# Visualization hyperparameters
flags.DEFINE_bool("visualize", True, "Whether to visualize via GUI")
flags.DEFINE_float("ghost_offset", 0., "Offset for reference ghost")

# Evaluation hyperparameters
flags.DEFINE_integer("n_eval_episodes", 0, "Number of episodes to numerically evaluate policy")
flags.DEFINE_integer("n_workers", 8, "Number of parallel workers")
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

    # Extract model zip file
    with zipfile.ZipFile(osp.join(model_path, 'best_model.zip')) as zip_ref:
        zip_ref.extractall(model_path)

    flags = utils.load_absl_flags(osp.join(FLAGS.exp_root, 'flags.txt'))

    # Make environment
    clip_length = utils.get_clip_length(flags['clip_id'])
    snippet_length = min(clip_length-int(flags['start_step']), int(flags['max_steps']))
    end_step = int(flags['start_step']) + snippet_length
    dataset = types.ClipCollection(
        ids=[flags['clip_id']],
        start_steps=[int(flags['start_step'])],
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
    features_extractor_class = lambda space: features_extractor.CmuHumanoidFeaturesExtractor(
        space,
        observable_keys=observables.TIME_INDEX_OBSERVABLES,
        obs_rms=obs_stats,
    )
    layer_sizes = int(flags['n_layers']) * [int(flags['layer_size'])]
    activation_fns = dict(relu=torch.nn.ReLU, tanh=torch.nn.Tanh, elu=torch.nn.ELU)
    activation_fn = activation_fns.get(flags['activation_fn'], torch.nn.ReLU)
    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        net_arch=[dict(pi=layer_sizes, vf=layer_sizes)],
        activation_fn=activation_fn
    )
    model = PPO("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1, device='cpu')
    params = torch.load(osp.join(model_path, 'policy.pth'))
    model.policy.load_state_dict(params)

    if FLAGS.n_eval_episodes > 0:
        ep_rews, ep_lens, ep_norm_rews, ep_norm_lens = evaluation.evaluate_locomotion_policy(
            model,
            vec_env,
            n_eval_episodes=FLAGS.n_eval_episodes,
            deterministic=True,
            return_episode_rewards=True
        )
        print("Mean return: %.1f +/- %.1f" % (np.mean(ep_rews), np.std(ep_rews)))
        print("Mean episode length: %.1f +/- %.1f" % (np.mean(ep_lens), np.std(ep_lens)))
        print("Mean normalized return: %.3f +/- %.3f" % (np.mean(ep_norm_rews), np.std(ep_norm_rews)))
        print("Mean normalized episode length: %.3f +/- %.3f" % (np.mean(ep_norm_lens), np.std(ep_norm_lens)))

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
