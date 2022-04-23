import os.path as osp
import pickle
from sre_parse import State
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
from stable_baselines3.common import env_util
from stable_baselines3.common import evaluation
from humanoid_control.sb3 import features_extractor
from humanoid_control.tasks import stand
from humanoid_control.envs import wrappers
from humanoid_control.joint import model

from humanoid_control.joint.ppo import PPOBC
from humanoid_control.joint.a2c import A2CBC

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
        'eval',
        'model'
    )

    # Extract model zip file
    with zipfile.ZipFile(osp.join(model_path, 'best_model.zip')) as zip_ref:
        zip_ref.extractall(model_path)

    # Make environment
    # VecEnv for evaluation
    vec_env = env_util.make_vec_env(
        env_id=stand.StandUpGymEnv,
        n_envs=FLAGS.n_workers,
        seed=FLAGS.seed,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=wrappers.Embedding,
        wrapper_kwargs=(dict(embed_dim=20))
    )

    # env for visualization
    env = wrappers.Embedding(stand.StandUpGymEnv(), embed_dim=20)

    # Normalization statistics
    with open(osp.join(model_path, 'vecnormalize.pkl'), 'rb') as f:
        norm_env = pickle.load(f)
        obs_stats = norm_env.obs_rms

    # Set up model
    #model = A2CBC.load(osp.join(model_path, 'best_model.zip'))
    model = PPOBC.load(osp.join(model_path, 'best_model.zip'))
    keys = dict(model.policy.features_extractor._keys)
    del keys['ref_encoder']
    model.policy.features_extractor = features_extractor.CmuHumanoidFeaturesExtractor(
        model.policy.features_extractor._observation_space,
        keys,
        obs_stats
    )

    if FLAGS.n_eval_episodes > 0:
        ep_rews, ep_lens = evaluation.evaluate_policy(
            model,
            vec_env,
            n_eval_episodes=FLAGS.n_eval_episodes,
            deterministic=False,
            return_episode_rewards=True
        )
        print("Mean return: %.1f +/- %.1f" % (np.mean(ep_rews), np.std(ep_rews)))
        print("Mean episode length: %.1f +/- %.1f" % (np.mean(ep_lens), np.std(ep_lens)))

        if FLAGS.save_path is not None:
            np.savez(
                osp.join(FLAGS.save_path, flags['clip_id']),
                ep_rews=ep_rews,
                ep_lens=ep_lens,
            )

    embed = None
    @torch.no_grad()
    def policy_fn(time_step):
        nonlocal embed
        if time_step.step_type == 0:
            embed = env.np_random.randn(20).astype(np.float32)
        obs = env.env._get_obs(time_step)
        obs['embedding'] = embed
        action, _ = model.predict(obs, deterministic=False)
        embed = action[-20:]
        action = action[:-20]
        return action

    if FLAGS.visualize:
        viewer_app = application.Application(title='Explorer', width=1024, height=768)
        viewer_app.launch(environment_loader=env.env._env, policy=policy_fn)

if __name__ == '__main__':
    app.run(main)
