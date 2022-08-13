"""
Callbacks used during policy training.
"""
import os.path as osp
from typing import Sequence, Text
from pathlib import Path
import numpy as np
import imageio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from dm_control.locomotion.tasks.reference_pose import types
from mocapact.envs import env_util
from mocapact.envs import tracking
from mocapact.sb3 import evaluation
from mocapact.sb3 import wrappers
from mocapact import utils

class PolicyEvaluationCallback(Callback):
    """
    A callback that evaluates the policy on a clip collection.
    """
    def __init__(
        self,
        clips: Sequence[Text],
        ref_steps: Sequence[int],
        n_eval_episodes: int,
        eval_freq: int,
        act_noise: float,
        min_steps: int,
        termination_error_threshold: float,
        always_init_at_clip_start: bool,
        n_workers: int,
        seed: int,
        prefix: str,
        log_dir: str,
        run_at_beginning: bool = False,
        serial_evaluation: bool = False,
        record_video: bool = False,
        verbose: int = 0
    ) -> None:
        self._clips = utils.make_clip_collection(clips)
        self._ref_steps = ref_steps
        self._n_eval_episodes = n_eval_episodes
        self._act_noise = act_noise
        self._seed = seed
        self._min_steps = min_steps
        self._always_init_at_clip_start = always_init_at_clip_start
        self._termination_error_threshold = termination_error_threshold
        self._n_workers = n_workers
        self._prefix = prefix
        self._eval_freq = eval_freq
        self._log_dir = log_dir
        self.n_calls = 0
        self._best_reward = float('-inf')
        self._n_call_offset = int(run_at_beginning)
        self._serial_evaluation = serial_evaluation
        self._record_video = record_video
        self.verbose = verbose

        self._steps = []
        self._rewards = []
        self._lengths = []
        self._norm_rewards = []
        self._norm_lengths = []
        self._rewards_per_step = []

    def _create_env(self) -> None:
        task_kwargs = dict(
            reward_type='comic',
            min_steps=self._min_steps-1,
            ghost_offset=np.array([1., 0., 0.]),
            always_init_at_clip_start = self._always_init_at_clip_start,
            termination_error_threshold = self._termination_error_threshold
        )
        env_kwargs = dict(
            dataset=self._clips,
            ref_steps=self._ref_steps,
            act_noise=self._act_noise,
            task_kwargs=task_kwargs
        )
        self._env = env_util.make_vec_env(
            env_id=tracking.MocapTrackingGymEnv,
            n_envs=self._n_workers,
            seed=self._seed,
            env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv if not self._serial_evaluation else DummyVecEnv,
            vec_monitor_cls=wrappers.MocapTrackingVecMonitor
        )

    def on_batch_end(self, trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
        self.n_calls += 1
        if model.global_rank == 0 and self._eval_freq > 0 and (self.n_calls-self._n_call_offset) % self._eval_freq == 0:
            self._create_env()
            self._env.seed(self._seed)
            ep_rews, ep_lens, ep_norm_rews, ep_norm_lens, ep_frames = evaluation.evaluate_locomotion_policy(
                model,
                self._env,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=False,
                return_episode_rewards=True,
                render=self._record_video
            )
            rews_per_step = np.array(ep_rews) / np.array(ep_lens)
            self._steps.append(trainer.global_step)
            self._rewards.append(ep_rews)
            self._lengths.append(ep_lens)
            self._norm_rewards.append(ep_norm_rews)
            self._norm_lengths.append(ep_norm_lens)
            self._rewards_per_step.append(rews_per_step)
            metrics = {
                f"eval/{self._prefix}norm_rew" :     np.mean(ep_norm_rews),
                f"eval/{self._prefix}norm_len" :     np.mean(ep_norm_lens),
                f"eval/{self._prefix}rew_per_step" : np.mean(rews_per_step),
            }
            if self.verbose:
                print(f"{self._prefix.strip('_')} n_calls = {self.n_calls}")
                print(f"Normalized episode reward: {np.mean(ep_norm_rews):.3f} +/- {np.std(ep_norm_rews):.3f}")
                print(f"Normalized episode length: {np.mean(ep_norm_lens):.3f} +/- {np.std(ep_norm_lens):.3f}")
                print(f"Reward per step:           {np.mean(rews_per_step):.3f} +/- {np.std(rews_per_step):.3f}")
            np.savez(
                osp.join(self._log_dir, 'evaluations.npz'),
                steps=self._steps,
                rewards=self._rewards,
                lengths=self._lengths,
                norm_rewards=self._norm_rewards,
                norm_lengths=self._norm_lengths,
                rewards_per_step=self._rewards_per_step
            )
            trainer.logger.log_metrics(metrics, trainer.global_step)
            if metrics[f"eval/{self._prefix}norm_rew"] > self._best_reward:
                self._best_reward = metrics[f"eval/{self._prefix}norm_rew"]
                trainer.save_checkpoint(osp.join(self._log_dir, 'best_model.ckpt'))
                if self._record_video:
                    imageio.mimwrite(osp.join(self._log_dir, 'rollouts.mp4'), ep_frames, fps=30)
            self._env.close() # close environment since it consumes a lot of memory

class LogEmbedCallback(Callback):
    def __init__(self, log_dir: str, n_workers: int, verbose: int = 0):
        super().__init__(verbose)
        self._log_dir = log_dir
        self._embeds = [[] for _ in range(n_workers)]
        self._ctr = 0

    def callback(self, locals, globals):
        del globals
        embeds, done, i = locals['states'], locals['done'], locals['i']
        self.embeds[i].append(embeds[i])
        if done:
            np.save(osp.join(self._log_dir, f"episode_{self._ctr}"), np.stack(self._embeds[i]))
            self._embeds[i] = []
            self._ctr += 1
