"""
Callbacks used when training the clip experts.
"""
import os
import os.path as osp
import numpy as np
from typing import Union, Optional, Callable
import gym
import imageio

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

from mocapact.sb3 import evaluation

class NormalizedRolloutCallback(BaseCallback):
    """
    Also logs the rollout episode reward and length which are normalized by the mocap
    clip length.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        log_interval, iteration = self.locals['log_interval'], self.locals['iteration']
        if log_interval is not None and iteration % log_interval == 0:
            if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
                r_norm_mean = safe_mean([ep_info['r_norm'] for ep_info in self.model.ep_info_buffer])
                l_norm_mean = safe_mean([ep_info['l_norm'] for ep_info in self.model.ep_info_buffer])
                self.logger.record("rollout/ep_rew_mean_norm", r_norm_mean)
                self.logger.record("rollout/ep_len_mean_norm", l_norm_mean)

class EarlyStoppingCallback(BaseCallback):
    """
    Stops the training if the evaluation episode length is "large enough" and the
    episode return has not improved enough for a period of time.
    """
    def __init__(
        self,
        norm_ep_len_threshold: float,
        min_norm_reward_delta: float,
        patience: int = 1,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.norm_ep_len_threshold = norm_ep_len_threshold
        self.min_norm_reward_delta = min_norm_reward_delta
        self.patience = patience

        self.best_norm_reward = float('-inf')
        self.wait = 0

    def _on_step(self) -> bool:
        if self.parent.last_mean_ep_length_norm < self.norm_ep_len_threshold:
            self.wait = 0
            return True

        if self.parent.last_mean_reward_norm >= self.best_norm_reward + self.min_norm_reward_delta:
            self.best_norm_reward = self.parent.last_mean_reward_norm
            if self.verbose > 0:
                print(f"New best reward for early stopping! {self.best_norm_reward:.5f}")
            self.wait = 0
        else:
            self.wait += 1

        if self.wait > self.patience:
            if self.verbose > 0:
                print("Stopping early!")
            return False

        return True

class MocapTrackingEvalCallback(EvalCallback):
    """
    An `EvalCallback` that additionally logs normalized episode rewards and lengths.
    """
    def __init__(
        self,
        eval_env_ctor: Callable[[], Union[gym.Env, VecEnv]],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        record_video: bool = False,
        verbose: int = 1,
        name: str = "eval",
        warn: bool = True
    ):
        self.eval_env_ctor = eval_env_ctor
        eval_env = eval_env_ctor()
        super().__init__(eval_env, callback_on_new_best=callback_on_new_best,
                         callback_after_eval=callback_after_eval,
                         n_eval_episodes=n_eval_episodes, eval_freq=eval_freq,
                         log_path=log_path, best_model_save_path=best_model_save_path,
                         deterministic=deterministic, render=False,
                         verbose=verbose, warn=warn)
        self.name = name
        self.evaluations_results_norm = []
        self.evaluations_length_norm = []
        self.record_video = record_video

    def _on_step(self) -> bool:
        """
        Additionally logs normalized reward and episode length.
        """

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            self.eval_env.seed(0)
            results = evaluation.evaluate_locomotion_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=self.record_video,
                return_episode_rewards=True,
                warn=self.warn,
            )
            episode_rewards, episode_lengths, episode_rewards_norm, episode_lengths_norm, episode_frames = results
            self.eval_env.close()
            self.eval_env = self.eval_env_ctor()

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                self.evaluations_results_norm.append(episode_rewards_norm)
                self.evaluations_length_norm.append(episode_lengths_norm)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    results_norm=self.evaluations_results_norm,
                    ep_lengths_norm=self.evaluations_length_norm,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            mean_reward_norm, std_reward_norm = np.mean(episode_rewards_norm), np.std(episode_rewards_norm)
            mean_ep_length_norm, std_ep_length_norm = np.mean(episode_lengths_norm), np.std(episode_lengths_norm)
            self.last_mean_reward = mean_reward
            self.last_mean_ep_length = mean_ep_length
            self.last_mean_reward_norm = mean_reward_norm
            self.last_mean_ep_length_norm = mean_ep_length_norm

            if self.verbose > 0:
                print(f"{self.name} num_timesteps={self.num_timesteps}, "
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                print(f"Normalized episode reward: {mean_reward_norm:.3f} +/- {std_reward_norm:.3f}")
                print(f"Normalized episode length: {mean_ep_length_norm:.3f} +/- {std_ep_length_norm:.3f}")
            # Add to current Logger
            self.logger.record(f"{self.name}/mean_reward", float(mean_reward))
            self.logger.record(f"{self.name}/mean_ep_length", mean_ep_length)

            # Additionally log normalized reward and length
            self.logger.record(f"{self.name}/mean_reward_norm", float(mean_reward_norm))
            self.logger.record(f"{self.name}/mean_ep_length_norm", mean_ep_length_norm)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                if self.record_video:
                    imageio.mimwrite(osp.join(self.log_path, "rollouts.mp4"), episode_frames, fps=30)
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            return continue_training

        return True
