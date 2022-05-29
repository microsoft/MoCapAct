import time
import torch
import numpy as np
from typing import Callable
from gym import spaces

from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

class MocapTrackingVecMonitor(VecMonitor):
    """
    A VecMonitor that additionally monitors the normalized episode return and length.
    """
    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])

        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                t_start, t_end = info['start_time_in_clip'], info['time_in_clip']
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_length_norm = (t_end - t_start) / (info['last_time_in_clip'] - t_start)
                episode_return_norm = episode_return / episode_length * episode_length_norm

                episode_info = dict(
                    r=episode_return,
                    r_norm=episode_return_norm,
                    l=episode_length,
                    l_norm=episode_length_norm,
                    t=round(time.time() - self.t_start, 6)
                )
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info['episode'] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos

class EmbedToActionVecWrapper(VecEnvWrapper):
    """
    A VecEnvWrapper that transforms a high-level "embedding" action
    to a low-level action to execute on the considered environment.
    """

    def __init__(
        self,
        venv: VecEnv,
        embed_dim: int,
        max_embed: float,
        embed_to_action: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ):
        action_space = spaces.Box(-max_embed, max_embed, (embed_dim,))
        super().__init__(venv, action_space=action_space)
        self.embed_to_action = embed_to_action

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        obs = {k: v.astype(np.float32) for k, v in obs.items()}
        self.obs = obs_as_tensor(obs, 'cpu')
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        acts = torch.tensor(actions)
        with torch.no_grad():
            low_level_actions = self.embed_to_action(self.obs, acts).numpy()
        low_level_actions = np.clip(low_level_actions, self.venv.action_space.low, self.venv.action_space.high)
        self.venv.step_async(low_level_actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rew, done, info = self.venv.step_wait()
        obs = {k: v.astype(np.float32) for k, v in obs.items()}
        self.obs = obs_as_tensor(obs, 'cpu')
        return obs, rew, done, info
