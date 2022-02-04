import time
import numpy as np
import gym
from gym import spaces
from copy import deepcopy
from dataclasses import dataclass, field

from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

######
# Gym
######
@dataclass
class EmbeddingRegularization:
    """
    For defining the KL regularization of the embedding.
    """
    weight: float
    correlation: float
    std_dev: float = field(init=False)

    def __post_init__(self):
        assert 0 <= self.correlation <= 1
        self.std_dev = np.sqrt(1 - self.correlation**2)

class Embedding(gym.Wrapper):
    """
    Includes a policy's embedding as an action and observation.
    """
    def __init__(
        self,
        env: gym.Env,
        embed_dim: int,
        embed_max: float = float('inf')
    ):
        super().__init__(env)
        self.embed_dim = embed_dim
        self.obs_key = 'embedding'

        self.observation_space = deepcopy(env.observation_space)
        self.observation_space[self.obs_key] = spaces.Box(
            low=float('-inf'),
            high=float('-inf'),
            shape=(np.maximum(embed_dim, 1),),
            dtype=np.float32
        )

        embed_act_max = np.full(embed_dim, embed_max, dtype=env.action_space.dtype)
        low = np.concatenate([env.action_space.low, -embed_act_max])
        high = np.concatenate([env.action_space.high, embed_act_max])
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )

    def reset(self, **kwargs):
        obs = self.env.reset()
        embed = self.np_random.randn(self.embed_dim).astype(np.float32) if self.embed_dim > 0 else np.array([0], dtype=np.float32)
        obs[self.obs_key] = embed #TODO: clip?
        return obs

    def step(self, action: np.ndarray):
        if self.embed_dim > 0:
            act, embed = np.split(action, [-self.embed_dim])
        else:
            act, embed = action, np.array([0], dtype=np.float32)
        obs, rew, done, info = self.env.step(act)
        obs[self.obs_key] = embed
        return obs, rew, done, info

#####################
# Stable-Baselines 3 
#####################
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
