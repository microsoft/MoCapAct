from xml.etree.ElementInclude import include
import numpy as np
import gym
from gym import spaces
from copy import deepcopy

from humanoid_control.observables import EMBEDDING_OBSERVABLE


class Embedding(gym.Wrapper):
    """
    Includes a policy's embedding as an action and observation.
    """
    def __init__(
        self,
        env: gym.Env,
        embed_dim: int,
        embed_std: float = 1.,
        embed_max: float = float('inf'),
        include_embed_in_obs: bool = False,
        include_embed_in_act: bool = False
    ):
        super().__init__(env)
        self.embed_dim = embed_dim
        self._last_embed = None

        self._embed_space = spaces.Box(
            low=-embed_max,
            high=embed_max,
            shape=(max(embed_dim, 1),),
            dtype=np.float32
        )
        self._embed_std = embed_std
        self._include_embed_in_obs = include_embed_in_obs
        self._include_embed_in_act = include_embed_in_act

        if self._include_embed_in_obs:
            self.observation_space = deepcopy(env.observation_space)
            self.observation_space.spaces[EMBEDDING_OBSERVABLE] = self._embed_space
        else:
            self.observation_space = env.observation_space

        if self._include_embed_in_act:
            embed_act_max = np.full(embed_dim, embed_max, dtype=env.action_space.dtype)
            low = np.concatenate([env.action_space.low, -embed_act_max])
            high = np.concatenate([env.action_space.high, embed_act_max])
            self.action_space = spaces.Box(
                low=low,
                high=high,
                shape=low.shape,
                dtype=low.dtype
            )
        else:
            self.action_space = env.action_space

    def reset(self, **kwargs):
        obs = self.env.reset()
        if self._include_embed_in_obs:
            embed = self._embed_std*self.np_random.randn(*self._embed_space.shape).astype(self._embed_space.dtype)
            embed = embed.clip(self._embed_space.low, self._embed_space.high)
            obs[EMBEDDING_OBSERVABLE] = self._last_embed = embed
        return obs

    def step(self, action: np.ndarray):
        if self._include_embed_in_act:
            if self.embed_dim > 0:
                act, embed = np.split(action, [-self.embed_dim])
            else:
                act, embed = action, np.array([0], dtype=np.float32)
        else:
            act = action
            embed = self._embed_std*self.np_random.randn(*self._embed_space.shape).astype(self._embed_space.dtype)
            embed = embed.clip(self._embed_space.low, self._embed_space.high)
        obs, rew, done, info = self.env.step(act)
        info['prev_embed'] = self._last_embed.copy()
        self._last_embed = embed
        if self._include_embed_in_obs:
            obs[EMBEDDING_OBSERVABLE] = embed
        return obs, rew, done, info
