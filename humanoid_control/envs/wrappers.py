import numpy as np
import gym
from gym import spaces
from copy import deepcopy


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
        self._obs_key = 'embedding'
        self._last_embed = None

        self._embed_space = spaces.Box(
            low=-embed_max,
            high=embed_max,
            shape=(np.maximum(embed_dim, 1),),
            dtype=np.float32
        )
        self.observation_space = deepcopy(env.observation_space)
        self.observation_space.spaces[self._obs_key] = self._embed_space

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
        embed = self.np_random.randn(*self._embed_space.shape).astype(self._embed_space.dtype)
        embed = embed.clip(self._embed_space.low, self._embed_space.high)
        obs[self._obs_key] = self._last_embed = embed
        return obs

    def step(self, action: np.ndarray):
        if self.embed_dim > 0:
            act, embed = np.split(action, [-self.embed_dim])
        else:
            act, embed = action, np.array([0], dtype=np.float32)
        obs, rew, done, info = self.env.step(act)
        info['prev_embed'] = self._last_embed.copy()
        obs[self._obs_key] = self._last_embed = embed
        return obs, rew, done, info
