"""
The features extractor that maps the raw observation from the environment (dict of numerical arrays)
to an observation that is passed to the policy's `predict` function (can be numerical array
or dict of numerical arrays). Also handles observation normalization when doing policy evaluation.
"""
from typing import Dict, Optional, Sequence, Text, Tuple, Union
import collections
import numpy as np
import torch
import gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.running_mean_std import RunningMeanStd

from mocapact import observables

class CmuHumanoidFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        observable_keys: Union[Sequence[Text], Dict[Text, Sequence[Text]]] = observables.CMU_HUMANOID_OBSERVABLES,
        obs_rms: Optional[RunningMeanStd] = None
    ):
        self._observation_space = observation_space
        self._keys = observable_keys
        self._normalize = obs_rms is not None
        self._dict_observables = not isinstance(observable_keys, collections.abc.Sequence)
        if self._dict_observables:
            self._extractors = {
                k: CmuHumanoidFeaturesExtractor(observation_space, obs_keys, obs_rms)
                for k, obs_keys in observable_keys.items()
            }
            self.sub_features_dim = {
                k: extractor.features_dim for k, extractor in self._extractors.items()
            }
            features_dim = sum(self.sub_features_dim.values())
        else:
            features_dim = sum([observation_space.spaces[k].shape[0] for k in self._keys])
            if self._normalize:
                means, stds = [], []
                for k in self._keys:
                    if k == 'walker/clip_id': # don't normalize
                        means.append(np.array([0.]))
                        stds.append(np.array([1.]))
                    elif k == 'embedding': # don't normalize
                        means.append(np.zeros(observation_space['embedding'].shape))
                        stds.append(np.ones(observation_space['embedding'].shape))
                    else:
                        means.append(obs_rms[k].mean)
                        stds.append(np.sqrt(obs_rms[k].var + 1e-8))
                self.mean = torch.tensor(np.concatenate(means), dtype=torch.float32)
                self.std = torch.tensor(np.concatenate(stds), dtype=torch.float32)
        super().__init__(observation_space, features_dim=features_dim)

    def forward(self, observations):
        if self._dict_observables:
            obs = {
                k: extractor.forward(observations) for k, extractor in self._extractors.items()
            }
        else:
            obs = []
            for k in self._keys:
                if k in observations:
                    obs.append(observations[k])
                else:
                    tmp = list(observations.values())[0]
                    shape = list(tmp.shape)
                    shape[-1] = self._observation_space[k].shape[0]
                    obs.append(torch.full(shape, torch.nan, device=tmp.device))
            obs = torch.cat(obs, dim=-1)
            if self._normalize:
                if self.mean.device != obs.device:
                    self.mean = self.mean.to(obs.device)
                    self.std = self.std.to(obs.device)
                return (obs - self.mean)/self.std
        return obs
