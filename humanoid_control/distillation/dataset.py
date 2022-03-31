import os
import bisect
import h5py
import itertools
import collections
import numpy as np
from typing import Dict, List, Sequence, Text, Tuple, Optional, Union
from gym import spaces
from pandas import concat
from torch.utils.data import Dataset
from stable_baselines3.common.running_mean_std import RunningMeanStd
from humanoid_control import observables

def weighted_average(arrays, weights):
    total = 0
    for array, weight in zip(arrays, weights):
        total += weight * array
    return total / sum(weights)

class ExpertDataset(Dataset):
    def __init__(
        self,
        h5py_fnames: Sequence[Text],
        observables: Union[Sequence[Text], Dict[Text, Sequence[Text]]],
        clip_ids: Optional[Sequence[Text]] = None,
        min_seq_steps: int = 1,
        max_seq_steps: int = 1,
        normalize_obs: bool = False,
        normalize_act: bool = False,
        concat_observables: bool = True,
        preload: bool = False,
        clip_centric_weight: bool = False,
        advantage_weights: bool = True,
        temperature: Optional[float] = None,
        max_weight: float = 20
    ):
        self._dsets = [h5py.File(fname, 'r') for fname in h5py_fnames]
        self._observables = observables

        self._clip_ids = []
        for dset in self._dsets:
            if clip_ids is None:
                self._clip_ids.append(tuple([k for k in dset.keys() if k.startswith('CMU')]))
            else:
                self._clip_ids.append(tuple([k for k in clip_ids if k in dset.keys()]))
        self._all_clip_ids = tuple(itertools.chain.from_iterable(self._clip_ids))
        self._unique_clip_ids = tuple([k.split('-')[0] for k in self._all_clip_ids])
        self._min_seq_steps = min_seq_steps
        self._max_seq_steps = max_seq_steps
        self._concat_observables = concat_observables
        self._preload = preload
        self._ref_steps = self._dsets[0]['ref_steps'][...]
        self._clip_centric_weight = clip_centric_weight
        self._advantage_weights = advantage_weights
        self._temperature = temperature
        self._max_weight = max_weight
        self._normalize_obs = normalize_obs
        self._normalize_act = normalize_act

        self._set_spaces()
        self._set_stats()

        self._create_offsets()
        if self._preload:
            self._preload_dataset()

    @property
    def all_clip_ids(self):
        return self._all_clip_ids

    @property
    def is_sequential(self):
        return not (self._min_seq_steps == self._max_seq_steps == 1)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def full_observation_space(self):
        return self._full_observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def ref_steps(self):
        return self._ref_steps

    @property
    def obs_mean(self):
        return self._obs_mean

    @property
    def obs_var(self):
        return self._obs_var

    @property
    def obs_std(self):
        return self._obs_std

    @property
    def act_mean(self):
        return self._act_mean

    @property
    def act_var(self):
        return self._act_var

    @property
    def act_std(self):
        return self._act_std

    @property
    def observable_indices(self):
        return self._dsets[0]['observable_indices']

    @property
    def obs_rms(self):
        return self._obs_rms

    def _set_spaces(self):
        # Observation space for all observables in the dataset
        obs_ind_dset = self.observable_indices['walker']
        obs_spaces = {
            f"walker/{k}": spaces.Box(-np.infty, np.infty, shape=(len(obs_ind_dset[k]),), dtype=np.float32)
            for k in obs_ind_dset
        }
        obs_spaces['walker/clip_id'] = spaces.Discrete(len(self._unique_clip_ids))
        self._full_observation_space = spaces.Dict(obs_spaces)

        # Observation space for the observables we're considering
        if not isinstance(self._observables, collections.abc.Sequence):  # observables is Dict[Text, Sequence[Text]]
            if self._concat_observables:
                observation_indices = {
                    k: np.concatenate([self.observable_indices[observable][...] for observable in subobservables])
                    for k, subobservables in self._observables.items()
                }
                self._observation_space = spaces.Dict({
                    k: spaces.Box(
                        low=-np.infty,
                        high=np.infty,
                        shape=observation_indices.shape,
                        dtype=np.float32
                    )
                    for k, observation_indices in observation_indices.items()
                })
            else:
                self._observation_space = spaces.Dict({
                    k: spaces.Dict({observable: spaces.Box(
                        low=-np.infty,
                        high=np.infty,
                        shape=self.observable_indices[observable].shape,
                        dtype=np.float32)
                        for observable in subobservables})
                    for k, subobservables in self._observables.items()
                })
        else: # observables is Sequence[Text]
            if self._concat_observables:
                observation_indices = np.concatenate(
                    [self.observable_indices[observable] for observable in self._observables]
                )
                self._observation_space = spaces.Box(
                    low=-np.infty,
                    high=np.infty,
                    shape=observation_indices.shape
                )
            else:
                self._observation_space = spaces.Dict({
                    observable: spaces.Box(
                        low=-np.infty,
                        high=np.infty,
                        shape=self.observable_indices[observable].shape,
                        dtype=np.float32
                    )
                    for observable in self._observables
                })

        # Action space
        self._action_space = spaces.Box(
            low=np.float32(-1.),
            high=np.float32(1.),
            shape=(self._dsets[0][f"{self._clip_ids[0][0]}/0/actions"].shape[1],),
            dtype=np.float32
        )

    def _set_stats(self):
        counts = [dset['stats/count'][...] for dset in self._dsets]
        means = [dset['stats/obs_mean'][...] for dset in self._dsets]
        vars = [dset['stats/obs_var'][...] for dset in self._dsets]
        second_moments = [var + mean**2 for var, mean in zip(means, vars)]
        self._obs_mean = weighted_average(means, counts).astype(np.float32)
        self._obs_var = np.clip(weighted_average(second_moments, counts) - self._obs_mean**2, 0., None).astype(np.float32)
        self._obs_std = (np.sqrt(self._obs_var) + 1e-4).astype(np.float32)

        means = [dset['stats/act_mean'][...] for dset in self._dsets]
        vars = [dset['stats/act_var'][...] for dset in self._dsets]
        second_moments = [var + mean**2 for var, mean in zip(means, vars)]
        self._act_mean = weighted_average(means, counts).astype(np.float32)
        self._act_var = np.clip(weighted_average(second_moments, counts) - self._act_mean**2, 0., None).astype(np.float32)
        self._act_std = (np.sqrt(self._act_var) + 1e-4).astype(np.float32)

        count = np.sum(counts)
        self._obs_rms = {}
        for k in observables.MULTI_CLIP_OBSERVABLES_SANS_ID:
            obs_rms = RunningMeanStd()
            obs_rms.mean = self.obs_mean[self.observable_indices[k][...]]
            obs_rms.var = self.obs_var[self.observable_indices[k][...]]
            obs_rms.count = count
            self._obs_rms[k] = obs_rms

    def _create_offsets(self):
        self._total_len = 0
        self._dset_indices = []
        self._logical_indices, self._dset_groups = [[] for _ in self._dsets], [[] for _ in self._dsets]
        self._clip_returns = [{} for _ in self._dsets]
        iterator = zip(self._dsets, self._clip_ids, self._logical_indices, self._dset_groups, self._clip_returns)
        for dset, clip_ids, logical_indices, dset_groups, clip_return in iterator:
            self._dset_indices.append(self._total_len)
            for clip_id in clip_ids:
                ret_iterator = itertools.chain(dset[f"{clip_id}/start_metrics/norm_episode_returns"],
                                               dset[f"{clip_id}/rsi_metrics/norm_episode_returns"])
                returns = [x for x in ret_iterator]
                clip_return[clip_id] = np.mean(returns)

                len_iterator = itertools.chain(dset[f"{clip_id}/start_metrics/episode_lengths"],
                                               dset[f"{clip_id}/rsi_metrics/episode_lengths"])
                for i, ep_len in enumerate(len_iterator):
                    logical_indices.append(self._total_len)
                    dset_groups.append(f"{clip_id}/{i}")
                    if ep_len < self._min_seq_steps:
                        continue
                    self._total_len += ep_len - (self._min_seq_steps - 1)
        self._avg_return = np.mean(list(itertools.chain(*[d.values() for d in self._clip_returns])))

    def _preload_dataset(self):
        self._obs_dsets, self._act_dsets, self._rew_dsets = [[] for _ in self._dsets], [[] for _ in self._dsets], [[] for _ in self._dsets]
        iterator = zip(self._dsets, self._clip_ids, self._obs_dsets, self._act_dsets, self._rew_dsets)
        for dset, clip_ids, obs_dset, act_dset, rew_dset in iterator:
            for clip_id in clip_ids:
                for i in range(len(dset[f"{clip_id}/episode_lengths"])):
                    obs_dset.append(dset[f"{clip_id}/{i}/observations"][...])
                    act_dset.append(dset[f"{clip_id}/{i}/actions"][...])
                    rew_dset.append(dset[f"{clip_id}/{i}/rewards"][...])

    def _extract_observations(self, all_obs: np.ndarray, observable_keys: Sequence[Text]):
        return {k: all_obs[..., self.observable_indices[k][...]] for k in observable_keys}

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        """
        TODO
        """
        dset_idx = bisect.bisect_right(self._dset_indices, idx)-1
        clip_idx = bisect.bisect_right(self._logical_indices[dset_idx], idx)-1

        if self._preload:
            obs_dset = self._obs_dsets[dset_idx][clip_idx]
            act_dset = self._act_dsets[dset_idx][clip_idx]
        else:
            obs_dset = self._dsets[dset_idx][f"{self._dset_groups[dset_idx][clip_idx]}/observations"]
            act_dset = self._dsets[dset_idx][f"{self._dset_groups[dset_idx][clip_idx]}/actions"]
        val_dset = self._dsets[dset_idx][f"{self._dset_groups[dset_idx][clip_idx]}/values"]
        adv_dset = self._dsets[dset_idx][f"{self._dset_groups[dset_idx][clip_idx]}/advantages"]

        if self.is_sequential:
            start_idx = idx - self._logical_indices[dset_idx][clip_idx]
            end_idx = min(start_idx + self._max_seq_steps, act_dset.shape[0]+1)
            all_obs = obs_dset[start_idx:end_idx]
            act = act_dset[start_idx:end_idx]
        else:
            rel_idx = idx - self._logical_indices[dset_idx][clip_idx]
            all_obs = obs_dset[rel_idx]
            act = act_dset[rel_idx]

        if self._normalize_obs:
            all_obs = (all_obs - self.obs_mean) / self.obs_std
        if self._normalize_act:
            act = (act - self.act_mean) / self.act_std
        
        # Extract observation
        if isinstance(self._observables, dict):
            obs = {
                k: self._extract_observations(all_obs, observable_keys)
                for k, observable_keys in self._observables.items()
            }
            if self._concat_observables:
                obs = {k: np.concatenate(list(v.values()), axis=-1) for k, v in obs.items()}
        else:
            obs = self._extract_observations(all_obs, self._observables)
            if self._concat_observables:
                obs = np.concatenate(list(obs.values()), axis=-1)

        if self._temperature is None:
            weight = np.ones(end_idx-start_idx) if self.is_sequential else 1.
        elif self._clip_centric_weight:
            key = self._dset_groups[dset_idx][clip_idx].split('/')[0]
            ret = self._clip_returns[dset_idx][key]
            weight = np.exp((ret - self._return_offset) / self._temperature)
            if self.is_sequential:
                weight = weight * np.ones(end_idx-start_idx)
        else: # state-action weight
            adv = adv_dset[start_idx:end_idx] if self.is_sequential else adv_dset[rel_idx]
            if self._advantage_weights:
                energy = adv - self._advantage_offset
            else:
                val = val_dset[start_idx:end_idx] if self.is_sequential else val_dset[rel_idx]
                energy = val + adv - self._q_value_offset
            weight = np.exp(energy / self._temperature)

        weight = np.array(np.minimum(weight, self._max_weight), dtype=np.float32)

        return obs, act, weight

if __name__ == "__main__":
    dset = ExpertDataset(
        [os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data', 'example.hdf5')],
        observables.MULTI_CLIP_OBSERVABLES_SANS_ID
    )

    sample = dset[100]
    print(sample)
