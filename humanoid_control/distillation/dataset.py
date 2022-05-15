import os
import bisect
import h5py
import itertools
import collections
import numpy as np
from typing import Dict, List, Sequence, Text, Tuple, Optional, Union
from scipy.special import logsumexp
from gym import spaces
from torch.utils.data import Dataset
from stable_baselines3.common.running_mean_std import RunningMeanStd
from humanoid_control import observables
import os
import glob
from tqdm import tqdm

def weighted_average(arrays, weights):
    total = 0
    for array, weight in zip(arrays, weights):
        total += weight*array
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
        clip_weighted: bool = False,
        advantage_weights: bool = True,
        temperature: Optional[float] = None,
        max_weight: float = float('inf'),
        metrics_path: Optional[Text] = None,
        keep_hdf5s_open: bool = False
    ):
        """
        h5py_fnames: List of paths to HDF5 files to load.
        observables: What observables to return in __getitem__.
        clip_ids: The clip IDs to consider. If None, considers every clip.
        min_seq_steps: The minimum number of steps in a returned sequence.
        max_seq_steps: The maximum number of steps in a returned sequence.
        normalize_obs: Whether to normalize the observation.
        normalize_act: Whether to normalize the action.
        concat_observables: Whether to concatenate the observables in __getitem__.
        clip_weighted: Whether to determine the weights based on the clip or the state-action.
        temperature: The temperature used in the data weighting.
        max_weight: The maximum weight for the data.
        metrics_path: The path used to load the dataset metrics. Otherwise, calculated manually.
        """
        self._h5py_fnames = h5py_fnames
        self._observables = observables

        self._keep_hdf5s_open = keep_hdf5s_open
        if self._keep_hdf5s_open:
            self._dsets = [h5py.File(fname, 'r') for fname in self._h5py_fnames]

        # Grab all clip snippet information
        # self._clip_snippets separates the snippets by file
        # self._clip_snippets_flat flattens self._clip_snippets
        # slef._clip_ids is the name of the clip IDs (start and end steps removed)
        self._clip_snippets = []
        for fname in self._h5py_fnames:
            with h5py.File(fname, 'r') as dset:
                if clip_ids is None: # get every clip ID in the dataset
                    self._clip_snippets.append(tuple([k for k in dset.keys() if k.startswith('CMU')]))
                else: # get only those clip IDs in the dataset that are in clip_ids
                    self._clip_snippets.append(tuple([k for k in clip_ids if k in dset.keys()]))
        self._clip_snippets_flat = tuple(itertools.chain.from_iterable(self._clip_snippets))
        self._clip_ids = tuple({k.split('-')[0] for k in self._clip_snippets_flat})

        self._min_seq_steps = min_seq_steps
        self._max_seq_steps = max_seq_steps
        self._concat_observables = concat_observables
        self._clip_weighted = clip_weighted
        self._advantage_weights = advantage_weights
        self._temperature = temperature
        self._max_weight = max_weight
        self._normalize_obs = normalize_obs
        self._normalize_act = normalize_act
        self._metrics_path = metrics_path

        # Grab the reference steps and indices for observables from the first HDF5.
        # We assume those two properties are the same for all the HDF5s.
        with h5py.File(self._h5py_fnames[0], 'r') as dset:
            self._ref_steps = dset['ref_steps'][...]
            obs_ind_dset = dset['observable_indices/walker']
            self._observable_indices = {
                f"walker/{k}" : obs_ind_dset[k][...] for k in obs_ind_dset
            }

        self._set_spaces()
        self._set_stats()

        self._create_offsets()

    @property
    def clip_snippets_flat(self):
        return self._clip_snippets_flat

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
        return self._observable_indices

    @property
    def obs_rms(self):
        return self._obs_rms

    @property
    def advantages(self):
        return self._advantages

    @property
    def values(self):
        return self._values

    @property
    def snippet_returns(self):
        return self._snippet_returns

    @property
    def count(self):
        return self._count

    def _set_spaces(self):
        """
        Sets the observation and action spaces.
        """
        # Observation space for all observables in the dataset
        obs_spaces = {
            k: spaces.Box(-np.infty, np.infty, shape=v.shape, dtype=np.float32)
            for k, v in self.observable_indices.items()
        }
        obs_spaces['walker/clip_id'] = spaces.Discrete(len(self._clip_snippets_flat))
        self._full_observation_space = spaces.Dict(obs_spaces)

        # Observation space for the observables we're considering
        def make_observation_space(observables):
            observation_indices = {k: self.observable_indices[k] for k in observables}
            if self._concat_observables:
                observation_indices = np.concatenate(list(observation_indices.values()))
                return spaces.Box(low=-np.infty, high=np.infty, shape=observation_indices.shape)
            return spaces.Dict({
                observable: spaces.Box(
                    low=-np.infty,
                    high=np.infty,
                    shape=indices.shape,
                    dtype=np.float32
                ) for observable, indices in observation_indices.items()
            })
        if isinstance(self._observables, collections.abc.Sequence): # observables is Sequence[Text]
            self._observation_space = make_observation_space(self._observables)
        else: # observables is Dict[Text, Sequence[Text]]
            self._observation_space = {
                k: make_observation_space(subobservables) for k, subobservables in self._observables.items()
            }

        # Action space
        with h5py.File(self._h5py_fnames[0], 'r') as dset:
            self._action_space = spaces.Box(
                low=np.float32(-1.),
                high=np.float32(1.),
                shape=dset[f"{self._clip_snippets[0][0]}/0/actions"].shape[1:],
                dtype=np.float32
            )

    def _set_stats(self):
        if self._metrics_path is not None:
            metrics = np.load(self._metrics_path, allow_pickle=True)
            self._count = metrics['count']
            self._obs_mean = metrics['obs_mean']
            self._obs_var = metrics['obs_var']
            self._act_mean = metrics['act_mean']
            self._act_var = metrics['act_var']
            self._snippet_returns = metrics['snippet_returns']
            self._advantages = metrics['advantages']
            self._values = metrics['values']
        else:
            counts, obs_means, obs_vars, act_means, act_vars = [], [], [], [], []
            self._snippet_returns = [{} for _ in self._h5py_fnames]
            all_advantages, all_values = [], []
            for fname, clip_snippets, snippet_return in zip(self._h5py_fnames, self._clip_snippets, self._snippet_returns):
                with h5py.File(fname, 'r') as dset:
                    counts.append(dset['stats/count'][...])
                    obs_means.append(dset['stats/obs_mean'][...])
                    obs_vars.append(dset['stats/obs_var'][...])
                    act_means.append(dset['stats/act_mean'][...])
                    act_vars.append(dset['stats/act_var'][...])
                    for snippet in clip_snippets:
                        ret_iterator = itertools.chain(
                            dset[f"{snippet}/start_metrics/norm_episode_returns"],
                            dset[f"{snippet}/rsi_metrics/norm_episode_returns"]
                        )
                        returns = list(ret_iterator)
                        snippet_return[snippet] = np.mean(returns)

                        n_rollouts = dset["n_start_rollouts"][...] + dset["n_random_rollouts"][...]
                        for i in range(n_rollouts):
                            all_advantages.append(dset[f"{snippet}/{i}/advantages"][...])
                            all_values.append(dset[f"{snippet}/{i}/values"][...])

            self._count = np.sum(counts)
            obs_second_moments = [var + mean**2 for mean, var in zip(obs_means, obs_vars)]
            act_second_moments = [var + mean**2 for mean, var in zip(act_means, act_vars)]
            self._obs_mean = weighted_average(obs_means, counts).astype(np.float32)
            self._obs_var = np.clip(
                weighted_average(obs_second_moments, counts) - self._obs_mean**2,
                0., None
            ).astype(np.float32)
            self._act_mean = weighted_average(act_means, counts).astype(np.float32)
            self._act_var = np.clip(
                weighted_average(act_second_moments, counts) - self._act_mean**2,
                0., None
            ).astype(np.float32)

            self._advantages = np.concatenate(all_advantages)
            self._values = np.concatenate(all_values)

        self._obs_std = (np.sqrt(self._obs_var) + 1e-4).astype(np.float32)
        self._act_std = (np.sqrt(self._act_var) + 1e-4).astype(np.float32)

        self._obs_rms = {}
        for k in observables.MULTI_CLIP_OBSERVABLES_SANS_ID:
            obs_rms = RunningMeanStd()
            obs_rms.mean = self.obs_mean[self.observable_indices[k]]
            obs_rms.var = self.obs_var[self.observable_indices[k]]
            obs_rms.count = self._count
            self._obs_rms[k] = obs_rms

        all_snippet_returns = list(itertools.chain(*[d.values() for d in self._snippet_returns]))
        self._return_offset = self._compute_offset(np.array(all_snippet_returns))
        self._advantage_offset = self._compute_offset(self.advantages)
        self._q_value_offset = self._compute_offset(self.values + self.advantages)

    def _create_offsets(self):
        self._total_len = 0
        self._dset_indices = []
        self._logical_indices, self._dset_groups = [[] for _ in self._h5py_fnames], [[] for _ in self._h5py_fnames]
        iterator = zip(
            self._h5py_fnames,
            self._clip_snippets,
            self._logical_indices,
            self._dset_groups
        )
        for fname, clip_snippets, logical_indices, dset_groups in iterator:
            with h5py.File(fname, 'r') as dset:
                self._dset_indices.append(self._total_len)
                for snippet in clip_snippets:
                    len_iterator = itertools.chain(dset[f"{snippet}/start_metrics/episode_lengths"],
                                                   dset[f"{snippet}/rsi_metrics/episode_lengths"])
                    for i, ep_len in enumerate(len_iterator):
                        logical_indices.append(self._total_len)
                        dset_groups.append(f"{snippet}/{i}")
                        if ep_len < self._min_seq_steps:
                            continue
                        self._total_len += ep_len - (self._min_seq_steps-1)

    def _compute_offset(self, array: np.ndarray):
        """
        Used to ensure the average data weight is approximately one.
        """
        if self._temperature is None:
            return 0.
        return self._temperature * logsumexp(array / self._temperature - np.log(array.size))

    def _extract_observations(self, all_obs: np.ndarray, observable_keys: Sequence[Text]):
        return {k: all_obs[..., self.observable_indices[k]] for k in observable_keys}

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        """
        TODO
        """
        dset_idx = bisect.bisect_right(self._dset_indices, idx)-1

        if self._keep_hdf5s_open:
            return self._getitem(self._dsets[dset_idx], idx)

        with h5py.File(self._h5py_fnames[dset_idx], 'r') as dset:
            return self._getitem(dset, idx)

    def _getitem(self, dset, idx):
        dset_idx = bisect.bisect_right(self._dset_indices, idx)-1
        clip_idx = bisect.bisect_right(self._logical_indices[dset_idx], idx)-1

        obs_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/observations"]
        act_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/actions"]
        val_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/values"]
        adv_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/advantages"]

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
        elif self._clip_weighted:
            key = self._dset_groups[dset_idx][clip_idx].split('/')[0]
            ret = self._snippet_returns[dset_idx][key]
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
