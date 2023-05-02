"""
Dataset used for training a policy. Formed from a collection of
HDF5 files and wrapped into a PyTorch Dataset.
"""
import bisect
import h5py
import itertools
import collections
import numpy as np
from typing import Dict, Sequence, Text, Optional, Union
from scipy.special import logsumexp
from gym import spaces
from torch.utils.data import Dataset
from stable_baselines3.common.running_mean_std import RunningMeanStd

MULTIPLIER = 10

def weighted_average(arrays, weights):
    total = 0
    for array, weight in zip(arrays, weights):
        total += weight*array
    return total / sum(weights)

class ExpertDataset(Dataset):
    def __init__(
        self,
        hdf5_fnames: Sequence[Text],
        observables: Union[Sequence[Text], Dict[Text, Sequence[Text]]],
        metrics_path: Text,
        clip_ids: Optional[Sequence[Text]] = None,
        max_clip_len: int = 210,
        min_seq_steps: int = 1,
        max_seq_steps: int = 1,
        normalize_obs: bool = False,
        normalize_act: bool = False,
        return_mean_act: bool = True,
        clip_len_upsampling: bool = False,
        n_start_rollouts: int = -1,
        n_rsi_rollouts: int = -1,
        concat_observables: bool = True,
        clip_weighted: bool = False,
        advantage_weights: bool = True,
        temperature: Optional[float] = None,
        max_weight: float = float('inf'),
        keep_hdf5s_open: bool = False
    ):
        """
        hdf5_fnames: List of paths to HDF5 files to load.
        observables: What observables to return in __getitem__.
        metrics_path: The path used to load the dataset metrics.
        clip_ids: The clip IDs to consider. If None, considers every clip in the dataset.
        min_seq_steps: The minimum number of steps in a returned sequence.
        max_seq_steps: The maximum number of steps in a returned sequence.
        normalize_obs: Whether to normalize the observation.
        normalize_act: Whether to normalize the action.
        return_mean_act: Whether to return the mean action or sampled action in __getitem__.
        concat_observables: Whether to concatenate the observables in __getitem__.
        clip_weighted: Whether to determine the weights based on the clip or the state-action.
        temperature: The temperature used in the data weighting.
        max_weight: The maximum weight for the data.
        """
        self._hdf5_fnames = hdf5_fnames
        self._observables = observables

        self._keep_hdf5s_open = keep_hdf5s_open
        if self._keep_hdf5s_open:
            self._dsets = [h5py.File(fname, 'r') for fname in self._hdf5_fnames]

        # Grab all clip snippet information
        # self._clip_snippets separates the snippets by file
        # self._clip_snippets_flat flattens self._clip_snippets
        # self._clip_ids is the name of the clip IDs (start and end steps removed)
        self._clip_snippets = []
        for fname in self._hdf5_fnames:
            with h5py.File(fname, 'r') as dset:
                if clip_ids is None: # get every clip ID in the dataset
                    self._clip_snippets.append(tuple([k for k in dset.keys() if k.startswith('CMU')]))
                else: # get only those clip IDs in the dataset that are in clip_ids
                    self._clip_snippets.append(tuple([k for k in clip_ids if k in dset.keys()]))
        self._clip_snippets_flat = tuple(itertools.chain.from_iterable(self._clip_snippets))
        self._clip_ids = tuple({k.split('-')[0] for k in self._clip_snippets_flat})

        self._max_clip_len = max_clip_len
        self._min_seq_steps = min_seq_steps
        self._max_seq_steps = max_seq_steps
        self._concat_observables = concat_observables
        self._clip_weighted = clip_weighted
        self._advantage_weights = advantage_weights
        self._temperature = temperature
        self._max_weight = max_weight
        self._normalize_obs = normalize_obs
        self._normalize_act = normalize_act
        self._return_mean_act = return_mean_act
        self._clip_len_upsampling = clip_len_upsampling
        self._n_start_rollouts = n_start_rollouts
        self._n_rsi_rollouts = n_rsi_rollouts
        self._metrics_path = metrics_path

        # Grab the reference steps and indices for observables from the first HDF5.
        # We assume those two properties are the same for all the HDF5s.
        with h5py.File(self._hdf5_fnames[0], 'r') as dset:
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
    def proprio_mean(self):
        return self._proprio_mean

    @property
    def proprio_var(self):
        return self._proprio_var

    @property
    def proprio_std(self):
        return self._proprio_std

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
        with h5py.File(self._hdf5_fnames[0], 'r') as dset:
            self._action_space = spaces.Box(
                low=np.float32(-1.),
                high=np.float32(1.),
                shape=dset[f"{self._clip_snippets[0][0]}/0/actions"].shape[1:],
                dtype=np.float32
            )

    def _set_stats(self):
        metrics = np.load(self._metrics_path, allow_pickle=True)
        self._count = metrics['count']
        self._proprio_mean = metrics['proprio_mean']
        self._proprio_var = metrics['proprio_var']
        if self._return_mean_act:
            self._act_mean = metrics['mean_act_mean']
            self._act_var = metrics['mean_act_var']
        else:
            self._act_mean = metrics['act_mean']
            self._act_var = metrics['act_var']
        self._snippet_returns = metrics['snippet_returns'].item()
        self._advantages = {k: v for k, v in metrics['advantages'].item().items() if k in self._clip_snippets_flat}
        self._values = {k: v for k, v in metrics['values'].item().items() if k in self._clip_snippets_flat}

        self._proprio_std = (np.sqrt(self.proprio_var) + 1e-4).astype(np.float32)
        self._act_std = (np.sqrt(self.act_var) + 1e-4).astype(np.float32)

        # Put observation statistics into RunningMeanStd objects
        self._obs_rms = dict()
        with h5py.File(self._hdf5_fnames[0], 'r') as dset:
            for k in dset['observable_indices/walker'].keys():
                key = "walker/" + k
                obs_rms = RunningMeanStd()
                obs_rms.mean = self.proprio_mean[self.observable_indices[key]]
                obs_rms.var = self.proprio_var[self.observable_indices[key]]
                obs_rms.count = self._count
                self._obs_rms[key] = obs_rms

        snippet_returns = np.array(list(self.snippet_returns.values()))
        advantages, values = [np.concatenate(list(x.values())) for x in [self.advantages, self.values]]
        self._return_offset = self._compute_offset(snippet_returns)
        self._advantage_offset = self._compute_offset(advantages)
        self._q_value_offset = self._compute_offset(values + advantages)

    def _create_offsets(self):
        self._total_len = 0
        self._dset_indices = []
        self._logical_indices, self._dset_groups = [[] for _ in self._hdf5_fnames], [[] for _ in self._hdf5_fnames]
        self._early_terminations = [[] for _ in self._hdf5_fnames]
        self._snippet_len_weights = [[] for _ in self._hdf5_fnames]
        iterator = zip(
            self._hdf5_fnames,
            self._clip_snippets,
            self._logical_indices,
            self._dset_groups,
            self._early_terminations,
            self._snippet_len_weights
        )
        for fname, clip_snippets, logical_indices, dset_groups, early_terminations, snippet_len_weights in iterator:
            with h5py.File(fname, 'r') as dset:
                self._dset_indices.append(self._total_len)
                dset_start_rollouts = dset['n_start_rollouts'][...]
                dset_rsi_rollouts = dset['n_rsi_rollouts'][...]
                n_start_rollouts = dset_start_rollouts if self._n_start_rollouts < 0 else min(self._n_start_rollouts, dset_start_rollouts)
                n_rsi_rollouts = dset_rsi_rollouts if self._n_rsi_rollouts < 0 else min(self._n_rsi_rollouts, dset_rsi_rollouts)
                for snippet in clip_snippets:
                    _, start, end = snippet.split('-')
                    clip_len = int(end)-int(start)
                    snippet_weight = int(self._max_clip_len / clip_len * MULTIPLIER) if self._clip_len_upsampling else 1

                    len_iterator = itertools.chain(
                        dset[f"{snippet}/start_metrics/episode_lengths"][:n_start_rollouts],
                        dset[f"{snippet}/rsi_metrics/episode_lengths"][:n_rsi_rollouts]
                    )
                    for i, ep_len in enumerate(len_iterator):
                        episode = i if i < n_start_rollouts else i-n_start_rollouts+dset_start_rollouts
                        logical_indices.append(self._total_len)
                        dset_groups.append(f"{snippet}/{episode}")
                        early_terminations.append(dset[f"{snippet}/early_termination"][episode])
                        snippet_len_weights.append(snippet_weight)
                        if ep_len < self._min_seq_steps:
                            continue
                        self._total_len += snippet_weight * (ep_len - (self._min_seq_steps-1))

    def _compute_offset(self, array: np.ndarray):
        """
        Used to ensure the average data weight is approximately one.
        """
        if self._temperature is None or self._temperature == float('inf'):
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
        if idx >= len(self):
            raise IndexError("Dataset index out of range")

        dset_idx = bisect.bisect_right(self._dset_indices, idx)-1

        if self._keep_hdf5s_open:
            return self._getitem(self._dsets[dset_idx], idx)

        with h5py.File(self._hdf5_fnames[dset_idx], 'r') as dset:
            return self._getitem(dset, idx)

    def _getitem(self, dset, idx):
        dset_idx = bisect.bisect_right(self._dset_indices, idx)-1
        clip_idx = bisect.bisect_right(self._logical_indices[dset_idx], idx)-1
        act_name = "mean_actions" if self._return_mean_act else "actions"

        proprio_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/observations/proprioceptive"]
        act_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/{act_name}"]
        val_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/values"]
        adv_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/advantages"]
        snippet_len_weight = self._snippet_len_weights[dset_idx][clip_idx]

        start_idx = int((idx - self._logical_indices[dset_idx][clip_idx]) / snippet_len_weight)
        if self.is_sequential:
            end_idx = min(start_idx + self._max_seq_steps, act_dset.shape[0]+1)
            all_obs = proprio_dset[start_idx:end_idx]
            act = act_dset[start_idx:end_idx]
        else:
            all_obs = proprio_dset[start_idx]
            act = act_dset[start_idx]

        if self._normalize_obs:
            all_obs = (all_obs - self.proprio_mean) / self.proprio_std
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

        if self._temperature is None or self._temperature == float('inf'):
            weight = np.ones(end_idx-start_idx) if self.is_sequential else 1.
        elif self._clip_weighted:
            key = self._dset_groups[dset_idx][clip_idx].split('/')[0]
            ret = self._snippet_returns[key]
            weight = np.exp((ret - self._return_offset) / self._temperature)
            if self.is_sequential:
                weight = weight * np.ones(end_idx-start_idx)
        else: # state-action weight
            adv = adv_dset[start_idx:end_idx] if self.is_sequential else adv_dset[start_idx]
            if self._advantage_weights:
                energy = adv - self._advantage_offset
            else:
                val = val_dset[start_idx:end_idx] if self.is_sequential else val_dset[start_idx]
                energy = val + adv - self._q_value_offset
            weight = np.exp(energy / self._temperature)

        weight = np.array(np.minimum(weight, self._max_weight), dtype=np.float32)

        return obs, act, weight
