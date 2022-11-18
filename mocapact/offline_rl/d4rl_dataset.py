import bisect
from typing import Dict, Sequence, Text, Optional, Union
import numpy as np

from mocapact.distillation.dataset import ExpertDataset

class D4RLDataset(ExpertDataset):
    """
    Class for datasets respecting the D4RL interface.
    For reference, please check: https://github.com/Farama-Foundation/D4RL
    """

    def __init__(
        self,
        hdf5_fnames: Sequence[Text],
        observables: Union[Sequence[Text], Dict[Text, Sequence[Text]]],
        metrics_path: Text,
        clip_ids: Optional[Sequence[Text]] = None,
        max_clip_len: int = 210,
        normalize_obs: bool = False,
        normalize_act: bool = False,
        return_mean_act: bool = False,
        clip_len_upsampling: bool = False,
        n_start_rollouts: int = -1,
        n_rsi_rollouts: int = -1,
        concat_observables: bool = True,
        keep_hdf5s_open: bool = False
    ):
        super().__init__(
            hdf5_fnames,
            observables,
            metrics_path,
            clip_ids=clip_ids,
            max_clip_len=max_clip_len,
            min_seq_steps=1,
            max_seq_steps=1,
            normalize_obs=normalize_obs,
            normalize_act=normalize_act,
            return_mean_act=return_mean_act,
            clip_len_upsampling=clip_len_upsampling,
            n_start_rollouts=n_start_rollouts,
            n_rsi_rollouts=n_rsi_rollouts,
            concat_observables=concat_observables,
            clip_weighted=False,
            advantage_weights=True,
            temperature=None,
            max_weight=float('inf'),
            keep_hdf5s_open=keep_hdf5s_open
        )

    def get_in_memory_rollouts(self, max_transitions=int(2e6)):
        data_dict = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
            'timeouts': []
        }

        n_transitions = 0
        for obs, act, rew, next_obs, terminals, timeouts in iter(self):
            if n_transitions == max_transitions:
                break

            data_dict['observations'].append(obs)
            data_dict['actions'].append(act)
            data_dict['rewards'].append(rew)
            data_dict['next_observations'].append(next_obs)
            data_dict['terminals'].append(terminals)
            data_dict['timeouts'].append(timeouts)

            n_transitions += 1

        for k in data_dict:
            data_dict[k] = np.array(data_dict[k])

        self._sanity_check(data_dict)

        return data_dict

    def _sanity_check(self, data_dict):
        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals', 'timeouts']:
            assert key in data_dict, 'Dataset is missing key "%s"' % key
        N_samples = data_dict['observations'].shape[0]
        if self.observation_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
                'Observation shape does not match env: "%s" vs "%s"' % (
                    str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['actions'].shape[1:] == self.action_space.shape, \
            'Action shape does not match env: "%s" vs "%s"' % (
                str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: "%s"' % (
            str(data_dict['rewards'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: "%s"' % (
            str(data_dict['rewards'].shape))
        if data_dict['timeouts'].shape == (N_samples, 1):
            data_dict['timeouts'] = data_dict['timeouts'][:, 0]
        assert data_dict['timeouts'].shape == (N_samples,), 'Timeouts has wrong shape: "%s"' % (
            str(data_dict['timeouts'].shape))

    def _getitem(self, dset, idx):
        dset_idx = bisect.bisect_right(self._dset_indices, idx)-1
        clip_idx = bisect.bisect_right(self._logical_indices[dset_idx], idx)-1
        act_name = "mean_actions" if self._return_mean_act else "actions"

        proprio_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/observations/proprioceptive"]
        act_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/{act_name}"]
        rew_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/rewards"]
        snippet_len_weight = self._snippet_len_weights[dset_idx][clip_idx]

        data_idx = int((idx - self._logical_indices[dset_idx][clip_idx]) / snippet_len_weight)
        all_obs = proprio_dset[data_idx]
        all_next_obs = proprio_dset[data_idx+1]
        act = act_dset[data_idx]
        rew = rew_dset[data_idx]

        if self._normalize_obs:
            all_obs = (all_obs - self.proprio_mean) / self.proprio_std
            all_next_obs = (all_next_obs - self.proprio_mean) / self.proprio_std
        if self._normalize_act:
            act = (act - self.act_mean) / self.act_std

        # Extract observation
        if isinstance(self._observables, dict):
            obs = {
                k: self._extract_observations(all_obs, observable_keys)
                for k, observable_keys in self._observables.items()
            }
            next_obs = {
                k: self._extract_observations(all_next_obs, observable_keys)
                for k, observable_keys in self._observables.items()
            }
            if self._concat_observables:
                obs = {k: np.concatenate(list(v.values()), axis=-1) for k, v in obs.items()}
                next_obs = {k: np.concatenate(list(v.values()), axis=-1) for k, v in next_obs.items()}
        else:
            obs = self._extract_observations(all_obs, self._observables)
            next_obs = self._extract_observations(all_next_obs, self._observables)
            if self._concat_observables:
                obs = np.concatenate(list(obs.values()), axis=-1)
                next_obs = np.concatenate(list(next_obs.values()), axis=-1)

        terminal, timeout = False, False
        if data_idx == act_dset.shape[0]-1: # end of episode
            terminal = self._early_terminations[dset_idx][clip_idx]
            timeout = not terminal

        return obs, act, rew, next_obs, terminal, timeout
