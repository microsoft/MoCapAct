import os
import bisect
import h5py
import collections
import itertools
from typing import Dict, List, Sequence, Text, Tuple, Optional, Union
import urllib.request
import numpy as np
from tqdm import tqdm

from humanoid_control import observables
from humanoid_control.utils import AzureBlobConnector
from humanoid_control.distillation.dataset import ExpertDataset

class D4RLDataset(ExpertDataset):
    """
    Class for datasets respecting the D4rl interface.
    For reference, please check: https://github.com/rail-berkeley/d4rl

    Args:
        dataset_url: URL pointing to the dataset.
        ref_max_score: Maximum score (for score normalization)
        ref_min_score: Minimum score (for score normalization)
    """

    def __init__(
        self,
        observables: Union[Sequence[Text], Dict[Text, Sequence[Text]]],
        dataset_url: Optional[Text] = None,
        dataset_local_path: Optional[Text] = None,
        h5py_fnames: Optional[Sequence[Text]] = None,
        ref_min_score=None,
        ref_max_score=None,
        **kwargs
    ):
        self.dataset_url = dataset_url
        self.dataset_local_path = dataset_local_path
        if not self.dataset_local_path:
            self.dataset_local_path = self._download_dataset_from_url(self.dataset_url)

        self.ref_max_score = ref_max_score
        self.ref_min_score = ref_min_score
        if h5py_fnames is None:
            h5py_fnames = [f for f in os.listdir(self.dataset_local_path) if f.endswith('.hdf5')]

        super().__init__([os.path.join(self.dataset_local_path, filename) for filename in h5py_fnames], observables, **kwargs)

    @staticmethod
    def _dataset_path_from_url(dataset_url, dataset_local_path=os.path.expanduser('~/.d4rl/datasets/humanoid_control')):
        return dataset_local_path, dataset_url

    @staticmethod
    def _get_keys(h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    @staticmethod
    def _download_dataset_from_url(dataset_url):
        dataset_path, dataset_url = D4RLDataset._dataset_path_from_url(dataset_url)
        if not os.path.exists(dataset_path):
            print('Downloading dataset:', dataset_url, 'to', dataset_path)
            blob_connector = AzureBlobConnector(dataset_url)
            for blob in blob_connector.list_blobs():
                local_file_path = os.path.join(dataset_path, blob['name'])
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                blob_connector.download_and_save_blob(blob['name'], local_file_path)

        dirs = os.listdir(dataset_path)
        if len(dirs) > 1:
            print("More than one root directory found for the dataset. Selecting the first one.")

        dataset_path = os.path.join(dataset_path, dirs[0])

        return dataset_path

    @property
    def dataset_filepath(self):
        return self._filepath_from_url(self.dataset_url)

    def get_normalized_score(self, score):
        if (self.ref_max_score is None) or (self.ref_min_score is None):
            raise ValueError("Reference score not provided for dataset")
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)

    def get_in_memory_rollouts(self, h5py_fpaths: Optional[Sequence[Text]] = None, clip_ids: Optional[Sequence[Text]] = None):
        if h5py_fpaths is None:
            if self.dataset_url is None:
                raise ValueError("D4RLDataset not configured with a dataset URL.")
            dataset_path = D4RLDataset._download_dataset_from_url(self.dataset_url)
            h5py_fpaths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.hdf5')]

        data_dict = {}
        for file_path in h5py_fpaths:
            with h5py.File(file_path, 'r') as dataset_file:
                new_data = self._load_dataset_in_memory(dataset_file, clip_ids)
                if not data_dict:
                    data_dict['observations'] = new_data['observations']
                    data_dict['actions'] = new_data['actions']
                    data_dict['rewards'] = new_data['rewards']
                    data_dict['terminals'] = new_data['terminals']
                    data_dict['timeouts'] = new_data['timeouts']
                    continue

                data_dict['observations'] = np.concatenate(data_dict['observations'], new_data['observations'])
                data_dict['actions'] = np.concatenate(data_dict['actions'], new_data['actions'])
                data_dict['rewards'] = np.concatenate(data_dict['rewards'], new_data['rewards'])
                data_dict['terminals'] = np.concatenate(data_dict['terminals'], new_data['terminals'])
                data_dict['timeouts'] = np.concatenate(data_dict['timeouts'], new_data['timeouts'])

        self._sanity_check(data_dict)

        return data_dict

    def _load_dataset_in_memory(self, dset, clip_ids: Optional[Sequence[Text]] = None):
        if clip_ids is None:
            clip_ids = [k for k in dset.keys() if k.startswith('CMU')]
        else:
            clip_ids = [k for k in clip_ids if k in dset.keys()]

        obs, act, rews, terminals, timeouts = [], [], [], [], []
        for clip_id in clip_ids:
            for episode in range(len(dset[f"{clip_id}/rsi_metrics/episode_lengths"])):
                obs.append(dset[f"{clip_id}/{episode}/observations"][...])
                act.append(dset[f"{clip_id}/{episode}/actions"][...])
                rews.append(dset[f"{clip_id}/{episode}/rewards"][...])
                episode_terminals = [0] * dset[f"{clip_id}/rsi_metrics/episode_lengths"][...][episode]
                episode_timeouts = [0] * dset[f"{clip_id}/rsi_metrics/episode_lengths"][...][episode]
                if dset[f"{clip_id}/early_termination"][episode]:
                    episode_terminals[-1] = 1
                else:
                    episode_timeouts[-1] = 1
                terminals.append(np.array(episode_terminals))
                timeouts.append(np.array(episode_timeouts))

        data_dict = {
            'observations': np.concatenate(obs),
            'actions': np.concatenate(act),
            'rewards': np.concatenate(rews),
            'terminals': np.concatenate(terminals),
            'timeouts': np.concatenate(timeouts),
        }

        return data_dict

    def _preload_dataset(self):
        self._obs_dsets, self._act_dsets, self._rew_dsets = [[] for _ in self._dsets], [[] for _ in self._dsets], [[] for _ in self._dsets]
        iterator = zip(self._dsets, self._clip_ids, self._obs_dsets, self._act_dsets, self._rew_dsets)
        for dset, clip_ids, obs_dset, act_dset, rew_dset in iterator:
            for clip_id in clip_ids:
                for i in range(len(dset[f"{clip_id}/episode_lengths"])):
                    obs_dset.append(dset[f"{clip_id}/{i}/observations"][...])
                    act_dset.append(dset[f"{clip_id}/{i}/actions"][...])
                    rew_dset.append(dset[f"{clip_id}/{i}/rewards"][...])

    def _sanity_check(self, data_dict):
        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals', 'timeouts']:
            assert key in data_dict, 'Dataset is missing key %s' % key
        N_samples = data_dict['observations'].shape[0]
        if self.observation_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
                'Observation shape does not match env: %s vs %s' % (
                    str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['actions'].shape[1:] == self.action_space.shape, \
            'Action shape does not match env: %s vs %s' % (
                str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        if data_dict['timeouts'].shape == (N_samples, 1):
            data_dict['timeouts'] = data_dict['timeouts'][:, 0]
        assert data_dict['timeouts'].shape == (N_samples,), 'Timeouts has wrong shape: %s' % (
            str(data_dict['timeouts'].shape))

    def __getitem__(self, idx):
        """
        TODO
        """
        dset_idx = bisect.bisect_right(self._dset_indices, idx) - 1
        clip_idx = bisect.bisect_right(self._logical_indices[dset_idx], idx) - 1

        if self._preload:
            obs_dset = self._obs_dsets[dset_idx][clip_idx]
            act_dset = self._act_dsets[dset_idx][clip_idx]
            rew_dset = self._rew_dsets[dset_idx][clip_idx]
        else:
            obs_dset = self._dsets[dset_idx][f"{self._dset_groups[dset_idx][clip_idx]}/observations"]
            act_dset = self._dsets[dset_idx][f"{self._dset_groups[dset_idx][clip_idx]}/actions"]
            rew_dset = self._dsets[dset_idx][f"{self._dset_groups[dset_idx][clip_idx]}/rewards"]

        val_dset = self._dsets[dset_idx][f"{self._dset_groups[dset_idx][clip_idx]}/values"]
        adv_dset = self._dsets[dset_idx][f"{self._dset_groups[dset_idx][clip_idx]}/advantages"]

        if self.is_sequential:
            start_idx = idx - self._logical_indices[dset_idx][clip_idx]
            end_idx = min(start_idx + self._max_seq_steps, act_dset.shape[0] + 1)
            next_obs_start_idx = end_idx
            next_obs_end_idx = min(next_obs_start_idx + self._max_seq_steps, act_dset.shape[0] + 1)

            all_obs = obs_dset[start_idx:end_idx]
            act = act_dset[start_idx:end_idx]
            rew = rew_dset[start_idx:end_idx]
            all_next_obs = obs_dset[next_obs_start_idx:next_obs_end_idx]
        else:
            rel_idx = idx - self._logical_indices[dset_idx][clip_idx]
            all_obs = obs_dset[rel_idx]
            all_next_obs = obs_dset[rel_idx + 1]
            act = act_dset[rel_idx]
            rew = rew_dset[rel_idx]

        if self._normalize_obs:
            all_obs = (all_obs - self.obs_mean) / self.obs_std
            all_next_obs = (all_next_obs - self.obs_mean) / self.obs_std
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

        if self._temperature is None:
            weight = np.ones(end_idx - start_idx) if self.is_sequential else 1.
        elif self._clip_centric_weight:
            key = self._dset_groups[dset_idx][clip_idx].split('/')[0]
            ret = self._clip_returns[dset_idx][key]
            weight = np.exp(ret - self._avg_return / self._temperature)
            if self.is_sequential:
                weight = weight * np.ones(end_idx - start_idx)
        else:  # state-action weight
            adv = adv_dset[start_idx:end_idx] if self.is_sequential else adv_dset[rel_idx]
            if self._advantage_weights:
                energy = adv
            else:
                val = val_dset[start_idx:end_idx] if self.is_sequential else val_dset[rel_idx]
                energy = val + adv
            weight = np.exp(energy / self._temperature)

        weight = np.array(np.minimum(weight, self._max_weight), dtype=np.float32)

        terminal, timeout = False, False
        if obs_dset.shape[0] == rel_idx:
            terminal = self._dsets[dset_idx][f"{self._all_clip_ids[clip_idx]}/early_termination"][clip_idx]
            timeout = not terminal

        return obs, act, rew, next_obs, terminal, timeout, weight

if __name__ == "__main__":
    dset = D4RLDataset(
        observables=observables.TIME_INDEX_OBSERVABLES,
        dataset_url='https://dilbertws7896891569.blob.core.windows.net/public?sv=2020-10-02&st=2022-03-31T02%3A16%3A46Z&se=2023-02-01T03%3A16%3A00Z&sr=c&sp=rl&sig=33NYiCqgT0m%2FWRU6kA638UrfxnVb%2FfBYaSkemYZPB14%3D',
        dataset_local_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data'),
        h5py_fnames=['CMU_001_01-0-178.hdf5'],
    )
    sample = dset[0]
    d4rl_data_dict = dset.get_in_memory_rollouts([os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data', 'CMU_001_01-0-178.hdf5')])
    print(d4rl_data_dict.shape)
