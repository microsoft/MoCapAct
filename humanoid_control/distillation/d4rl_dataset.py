import os
import h5py
import collections
import itertools
from typing import Dict, List, Sequence, Text, Tuple, Optional, Union
import urllib.request

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class D4RLDataset(Dataset):
    """
    Base class for datasets respecting the D4rl interface.
    For reference, please check: https://github.com/rail-berkeley/d4rl

    Args:
        dataset_url: URL pointing to the dataset.
        ref_max_score: Maximum score (for score normalization)
        ref_min_score: Minimum score (for score normalization)
    """
    
    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs):
        super(Dataset, self).__init__(**kwargs)
        self.dataset_url = self._dataset_url = dataset_url
        self.ref_max_score = ref_max_score
        self.ref_min_score = ref_min_score

    @staticmethod
    def _filepath_from_url(dataset_url, dataset_local_path=os.path.expanduser('~/.d4rl/datasets')):
        _, dataset_name = os.path.split(dataset_url)
        dataset_filepath = os.path.join(dataset_local_path, dataset_name)
        return dataset_filepath

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
        dataset_filepath = D4RLDataset._filepath_from_url(dataset_url)
        if not os.path.exists(dataset_filepath):
            print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
            urllib.request.urlretrieve(dataset_url, dataset_filepath)
        if not os.path.exists(dataset_filepath):
            raise IOError("Failed to download dataset from %s" % dataset_url)
        return dataset_filepath

    @property
    def dataset_filepath(self):
        return self._filepath_from_url(self.dataset_url)

    def get_normalized_score(self, score):
        if (self.ref_max_score is None) or (self.ref_min_score is None):
            raise ValueError("Reference score not provided for dataset")
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)

    def get_d4rl_dataset_from_expert_rollouts(self, h5path=None, clip_ids: Optional[Sequence[Text]] = None):
        if h5path is None:
            if self._dataset_url is None:
                raise ValueError("D4RLDataset not configured with a dataset URL.")
            h5path = D4RLDataset._download_dataset_from_url(self.dataset_url)

        with h5py.File(h5path, 'r') as dataset_file:
            data_dict = self._load_data(dataset_file, clip_ids)

        self._sanity_check(data_dict)

        return data_dict

    def _load_data(self, dset, clip_ids: Optional[Sequence[Text]] = None):
        if clip_ids is None:
            self._clip_ids.append(tuple([k for k in dset.keys() if k.startswith('CMU')]))
        else:
            self._clip_ids.append(tuple([k for k in clip_ids if k in dset.keys()]))

        obs, act, rews, terms = [], [], [], []
        for clip_id in self._clip_ids:          
            for episode in range(len(dset[f"{clip_id}/{episode}/episode_lengths"])):
                obs.append(dset[f"{clip_id}/{episode}/observations"])
                act.append(dset[f"{clip_id}/{episode}/actions"])
                rews.append(dset[f"{clip_id}/{episode}/rewards"])
                terminals = [0] * dset[f"{clip_id}/{episode}/episode_lengths"]
                terminals[-1] = 1
                terms.append(terminals)
              
        data_dict = {
            'observations': np.array(obs),
            'actions': np.array(obs),
            'rewards': np.array(obs),
            'terminals': np.array(obs)
        }

        return data_dict

    def _sanity_check(self, data_dict):
        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals']:
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


if __name__ == "__main__":
    dset = D4RLDataset()
    dset.get_dataset(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data', 'CMU_016_22.hdf5'))