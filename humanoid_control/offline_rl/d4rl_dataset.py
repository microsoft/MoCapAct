import os
import h5py
import collections
import itertools
from typing import Dict, List, Sequence, Text, Tuple, Optional, Union
import urllib.request
import numpy as np
from tqdm import tqdm

from humanoid_control import observables
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
        dataset_url,
        h5py_fnames: Sequence[Text],
        observables: Union[Sequence[Text], Dict[Text, Sequence[Text]]],
        ref_min_score=None,
        ref_max_score=None, **kwargs
    ):
        self.dataset_url = dataset_url
        dataset_path = self._download_dataset_from_url(dataset_url)
        self.ref_max_score = ref_max_score
        self.ref_min_score = ref_min_score
        super(ExpertDataset, self).__init__([os.path.join(dataset_path, filename) for filename in h5py_fnames], observables, **kwargs)

    @staticmethod
    def _dataset_path_from_url(dataset_url, dataset_local_path=os.path.expanduser('~/.datasets')):
        _, dataset_name = os.path.split(dataset_url)
        dataset_path = os.path.join(dataset_local_path, dataset_name)
        return dataset_path

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
        dataset_path = D4RLDataset._dataset_path_from_url(dataset_url)
        if not os.path.exists(dataset_path):
            print('Downloading dataset:', dataset_url, 'to', dataset_path)
            urllib.request.urlretrieve(dataset_url, dataset_path)
        if not os.path.exists(dataset_path):
            raise IOError("Failed to download dataset from %s" % dataset_url)
        return dataset_path

    @property
    def dataset_filepath(self):
        return self._filepath_from_url(self.dataset_url)

    def get_normalized_score(self, score):
        if (self.ref_max_score is None) or (self.ref_min_score is None):
            raise ValueError("Reference score not provided for dataset")
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)

    def get_in_memory_rollouts(self, h5path=None, clip_ids: Optional[Sequence[Text]] = None):
        if h5path is None:
            if self.dataset_url is None:
                raise ValueError("D4RLDataset not configured with a dataset URL.")
            h5path = D4RLDataset._download_dataset_from_url(self.dataset_url)

        with h5py.File(h5path, 'r') as dataset_file:
            data_dict = self._load_dataset_in_memory(dataset_file, clip_ids)

        self._sanity_check(data_dict)

        return data_dict

    def _load_dataset_in_memory(self, dset, clip_ids: Optional[Sequence[Text]] = None):
        clip_ids = []
        if clip_ids is None:
            clip_ids = [k for k in dset.keys() if k.startswith('CMU')]
        else:
            clip_ids = [k for k in clip_ids if k in dset.keys()]

        obs, act, rews, terminals, timeouts = [], [], [], [], []
        for clip_id in clip_ids:
            for episode in range(len(dset[f"{clip_id}/loaded_metrics/episode_lengths"])):
                obs.append(dset[f"{clip_id}/{episode}/observations"][...])
                act.append(dset[f"{clip_id}/{episode}/actions"][...])
                rews.append(dset[f"{clip_id}/{episode}/rewards"][...])
                episode_terminals = [0] * dset[f"{clip_id}/episode_lengths"][...][episode]
                episode_timeouts = [0] * dset[f"{clip_id}/episode_lengths"][...][episode]
                if dset[f"{clip_id}/early_termination"]:
                    episode_terminals[-1] = 1
                else:
                    episode_timeouts[-1] = 1
                terminals.append(np.array(terminals))
                timeouts.append(np.array(timeouts))

        data_dict = {
            'observations': np.concatenate(obs),
            'actions': np.concatenate(act),
            'rewards': np.concatenate(rews),
            'terminals': np.concatenate(terminals),
            'timeouts': np.concatenate(timeouts),
        }

        return data_dict

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

if __name__ == "__main__":
    dset = D4RLDataset(
        dataset_url='https://rlnexusstorage2.blob.core.windows.net/00-share-data/humanoid_control/public/example.hdf5?sv=2020-10-02&st=2022-03-24T02%3A30%3A48Z&se=2022-07-01T02%3A30%3A00Z&sr=b&sp=r&sig=mZ2EUrx%2Bv6gdkzHCHgussSSzoASK04JL9q%2BaiKks5Oc%3D',
        h5py_fnames=['example.hdf5'],
        observables=observables.TIME_INDEX_OBSERVABLES
    )
    sample = dset[0]
    d4rl_data_dict = dset.get_in_memory_rollouts(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data', 'example.hdf5'))
