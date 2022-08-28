import os.path as osp
from pathlib import Path
from importlib import import_module

from absl import logging

from dm_control.locomotion.mocap import cmu_mocap_data, loader
from dm_control.locomotion.tasks.reference_pose import types
from dm_control.locomotion.tasks.reference_pose.tracking import _MAX_END_STEP

from tqdm import tqdm
from azure.storage.blob import ContainerClient, ExponentialRetry
from typing import List, Optional, Text, Union

def str_to_callable(callable_name):
    module_name, method_name = callable_name.rsplit('.', 1)
    module = import_module(module_name)
    method = getattr(module, method_name)
    return method

def load_absl_flags(fname):
    """
    Loads the abseil flags from a text file. Does not include booleans.
    """
    flags = dict()
    with open(fname, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "=" in line:
                flag, val = line.split("=")
                flags[flag.strip("--")] = val.strip("\n")
    return flags

def log_flags(flags, log_dir):
    """Logs the value of each of the flags."""
    for k in dir(flags):
        if k != '?':
            flag = 'FLAGS.{}'.format(k)
            logging.info('{}: {}'.format(flag, eval(flag)))
    flags.append_flags_into_file(osp.join(log_dir, 'flags.txt'))

def get_clip_length(clip_id, mocap_path=None):
    mocap_path = mocap_path or cmu_mocap_data.get_path_for_cmu(version='2020')
    clip_loader = loader.HDF5TrajectoryLoader(mocap_path)
    clip = clip_loader.get_trajectory(clip_id, start_step=0, end_step=_MAX_END_STEP)
    # We subtract one from the end step due to how the ReferencePosesTask handles the
    # last step of a reference trajectory.
    return clip.end_step-1

def make_clip_collection(snippets: List[Text], mocap_path: Optional[Union[str, Path]] = None):
    ids, start_steps, end_steps = [], [], []
    for snippet in snippets:
        substrings = snippet.split('-')
        ids.append(substrings[0])
        if len(substrings) >= 2:
            start_steps.append(int(substrings[1]))
        else:
            start_steps.append(0)

        if len(substrings) >= 3:
            end_steps.append(int(substrings[2]))
        else:
            end_steps.append(get_clip_length(substrings[0], mocap_path))

    return types.ClipCollection(ids=ids, start_steps=start_steps, end_steps=end_steps)


class AzureBlobConnector:
    """
    Class to connect to Azure Blob Storage and download/upload files,
    handling the authentication, retry policy and other complexities
    of the protocol.
    """
    def __init__(
        self,
        blob_url: str = None,
        account_name: str = None,
        sas_token: str = None,
        connection_string: str = None,
        container_name: str = None
    ):
        retry_policy = ExponentialRetry(retry_total=6)
        if blob_url:
            self.container_client = ContainerClient.from_container_url(
                blob_url,
                retry_policy=retry_policy
            )
        elif account_name and container_name and sas_token:
            self.container_client = ContainerClient.from_container_url(
                f'https://{account_name}.blob.core.windows.net/{container_name}{sas_token}',
                retry_policy=retry_policy
            )
        elif connection_string and container_name:
            self.container_client = ContainerClient.from_connection_string(
                connection_string,
                container_name,
                retry_policy=retry_policy
            )
        else:
            print('No storage account credentials passed.')

    def create_container(self) -> None:
        self.container_client.create_container()

    def upload_to_blob(self, blob_name: str, local_file_path: str) -> None:
        blob_client = self.container_client.get_blob_client(blob_name)
        with open(local_file_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)

    def download_and_save_blob(self, blob_name: str, local_file_path: str, max_concurrency: int = 1) -> None:
        print(f'Downloading {blob_name}')
        blob_client = self.container_client.get_blob_client(blob_name)
        with open(local_file_path, 'wb') as local_file:
            download_stream = blob_client.download_blob(max_concurrency=max_concurrency)
            pbar = tqdm(total=download_stream.size, unit='B', unit_scale=True, unit_divisor=1024)
            for chunk in download_stream.chunks():
                local_file.write(chunk)
                pbar.update(len(chunk))

    def list_blobs(self, prefix_filter=None):
        if prefix_filter is not None:
            return self.container_client.list_blobs(name_starts_with=prefix_filter)

        return self.container_client.list_blobs()

    def get_blob_properties(self, blob_name: str):
        blob_client = self.container_client.get_blob_client(blob_name)
        return blob_client.get_blob_properties()

    def delete_container(self):
        return self.container_client.delete_container()

    def blob_exists(self, blob_name: str) -> bool:
        blob_client = self.container_client.get_blob_client(blob_name)
        return blob_client.exists()

    def delete_blob(self, blob_name: str) -> None:
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.delete_blob()
