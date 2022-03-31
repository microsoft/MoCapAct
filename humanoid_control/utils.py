import os.path as osp
from importlib import import_module

from absl import logging
from azure.storage.blob import ContainerClient, BlobProperties, ExponentialRetry

from dm_control.locomotion.mocap import cmu_mocap_data, loader
from dm_control.locomotion.tasks.reference_pose.tracking import _MAX_END_STEP

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

def get_clip_length(clip_id):
    """
    We subtract one from the end step due to how the ReferencePosesTask handles the
    last step of a reference trajectory.
    """
    clip_loader = loader.HDF5TrajectoryLoader(cmu_mocap_data.get_path_for_cmu(version='2020'))
    clip = clip_loader.get_trajectory(clip_id, start_step=0, end_step=_MAX_END_STEP)
    return clip.end_step-1


class AzureBlobConnector():
    def __init__(self, blob_url: str = None, account_name: str = None, sas_token: str = None, connection_string: str = None, container_name: str = None):
        retry_policy = ExponentialRetry(retry_total=6)
        if blob_url:
            self.container_client = ContainerClient.from_container_url(blob_url, retry_policy=retry_policy)
        elif account_name and container_name and sas_token:
            self.container_client = ContainerClient.from_container_url(f'https://{account_name}.blob.core.windows.net/{container_name}{sas_token}', retry_policy=retry_policy)
        elif connection_string and container_name:
            self.container_client = ContainerClient.from_connection_string(connection_string, container_name, retry_policy=retry_policy)
        else:
            raise Exception('No storage account credentials passed.')

    def create_container(self) -> None:
        self.container_client.create_container()

    def upload_to_blob(self, blob_name: str, local_file_path: str) -> None:
        blob_client = self.container_client.get_blob_client(blob_name)
        with open(local_file_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)

    def download_and_save_blob(self, blob_name: str, local_file_path: str, max_concurrency: int = 1) -> None:
        blob_client = self.container_client.get_blob_client(blob_name)
        with open(local_file_path, 'wb') as local_file:
            download_stream = blob_client.download_blob(max_concurrency=max_concurrency)
            local_file.write(download_stream.readall())

    def list_blobs(self):
        return self.container_client.list_blobs()

    def get_blob_properties(self, blob_name: str) -> BlobProperties:
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
