"""
Script used to download a subset of the dataset.
"""
import argparse
import os
import os.path as osp
from mocapact.utils import AzureBlobConnector
from dm_control.locomotion.tasks.reference_pose import cmu_subsets


DATASET_URL = 'https://mocapact.blob.core.windows.net/public?sv=2020-10-02&si=public-1819108CAA5&sr=c&sig=Jw1zsVs%2BK2G6QP%2Bo%2FFPQb1rSUY8AL%2F24k4zhQuw5WPo%3D'
CMU_SUBSETS = 'https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/tasks/reference_pose/cmu_subsets.py'
SUBSET_NAMES = '<get_up | walk_tiny | run_jump_tiny | locomotion_small | all>'


def download_dataset_from_url(dataset_url, blob_prefix, local_dest_path='./data', clips=None):
    print(f'Downloading {blob_prefix} dataset from:', dataset_url, 'to', local_dest_path)
    os.makedirs(osp.dirname(local_dest_path), exist_ok=True)
    blob_connector = AzureBlobConnector(dataset_url)

    if blob_prefix == 'experts':
        for clip in clips:
            expert_blobs = blob_connector.list_blobs(prefix_filter=osp.join(blob_prefix, clip))

            for blob in expert_blobs:
                file_path = osp.join(local_dest_path, blob['name'])
                os.makedirs(osp.dirname(file_path), exist_ok=True)
                blob_connector.download_and_save_blob(blob['name'], file_path, max_concurrency=8)

        return local_dest_path

    for i, clip in enumerate(clips):
        if not clip.endswith('.hdf5'):
            clips[i] = clip + '.hdf5'

    for clip in clips:
        if not blob_connector.blob_exists(osp.join(blob_prefix, clip)):
            raise Exception(f"Clip {clip} does not exist in the dataset, please check the available snippets at {CMU_SUBSETS}.")

    for clip in clips:
        file_path = osp.join(local_dest_path, blob_prefix, clip)
        os.makedirs(osp.dirname(file_path), exist_ok=True)
        blob_connector.download_and_save_blob(osp.join(blob_prefix, clip), file_path, max_concurrency=8)

    # dataset metrics
    metrics_path = osp.join(local_dest_path, blob_prefix, 'dataset_metrics.npz')
    if not osp.exists(metrics_path):
        blob_connector.download_and_save_blob(osp.join(blob_prefix, 'dataset_metrics.npz'), metrics_path, max_concurrency=8)

    return local_dest_path

def list_files_in_url(dataset_url):
    blob_connector = AzureBlobConnector(dataset_url)
    print('\nFiles available in the dataset:')
    for blob in blob_connector.list_blobs():
        print(blob['name'])

    print("\nUse the command `python download_dataset.py -f <COMMA_SEPARATED_LIST_OF_CLIPS>`, to download the desired files.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, help="Type of dataset to be downloaded <experts | small_dataset | large_dataset>")
    parser.add_argument('-c', '--clips', type=str, default='CMU_001_01,CMU_002_01',
                        help=f"Comma-separated list of clips to be downloaded OR specific subset from {CMU_SUBSETS} of form {SUBSET_NAMES}")
    parser.add_argument('-d', '--dest_path', type=str, default='./data', help="Path to the local folder where the dataset will be downloaded")

    args = parser.parse_args()

    if args.type is None:
        print("Please specify a dataset type -t from <experts | small_dataset | large_dataset>.\n")
        exit()

    clips = args.clips.split(',')
    if not clips[-1]:
        clips.pop()

    expanded_clips = []
    for clip in clips:
        if 'CMU' in clip:
            expanded_clips.append(clip)
        else:
            if clip.lower() == 'get_up':
                expanded_clips.extend(cmu_subsets.GET_UP.ids)
            elif clip.lower() == 'walk_tiny':
                expanded_clips.extend(cmu_subsets.WALK_TINY.ids)
            elif clip.lower() == 'run_jump_tiny':
                expanded_clips.extend(cmu_subsets.RUN_JUMP_TINY.ids)
            elif clip.lower() == 'locomotion_small':
                expanded_clips.extend(cmu_subsets.LOCOMOTION_SMALL.ids)
            elif clip.lower() == 'all':
                expanded_clips.extend(cmu_subsets.ALL.ids)
            else:
                raise ValueError(
                    "Please provide the --clips to download: "
                    "either a comma-separated list of clips to be downloaded "
                    f"or a specific subset from {CMU_SUBSETS} of form {SUBSET_NAMES}"
                )

    if args.type == 'large_dataset':
        blob_prefix = 'dataset/large'
    elif args.type == 'small_dataset':
        blob_prefix = 'dataset/small'
    elif args.type == 'experts':
        blob_prefix = 'experts'

    download_dataset_from_url(DATASET_URL, blob_prefix, local_dest_path=args.dest_path, clips=expanded_clips)
