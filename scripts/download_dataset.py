import argparse
import os
from humanoid_control.utils import AzureBlobConnector


DATASET_URL = 'https://dilbertws7896891569.blob.core.windows.net/public?sv=2020-10-02&st=2022-03-31T02%3A16%3A46Z&se=2023-02-01T03%3A16%3A00Z&sr=c&sp=rl&sig=33NYiCqgT0m%2FWRU6kA638UrfxnVb%2FfBYaSkemYZPB14%3D'

def download_dataset_from_url(dataset_url=DATASET_URL, local_dest_path='./data', files=None):
    print('Downloading dataset:', dataset_url, 'to', local_dest_path)
    blob_connector = AzureBlobConnector(dataset_url)
    if files is None:
        files = [blob['name'] for blob in blob_connector.list_blobs()]

    for file in files:
        if not blob_connector.blob_exists(file):
            raise Exception(f"File {file} does not exist in the dataset, please check the available files with 'python download_dataset.p -l' flag.")

    os.makedirs(os.path.dirname(local_dest_path), exist_ok=True)
    for file in files:
        file_path = os.path.join(local_dest_path, file)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob_connector.download_and_save_blob(file, file_path, max_concurrency=4)

    return local_dest_path

def list_files_in_url(dataset_url=DATASET_URL):
    blob_connector = AzureBlobConnector(dataset_url)
    print('\nFiles available in the dataset:')
    for blob in blob_connector.list_blobs():
        print(blob['name'])

    print("\nUse the command 'python download_dataset.py -f <COMMA_SEPARATED_LIST_OF_FILES>', to download the desired files.\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, default=DATASET_URL)
    parser.add_argument('-l', '--list', action='store_true')
    parser.add_argument('-f', '--files', type=str, default='CMU_001_01,CMU_002_01')

    args = parser.parse_args()

    if args.list:
        list_files_in_url(dataset_url=args.url)
    else:
        files = args.files.split(',')
        if not files[-1]:
            files.pop()

        for i, file in enumerate(files):
            if not file.endswith('.hdf5'):
                files[i] = file + '.hdf5'

        download_dataset_from_url(dataset_url=args.url, local_dest_path='./data', files=files)
