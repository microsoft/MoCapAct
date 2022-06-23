import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to folder to be zipped')
    parser.add_argument('--output_path', type=str, help='Path to the output the compressed file')
    parser.add_argument('--file_name', type=str, help='File name of the compressed file')

    args = parser.parse_args()

    output_directory = args.output_path
    os.makedirs(output_directory, exist_ok=True)

    tar = tarfile.TarFile.gzopen(os.path.join(output_directory, args.file_name), mode="w")
    for root, dirs, files in os.walk(args.input_path):
        for file in files:
            tar.add(os.path.join(root, file))

    tar.close()

if __name__ == '__main__':
    main()
