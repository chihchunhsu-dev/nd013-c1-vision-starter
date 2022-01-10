import argparse
import glob
import os
import random

import shutil
from sklearn.model_selection import train_test_split

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    tfrecords = glob.glob(os.path.join(source, "*.tfrecord"))
    train_files, val_test_files = train_test_split(tfrecords, test_size=0.4)
    val_files, test_files= train_test_split(val_test_files, test_size=0.5)
    
    train_dir = os.path.join(destination, 'train')
    val_dir = os.path.join(destination, 'val')
    test_dir = os.path.join(destination, 'test')

    files_list = [train_files, val_files, test_files]
    dirs = [train_dir, val_dir, test_dir]

    for files, dir in zip(files_list, dirs):
        for file in files:
            shutil.move(file, dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)