"""
分割数据集
训练集和测试集比例为 8:2
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def check_create_path(path):
    """
    check and mkdir if not exist
    recursive
    """
    if not os.path.exists(path):
        os.makedirs(path)


def check_empty_dirs(path):
    """
    check if the directory is empty
    recursive
    :return: True if empty
    """
    for root, dirs, files in os.walk(path, topdown=False):
        if files:
            return False
        else:
            if dirs:
                for dir in dirs:
                    if not check_empty_dirs(os.path.join(root, dir)):
                        return False
            os.rmdir(root)
    return True


# working: str = f'occlusion/board'
# working: str = f'occlusion/cabinet'
# working: str = f'occlusion/desk'
working: str = f'light/dark'


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # current directory

DATASETS = ROOT / 'datasets'
RAW = DATASETS / 'raw'  # ./datasets/raw
WORKING = RAW / working  # ./datasets/raw/occlusion/board
OUTPUT = DATASETS / working  # ./datasets/occlusion/board
check_create_path(OUTPUT)


def split(path):
    """
    对每个模态、每个动作分别划分数据集和测试集
    """
    for mod in os.listdir(path):
        MOD = path / mod

        for label in os.listdir(MOD):
            LABEL = MOD / label  # ./datasets/raw/occlusion/board/CSI/clap

            train, test = split_train_test(LABEL)  # pandas.DataFrame

            TRAIN = OUTPUT / 'train' / mod / label  # ./datasets/occlusion/board/train/CSI/clap
            TEST = OUTPUT / 'test' / mod / label  # ./datasets/occlusion/board/test/CSI/clap
            check_create_path(TRAIN)
            check_create_path(TEST)

            move_datasets(train, TRAIN)
            move_datasets(test, TEST)


def move_datasets(files, path):
    """
    :param files: pandas.DataFrame file list
    :param path: output path
    :return: None
    """
    # print(files)
    # print(files.dtypes)
    # print(files[0])
    for file in files[0]:
        # print(file)
        shutil.move(file, path)


def split_train_test(path, train_radio=0.8):
    FILES = os.listdir(path)
    FILES = [path / file for file in FILES]
    FILES = pd.DataFrame(FILES)

    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(FILES))
    train_set_size = int(len(FILES) * train_radio)
    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size:]

    return FILES.iloc[train_indices], FILES.iloc[test_indices]


if __name__ == '__main__':
    split(WORKING)
    if not check_empty_dirs(WORKING):
        print("Some files still in raw directory, check manually")
