"""
处理CSI/Mat文件夹内容和Video内容不匹配的情况
"""

import os
import shutil
from pathlib import Path

working: str = f'occlusion/board'

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # current directory
DATASETS = ROOT / 'datasets'

WORKING = DATASETS / working


def renameDir(path, train_test):
    d = path / train_test
    MAT = d / 'Mat'
    VIDEO = d / 'Video'
    label = os.listdir(MAT)[0]
    MAT_LABEL = MAT / label
    VIDEO_LABEL = VIDEO / label
    mat_post = os.listdir(MAT_LABEL)[0].split('.')[1:]
    video_post = os.listdir(VIDEO_LABEL)[0].split('.')[1:]
    if video_post[0] == 'dat':
        remove(VIDEO, 'dat')


def remove(path, post):
    for label in os.listdir(path):
        LABEL = path / label
        for dir in os.listdir(LABEL):
            os.rename(LABEL / dir, LABEL / dir.split('.')[0])


# renameDir(WORKING, 'train')
# renameDir(WORKING, 'test')

# MAT = WORKING / 'train/Video'
MAT = WORKING / 'test/Video'
for label in os.listdir(MAT):
    LABEL = MAT / label
    for file in os.listdir(LABEL):
        s = file.split('-')
        ss = [s[0], '1', '1', '1', s[-1]]
        filename = '-'.join(ss)
        # print(filename)
        os.rename(LABEL / file, LABEL / filename)