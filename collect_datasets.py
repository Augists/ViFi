import os
from pathlib import Path
import shutil


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # current directory
DATASETS = ROOT / 'datasets'
SAVE = DATASETS / 'collect'
check_path(SAVE)

diverse = ['light', 'occlusion']


for div in diverse:
    DIV = DATASETS / div
    for condition in os.listdir(DIV):
        if condition == 'cabinet':
            continue
        CONDITION = DIV / condition / 'train'
        CROP = CONDITION / 'crop'
        MAT = CONDITION / 'Mat'
        for label in os.listdir(CROP):
            CROP_LABEL = CROP / label
            MAT_LABEL = MAT / label
            for file in os.listdir(CROP_LABEL):
                FILE = CROP_LABEL / file
                SAVE_PATH = SAVE / 'crop' / label
                # print(FILE)
                # print(SAVE_PATH / (condition + "-" + file))
                check_path(SAVE_PATH)
                if os.path.isdir(FILE):
                    shutil.copytree(FILE, SAVE_PATH / (condition + "-" + file))
                else:
                    shutil.copyfile(FILE, SAVE_PATH / (condition + "-" + file))
            for file in os.listdir(MAT_LABEL):
                FILE = MAT_LABEL / file
                SAVE_PATH = SAVE / 'Mat' / label
                check_path(SAVE_PATH)
                if os.path.isdir(FILE):
                    shutil.copytree(FILE, SAVE_PATH / (condition + "-" + file))
                else:
                    shutil.copyfile(FILE, SAVE_PATH / (condition + "-" + file))