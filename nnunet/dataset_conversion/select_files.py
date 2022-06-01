import json
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from file_conversions import convert_2d_image_to_nifti
import random

TEST_CUT = 0.2 
SPLITS = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

if __name__ == '__main__':

    base = '/arc/project/st-rohling-1/data/kidney'
    repeats = "/arc/project/st-rohling-1/data/kidney/repeats.txt"

    # convert the training examples. Not all training images have labels, so we just take the cases for which there are
    # labels
    labels_dir = join(base, 'nnUNet_data/masks_jr/capsule')
    images_dir = join(base, 'images')
    all_files = subfiles(labels_dir, suffix='.png', join=False)

    with open(repeats) as fp:
        repeated_files = fp.readlines()
    print(repeated_files)
    print(len(all_files))
    cases = [x for x in all_files if x + '\n' not in repeated_files]
    print(len(cases))
    sz = len(cases)
    cut = int(0.20 * sz) #80% of the list
    random.shuffle(cases)
    testing_cases = cases[:cut]
    val_cases = cases[cut:2*cut]
    training_cases = cases[2*cut:]
    all_splits = {}
    for split in SPLITS:
        all_splits[split] = {"train": training_cases[0:int(split*len(training_cases))],
                             "val": val_cases[0:int(split*len(val_cases))],
                             "test": testing_cases}

    with open("test_cases.json", "w+") as fp:
        json.dump(all_splits, fp)
    