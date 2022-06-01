import json
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from file_conversions import convert_2d_image_to_nifti
import random
if __name__ == '__main__':
    base = '/arc/project/st-rohling-1/data/kidney'
    repeats = "/arc/project/st-rohling-1/data/kidney/repeats.txt"

    # convert the training examples. Not all training images have labels, so we just take the cases for which there are
    # labels
    labels_dir = join(base, 'nnUNet_data/masks_vl/capsule')
    images_dir = join(base, 'images')
    all_files = subfiles(labels_dir, suffix='.png', join=False)

    with open(repeats) as fp:
        repeated_files = fp.readlines()

    # print(all_files)
    print(len(all_files))
    print(all_files[0])
    print(repeated_files)
    cases = [x for x in all_files if x + '\n' not in repeated_files]
    print(len(cases))
    # print(cases)
    sz = len(cases)
    cut = int(0.80 * sz) #80% of the list
    random.shuffle(cases)
    training_cases = cases[:cut] # first 80% of shuffled list
    testing_cases = cases[cut:] # last 20% of shuffled list
    test_cases = {"train": training_cases, "test": testing_cases}
    with open("split_80.json", "w+") as fp:
        json.dump(test_cases, fp)