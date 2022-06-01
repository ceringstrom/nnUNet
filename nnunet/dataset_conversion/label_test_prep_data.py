import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from utils import generate_dataset_json
from nnunet.paths import preprocessing_output_dir
from file_conversions import convert_2d_image_to_nifti
import random

base = '/arc/project/st-rohling-1/data/kidney/'

def prep_task_folder(training_cases, testing_cases, target_base, labels, task, task_name):
    randomized = False

    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)


    labels_dir = join(base, 'nnUNet_data/masks_jr', task)

    images_dir = join(base, 'cleaned_annotated_images/included')

    i=0
    for t in training_cases:
        unique_name = t[:-4]  # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        input_segmentation_file = join(labels_dir, t)
        input_image_file = join(images_dir, t)

        output_image_file = join(target_imagesTr, unique_name)  # do not specify a file ending! This will be done for you
        output_seg_file = join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=True)

        # the labels are stored as 0: background, 255: road. We need to convert the 255 to 1 because nnU-Net expects
        # the labels to be consecutive integers. This can be achieved with setting a transform
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True)

        i += 1
    j=0
    for ts in testing_cases:
        if randomized:
            labels_dir = labels_dirs[testing_selection[j]]
        unique_name = ts[:-4]
        input_segmentation_file = join(labels_dir, ts)
        input_image_file = join(images_dir, ts)

        output_image_file = join(target_imagesTs, unique_name)
        output_seg_file = join(target_labelsTs, unique_name)

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)

        # the labels are stored as 0: background, 255: road. We need to convert the 255 to 1 because nnU-Net expects
        # the labels to be consecutive integers. This can be achieved with setting a transform
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True)

        j += 1

    # labels = {0: "background", 1:"Central Echo Complex", 2:"Medulla", 3:"Cortex"}

    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('greyscale'),
                          labels=labels, dataset_name=task_name, license='hands off!')

if __name__ == '__main__':
    with open("test_cases.json", "r") as fp:
        all_cases = json.load(fp)

    cap_labels = {0: "background", 1:"Capsule"}
    reg_labels = {0: "background", 1:"Central Echo Complex", 2:"Medulla", 3:"Cortex"}

    for split in all_cases.keys():
        split_dict = all_cases[split]
        training_cases = split_dict['train'] + split_dict['val']
        testing_cases = split_dict['test']
        cap_target_base = join(base, 'nnUNet_data/data_splits', 'raw_data_split_{}'.format(int(100*float(split))), 'nnUNet_raw_data', 'Task001_KidneyCapsule')
        reg_target_base = join(base, 'nnUNet_data/data_splits', 'raw_data_split_{}'.format(int(100*float(split))), 'nnUNet_raw_data', 'Task002_KidneyRegions')
        print(cap_target_base)
        print(reg_target_base)
        prep_task_folder(training_cases, testing_cases, cap_target_base, cap_labels, 'capsule', 'Task001_KidneyCapsule')
        prep_task_folder(training_cases, testing_cases, reg_target_base, reg_labels, 'regions', 'Task002_KidneyRegions')
        del split_dict['test']
        split_dict['train'] = [x.replace('.png', '') for x in split_dict['train']]
        split_dict['val'] = [x.replace('.png', '') for x in split_dict['val']]
        splits = [split_dict, split_dict, split_dict, split_dict, split_dict]
        save_pickle(splits, join(cap_target_base, 'splits_final.pkl'))
        save_pickle(splits, join(reg_target_base, 'splits_final.pkl'))
