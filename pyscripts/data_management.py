"""
data management script for remote sensing transfer learning 
"""
import os
from distutils.dir_util import copy_tree
import random
from random import shuffle
from shutil import copyfile
from shutil import rmtree

import numpy as np

import Augmentor



def split_data(folder_in, fraction_test, fraction_validation):
    """
    splits the data randomly into train, validation, test
    :param folder_in:
    :param fraction_test:
    :param fraction_validation:
    :return:
    """

    # create test folder:
    test_dir = os.path.join(os.path.dirname(folder_in), f"{os.path.split(folder_in)[-1]}_test")
    rmtree(test_dir, ignore_errors=True)
    os.mkdir(test_dir)

    # create validation folder:
    validation_dir = os.path.join(os.path.dirname(folder_in), f"{os.path.split(folder_in)[-1]}_validation")
    rmtree(validation_dir, ignore_errors=True)
    os.mkdir(validation_dir)

    # create  folder:
    train_dir = os.path.join(os.path.dirname(folder_in), f"{os.path.split(folder_in)[-1]}_train")
    rmtree(train_dir, ignore_errors=True)
    os.mkdir(train_dir)

    # get all folders:
    list_classes = os.listdir(folder_in)

    for i_dir in list_classes:

        imgs = os.listdir(os.path.join(folder_in, i_dir))
        num_samples = len(imgs)
        # shuffle images:
        shuffle(imgs)

        # create test folder:
        os.mkdir(os.path.join(test_dir, i_dir))
        # copy data to test folder:
        for ii in range(np.int(np.ceil(fraction_test * num_samples))):
            img = imgs.pop()
            copyfile(os.path.join(folder_in, i_dir, img), os.path.join(test_dir, i_dir, img))

        # create validation folder:
        os.mkdir(os.path.join(validation_dir, i_dir))
        # copy data to validation folder:
        for ii in range(np.int(np.ceil(fraction_validation * num_samples))):
            img = imgs.pop()
            copyfile(os.path.join(folder_in, i_dir, img), os.path.join(validation_dir, i_dir, img))

        # remaining images will go into training:
        # create training folder:
        os.mkdir(os.path.join(train_dir, i_dir))
        # copy data to test folder:
        train_samples = len(imgs)
        for img in imgs:
            copyfile(os.path.join(folder_in, i_dir, img), os.path.join(train_dir, i_dir, img))

        print(f"{i_dir:20} -- Training: ", end=" ")
        print(f"{train_samples:5d} "
              f" -- Validation: {np.int(np.ceil(fraction_validation * num_samples)):5d} "
              f" -- Test: {np.int(np.ceil(fraction_test * num_samples)):5d}")

def small_split_data(folder_in, fraction):
    """
    selects a fraction of the data in folder in 
    :param folder_in:
    :param fraction:
    :return:
    """

    # create small folder:
    small_dir = os.path.join(os.path.dirname(folder_in), f"{os.path.split(folder_in)[-1]}_small")
    rmtree(small_dir, ignore_errors=True)
    os.mkdir(small_dir)

    # get all folders:
    list_classes = os.listdir(folder_in)

    for i_dir in list_classes:
        imgs = os.listdir(os.path.join(folder_in, i_dir))
        num_samples = len(imgs)
        # shuffle images:
        shuffle(imgs)

        # create class in small folder:
        os.mkdir(os.path.join(small_dir, i_dir))
        # copy data to smal folder:
        for ii in range(np.int(np.ceil(fraction * num_samples))):
            img = imgs.pop()
            copyfile(os.path.join(folder_in, i_dir, img), os.path.join(small_dir, i_dir, img))
            
        print(f"{i_dir:20} -- Data: ", end=" ")
        print(f"{np.int(np.ceil(fraction * num_samples)):5d}")

def data_augment(folder_in, folder_out, num_samples, img_wid=256, img_hei=256):
    """
    data augmentation
    :param folder_in: folder with data to be sampled (with subfolders)
    :param folder_out: output folder (data augmented folder)
    :param num_samples: number of samples for each one of the subfolders
    :param img_wid: width dimension of augmented image
    :param img_hei: height dimension of augmented image
    :return:
    """
    # create output folder
    rmtree(folder_out, ignore_errors=True)
    os.mkdir(folder_out)

    # get all folders:
    list_classes = os.listdir(folder_in)
    # loop through all the subfolders
    for i_dir in list_classes:
        # create folder for current class:
        os.mkdir(os.path.join(folder_out, i_dir))

        # use data augmentation pipeline
        p = Augmentor.Pipeline(os.path.join(folder_in, i_dir))
        p.rotate180(probability=0.5)
        p.flip_left_right(probability=0.5)
        p.flip_top_bottom(probability=0.5)
        p.rotate(probability=0.75, max_left_rotation=25, max_right_rotation=25)
        p.random_color(probability=0.5, min_factor=0.01, max_factor=0.8)
        p.random_contrast(probability=0.5, min_factor=0.70, max_factor=1.1)
        p.random_brightness(probability=0.5, min_factor=0.70, max_factor=1.1)
        p.crop_centre(probability=0.5, percentage_area=0.90)
        p.resize(probability=1, width=img_wid, height=img_hei)
        p.sample(num_samples)

        # Augmentator creates a folder called "output", move the folder to the desired location
        copy_tree(os.path.join(folder_in, i_dir, "output"), os.path.join(folder_out, i_dir))
        rmtree(os.path.join(folder_in, i_dir, "output"), ignore_errors=True)


if __name__ == "__main__":
    # set the seed for repetition purposes 
    random.seed(0)
    
    data_dir = '../data'
    datasets_dir = [os.path.join(data_dir, 'AID', 'images'),
                    os.path.join(data_dir, 'PatternNet', 'images'), 
                    os.path.join(data_dir, 'UCMerced_LandUse', 'images')]

    # split data in train, validation, test (70%, 10%, 20%)
    for folder_in in datasets_dir:
        # split data:
        print(f"-------------------------------------------------------------")
        print(f"{folder_in}: \n")
        split_data(folder_in, fraction_test=0.20, fraction_validation=0.10)

    # create small set for PatterNet:
    print(f"-------------------------------------------------------------")
    small_split_data(os.path.join(os.path.split(datasets_dir[1])[0], 
                                  'images_train'), 
                     fraction=0.114)
    # create augmented set for small PatterNet:
    print(f"-------------------------------------------------------------")
    data_augment(os.path.join(os.path.split(datasets_dir[1])[0], 
                                  'images_train_small'), 
                os.path.join(os.path.split(datasets_dir[1])[0], 
                                  'images_train_aug'),
                num_samples=640)

