from __future__ import division

import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd

import tensorflow as tf
from keras.preprocessing.image import Iterator
from keras.utils.np_utils import to_categorical
import keras.backend as K

def angle_difference(angle1, angle2):
    """
    Calculate the difference between two angles in the range [0, 1].
    """
    # Calculate the absolute values of angle differences
    angle_diff = tf.abs(angle1 - angle2)
    
    # If the angle difference exceeds 180°, subtract 360° to ensure the angle difference stays within the range [-180, 180]
    angle_diff = tf.where(angle_diff > 180, 360 - angle_diff, angle_diff)
    return angle_diff


def angle_error_regression(y_true, y_pred):
    """
    Calculate the mean angle difference between the true angles and the predicted angles.
    Each angle is represented as a (sin, cos) vector.
    """
    # Use the angle_difference function to calculate the angle difference for each sample
    angle_diff = angle_difference(y_true, y_pred)
    
    # Calculate the average angle difference
    return tf.reduce_mean(angle_diff)


def binarize_images(x):
    """
    Convert images to range 0-1 and binarize them by making
    0 the values below 0.1 and 1 the values above 0.1.
    """
    x /= 255
    x[x >= 0.1] = 1
    x[x < 0.1] = 0
    return x


def manual_train_test_split(X, y, test_size=0.2, random_seed=42):
    """
    Manually split the dataset into training and testing sets
    
    Args:
    - X: Input data (images)
    - y: Label data (rotation angles)
    - test_size: Proportion of the test set, default is 20%
    - random_seed: Random seed to ensure consistency in data splitting each time

    Returns:
    - X_train, X_test, y_train, y_test: The training and testing datasets after splitting
    """
    # Set the random seed
    np.random.seed(random_seed)

    # Calculate the size of the test set
    test_samples = int(len(X) * test_size)

    # Shuffle the data order
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Split the dataset
    X_train = X_shuffled[test_samples:]
    X_test = X_shuffled[:test_samples]
    y_train = y_shuffled[test_samples:]
    y_test = y_shuffled[:test_samples]

    return X_train, X_test, y_train, y_test


def load_and_pair_data(image_dir, csv_path, target_size):
    """
    Load images and pair them with their rotation angles to create a training dataset while also returning the original images.
    
    Args:
    - image_dir (str): Path to the image folder
    - csv_path (str): Path to the CSV file containing image names and rotation angles
    - target_size (tuple): Target image size, default is (128, 128)
    
    Returns:
    - images (numpy.ndarray): Modified image data, with shape as (num_images, height, width)
    - angles (numpy.ndarray): Rotation angle labels, with shape as (num_images,)
    - origin_images (numpy.ndarray): Original image data, with shape as (num_images, original_height, original_width)
    """
    # Read the CSV file to obtain file names and rotation angles
    df = pd.read_csv(csv_path)

    file_names = df['file_name'].values  # Obtain the list of file names
    rotation_angles = df['rotation_angle'].values  # Obtain the list of rotation angles

    images = []
    angles = []
    origin_images = []  # Store the original images

    # Iterate through each image file name
    for file_name, angle in zip(file_names, rotation_angles):
        img_path = os.path.join(image_dir, file_name)

        # Check if the file exists
        if os.path.exists(img_path):
            # Use cv2 to read the image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read the grayscale image

            if img is None:
                continue  # If the image fails to load, skip it

            # Store the original image
            origin_images.append(np.array(img))

            # Resize the image to the target size
            img = cv2.resize(img, target_size)  # Directly resize to the target size

            # Convert the modified image to a NumPy array and add it to the list
            images.append(np.array(img))

            # Add the rotation angle label
            angles.append(angle)

    # Convert the images and angles to NumPy arrays
    images = np.array(images)
    angles = np.array(angles)
    origin_images = np.array(origin_images)  # Convert the original images to NumPy arrays

    return images, angles, origin_images

def load_and_pair_data2(image_dir, csv_path, target_size):
    """
    Load images and pair them with their rotation angles to create a training dataset

    Args:
    - image_dir (str): Path to the image directory
    - csv_path (str): Path to the CSV file containing image names and rotation angles
    - target_size (tuple): Target image size, default is (128, 128)
    - crop_center (bool): Whether to crop the image to a square and center it, default is False

    Returns:
    - images (numpy.ndarray): Image data, with shape as (num_images, height, width)
    - angles (numpy.ndarray): Rotation angle labels, with shape as (num_images,)
    """
    # Read the CSV file to retrieve file names and rotation angles
    df = pd.read_csv(csv_path)

    file_names = df['file_name'].values  # List of file names
    rotation_angles = df['rotation_angle'].values  # List of rotation angles
    sides = df['carcass_side'].values # List of side labels

    images = []
    angles = []
    side_labels = []

    # Iterate through each image file name
    for file_name, angle, side in zip(file_names, rotation_angles,sides):
        img_path = os.path.join(image_dir, file_name)

        # Check if the file exists
        if os.path.exists(img_path):
            # Read the image using cv2
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read grayscale image

            if img is None:
                continue  # Skip if the image fails to load

            # Resize the image to the target size
            img = cv2.resize(img, target_size)  

            # Convert the image to a NumPy array and add it to the list
            images.append(np.array(img))

            # Add the rotation angle labels
            angles.append(angle)
            side_labels.append(side)

    # Convert the image and label lists to NumPy arrays
    images = np.array(images)
    angles = np.array(angles)
    side_labels = np.array(side_labels)

    return images, angles,side_labels

def load_and_pair_data3(image_dir, csv_path, target_size):
    """
    Load images and pair them with their corresponding rotation angles to create a training dataset

    Args:
    - image_dir (str): Path to the directory containing images
    - csv_path (str): Path to the CSV file containing image names and rotation angles
    - target_size (tuple): Target size for resizing images (default is (128, 128))

    Returns:
    - images (numpy.ndarray): Image data, shape (num_images, height, width, channels)
    - angles (numpy.ndarray): Rotation angle labels, shape (num_images,)
    - side_labels (numpy.ndarray): Side information of the images (e.g., 'Left' or 'Right' labels), with shape (num_images,)
    """
    import pandas as pd
    import numpy as np
    import cv2
    import os

    # Read the CSV file to get file names, rotation angles, and side labels
    df = pd.read_csv(csv_path)

    file_names = df['file_name'].values  # List of file names
    rotation_angles = df['rotation_angle'].values  # List of rotation angles
    sides = df['carcass_side'].values  # List of side labels

    images = []
    angles = []
    side_labels = []

    # Iterate through each image file name
    for file_name, angle, side in zip(file_names, rotation_angles, sides):
        img_path = os.path.join(image_dir, file_name)

        # Check if the file exists
        if os.path.exists(img_path):
            # Read the image using cv2 (three channels)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read color image

            if img is None:
                continue  # Skip if the image fails to load

            # Resize the image to the target size
            img = cv2.resize(img, target_size)  

            # Convert the image to a NumPy array and add it to the list
            images.append(np.array(img))

            # Add the rotation angle label
            angles.append(angle)
            side_labels.append(side)

    # Convert the image and label lists to NumPy arrays
    images = np.array(images)
    angles = np.array(angles)
    side_labels = np.array(side_labels)

    return images, angles, side_labels
