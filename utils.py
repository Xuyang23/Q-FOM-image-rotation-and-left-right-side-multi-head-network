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
    # 计算角度差的绝对值
    angle_diff = tf.abs(angle1 - angle2)
    
    # 如果角度差超过 180°，则通过减去 360° 来保证角度差在 [-180, 180] 范围内
    angle_diff = tf.where(angle_diff > 180, 360 - angle_diff, angle_diff)
    return angle_diff


def angle_error_regression(y_true, y_pred):
    """
    Calculate the mean angle difference between the true angles and the predicted angles.
    Each angle is represented as a (sin, cos) vector.
    """
    # 使用 angle_difference 函数计算每个样本的角度差异
    angle_diff = angle_difference(y_true, y_pred)
    
    # 计算平均角度差
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
    手动划分训练集和测试集。
    
    Args:
    - X: 输入数据（图像）
    - y: 标签数据（旋转角度）
    - test_size: 测试集占比，默认为 20%
    - random_seed: 随机种子，确保每次分割数据时的一致性

    Returns:
    - X_train, X_test, y_train, y_test: 分割后的训练集和测试集数据
    """
    # 设置随机种子
    np.random.seed(random_seed)

    # 计算测试集的大小
    test_samples = int(len(X) * test_size)

    # 打乱数据顺序
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # 分割数据集
    X_train = X_shuffled[test_samples:]
    X_test = X_shuffled[:test_samples]
    y_train = y_shuffled[test_samples:]
    y_test = y_shuffled[:test_samples]

    return X_train, X_test, y_train, y_test


def load_and_pair_data(image_dir, csv_path, target_size):
    """
    加载图像并将它们与旋转角度配对，生成训练数据集，同时返回原始图像。
    
    Args:
    - image_dir (str): 图像文件夹的路径
    - csv_path (str): 包含图像名称和旋转角度的 CSV 文件路径
    - target_size (tuple): 目标图像大小，默认为 (128, 128)
    
    Returns:
    - images (numpy.ndarray): 修改后的图像数据，形状为 (num_images, height, width)
    - angles (numpy.ndarray): 旋转角度标签，形状为 (num_images,)
    - origin_images (numpy.ndarray): 原始图像数据，形状为 (num_images, original_height, original_width)
    """
    # 读取 CSV 文件，获取文件名和旋转角度
    df = pd.read_csv(csv_path)

    file_names = df['file_name'].values  # 获取文件名列表
    rotation_angles = df['rotation_angle'].values  # 获取旋转角度列表

    images = []
    angles = []
    origin_images = []  # 存储原始图像

    # 遍历每个图像文件名
    for file_name, angle in zip(file_names, rotation_angles):
        img_path = os.path.join(image_dir, file_name)

        # 检查文件是否存在
        if os.path.exists(img_path):
            # 使用 cv2 读取图像
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像

            if img is None:
                continue  # 如果图像读取失败，跳过该图像

            # 存储原始图像
            origin_images.append(np.array(img))

            # 缩放图像到目标大小
            img = cv2.resize(img, target_size)  # 直接缩放到目标大小

            # 将修改后的图像转换为 NumPy 数组并添加到列表中
            images.append(np.array(img))

            # 添加旋转角度标签
            angles.append(angle)

    # 将图像和角度转为 NumPy 数组
    images = np.array(images)
    angles = np.array(angles)
    origin_images = np.array(origin_images)  # 将原始图像转为 NumPy 数组

    return images, angles, origin_images



