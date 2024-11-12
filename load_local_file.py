import os
import pandas as pd
import numpy as np
from PIL import Image

def load_and_pair_data(image_dir, csv_path):
    """
    加载图像并将它们与旋转角度配对，生成训练数据集。

    Args:
    - image_dir (str): 图像文件夹的路径
    - csv_path (str): 包含图像名称和旋转角度的 CSV 文件路径

    Returns:
    - images (numpy.ndarray): 图像数据，形状为 (num_images, 600, 800)
    - angles (numpy.ndarray): 旋转角度标签，形状为 (num_images,)
    """
    # 读取CSV文件，获取文件名和旋转角度
    df = pd.read_csv(csv_path)

    file_names = df['file_name'].values  # 获取文件名列表
    rotation_angles = df['rotation_angle'].values  # 获取旋转角度列表

    images = []
    angles = []

    # 遍历每个图像文件名
    for file_name, angle in zip(file_names, rotation_angles):
        img_path = os.path.join(image_dir, file_name)

        # 检查文件是否存在
        if os.path.exists(img_path):
            # 使用 PIL 打开图像，并进行必要的预处理（转换为灰度图、调整大小等）
            img = Image.open(img_path)
            img = img.convert('L')  # 转为灰度图
            img = img.resize((100, 75))  # 调整图像大小

            # 将图像转换为 NumPy 数组并添加到列表中
            images.append(np.array(img))

            # 添加旋转角度标签
            angles.append(angle)

    # 将图像和角度转为 NumPy 数组
    images = np.array(images)
    angles = np.array(angles)

    return images, angles

# 测试加载数据并配对
image_dir = r"C:\Users\xuyan\RotNet\data\RotationAngle\DATASET\STANDARDIZED\cut_masks"
csv_path = r"C:\Users\xuyan\RotNet\data\RotationAngle\DATASET\dataset_image_rotation_data.csv"

images, angles = load_and_pair_data(image_dir, csv_path)

# 打印一些基本信息
print(f"Loaded {len(images)} images.")
print(f"Sample image shape: {images[0].shape}")  # 打印一张图像的形状
print(f"Sample angle: {angles[0]}")  # 打印第一个图像的角度
