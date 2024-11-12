from __future__ import print_function

import os
import sys
import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input, Conv2D,MaxPooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


# Add project directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom functions and utilities
from utils import angle_error_regression, manual_train_test_split, load_and_pair_data,RotNetDataGenerator,binarize_images

# Set paths for image directory and CSV file
image_dir = r"C:\Users\xuyan\RotNet\data\RotationAngle\DATASET\STANDARDIZED\cut_masks"
csv_path = r"C:\Users\xuyan\RotNet\data\RotationAngle\DATASET\dataset_image_rotation_data.csv"

# Load and prepare the data
images, angles = load_and_pair_data(image_dir, csv_path, target_size=(96, 128))



X_train, X_test, y_train, y_test = manual_train_test_split(images, angles, test_size=0.2)

# Ensure data is in float32 format for compatibility with ImageDataGenerator
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Display data shapes for verification
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")

# Model configuration
model_name = 'rotnet_carcass_regression'


def create_optimized_model(input_shape=(128, 96, 1), nb_filters=128, kernel_size=(5, 5), dropout_rate=0.01):
    """
    Creates a more complex CNN model for regression with added pooling layers, BatchNormalization, and Dropout.
    The model is designed to handle a larger dataset.
    """
    input_layer = Input(shape=input_shape)

    # First convolutional block with MaxPooling
    x = Conv2D(nb_filters, kernel_size, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # 添加 MaxPooling 层
    x = Dropout(dropout_rate)(x)

    # Second convolutional block with MaxPooling
    x = Conv2D(nb_filters * 2, kernel_size, activation='relu')(x)  # Increased filters
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # 添加 MaxPooling 层
    x = Dropout(dropout_rate)(x)

    # Third convolutional block with MaxPooling
    x = Conv2D(nb_filters * 4, kernel_size, activation='relu')(x)  # Increased filters
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # 添加 MaxPooling 层
    x = Dropout(dropout_rate)(x)

    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)  # Increased neurons
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Output layer for angle regression
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output)
    return model

# Initialize the model
model = create_optimized_model(input_shape=(128, 96, 1))
model.summary()

optimizer = Adam(learning_rate=1e-4)
model.compile(loss=angle_error_regression, optimizer=optimizer)


# Training parameters
batch_size = 32
nb_epoch = 50

# Check if output folder exists and create it if not
output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define callbacks for training
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, model_name + '.hdf5'),
    save_best_only=True
)
early_stopping = EarlyStopping(patience=8)
tensorboard = TensorBoard()

# Model training
model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_data=(X_test, y_test),
    shuffle=True,
    verbose=1,
    callbacks=[checkpointer, early_stopping, tensorboard]
)
