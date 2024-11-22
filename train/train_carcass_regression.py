from __future__ import print_function
import os
import sys
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


# Add project directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom functions and utilities
from utils import angle_error_regression, manual_train_test_split, load_and_pair_data 

# Set paths for image directory and CSV file
image_dir = r"C:\Users\xuyan\RotNet\data\RotationAngle\DATASET\STANDARDIZED\images"
csv_path = r"C:\Users\xuyan\RotNet\data\RotationAngle\DATASET\dataset_image_rotation_data.csv"

# Load and prepare the data
images, angles,origin_images = load_and_pair_data(image_dir, csv_path, target_size=(96, 128))

X_train, X_test, y_train, y_test = manual_train_test_split(images, angles, test_size=0.2)
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
# Split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Ensure data is in float32 format for compatibility with ImageDataGenerator
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

# Display data shapes for verification
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_val.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_val.shape}")

# Custom data generator for rotation augmentation
def custom_data_generator(X, y, batch_size, rotation_range=0):
    while True:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch_indices = indices[start:end]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            augmented_X = []
            augmented_y = []
            for i in range(len(batch_X)):
                image = batch_X[i]
                angle = batch_y[i]
                
                # Random rotation
                random_angle = random.uniform(-rotation_range, rotation_range)
                rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), random_angle, 1.0)
                rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
                
                augmented_X.append(rotated_image)
                new_angle = (angle + random_angle) % 360
                if new_angle > 180:
                    new_angle -= 360
                augmented_y.append(new_angle)
            
            yield np.array(augmented_X), np.array(augmented_y)

# Model configuration
model_name = 'rotnet_carcass_regression_color'


def create_optimized_model(input_shape=(128, 96, 1), nb_filters=128, kernel_size=(3, 3), dropout_rate=0.3):
    input_layer = Input(shape=input_shape)

    # First convolutional block with MaxPooling
    x = Conv2D(nb_filters, kernel_size, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)

    # Additional convolutional block for more complexity
    x = Conv2D(nb_filters * 2, kernel_size, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)

    # Second convolutional block with MaxPooling
    x = Conv2D(nb_filters * 2, kernel_size, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)


    # Fourth convolutional block with MaxPooling
    x = Conv2D(nb_filters * 8, kernel_size, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)

    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Output layer for angle regression
    output = Dense(1, activation='linear')(x)
    model = Model(inputs=input_layer, outputs=output)
    return model

# Initialize the model
model = create_optimized_model(input_shape=(128, 96, 1))
model.summary()


# Compile the model
optimizer = Adam(learning_rate=1e-3)
model.compile(loss=angle_error_regression, optimizer=optimizer)

# Training parameters
batch_size = 64
nb_epoch = 50

# Check if output folder exists and create it if not
output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define callbacks for training
checkpointer = ModelCheckpoint(filepath=os.path.join(output_folder, model_name + '.hdf5'), save_best_only=True)
early_stopping = EarlyStopping(patience=8)
tensorboard = TensorBoard()

# Train the model with custom data generator
model.fit(
    custom_data_generator(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=nb_epoch,
    validation_data=(X_val, y_val),
    shuffle=True,
    verbose=1,
    callbacks=[checkpointer, early_stopping, tensorboard]
)
