from __future__ import print_function
import sys
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Add project directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom functions and utilities
from utils import manual_train_test_split, load_and_pair_data 

# Set paths for image directory and CSV file
image_dir = r"C:\Users\xuyan\RotNet\data\RotationAngle\DATASET\STANDARDIZED\cut_masks"
csv_path = r"C:\Users\xuyan\RotNet\data\RotationAngle\DATASET\dataset_image_rotation_data.csv"

# Load and prepare the data
images, labels = load_and_pair_data(image_dir, csv_path, target_size=(96, 128))

# Convert labels to a NumPy array of strings for compatibility
labels = np.array(labels, dtype=str)

# Convert 'carcass_side' labels to binary: 'Left' -> 0, 'Right' -> 1
binary_labels = np.where(labels == 'Left', 0, 1)

# Encode the labels as one-hot vectors
one_hot_labels = to_categorical(binary_labels, num_classes=2)  # Ensure labels are (batch_size, 2) for binary classification

# Split data into training and testing sets
X_train, X_test, y_train, y_test = manual_train_test_split(images, one_hot_labels, test_size=0.2)

# Ensure data is in float32 format for compatibility with ImageDataGenerator
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Ensure the data is 4-dimensional: (num_samples, height, width, channels)
# Convert data from (1870, 128, 96) to (1870, 128, 96, 1), assuming grayscale
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Split training set further into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Display data shapes for verification
print(f"Training data shape: {X_train.shape}")  # Should be (batch, 128, 96, 1)
print(f"Validation data shape: {X_val.shape}")  # Should be (batch, 128, 96, 1)
print(f"Testing data shape: {X_test.shape}")    # Should be (batch, 128, 96, 1)
print(f"Training labels shape: {y_train.shape}")
print(f"Validation labels shape: {y_val.shape}")
print(f"Testing labels shape: {y_test.shape}")

# Model configuration
model_name = 'rotnet_carcass_classification'

def create_classification_model(input_shape=(128, 96, 1), nb_filters=128, kernel_size=(3, 3), dropout_rate=0.01):
    input_layer = Input(shape=input_shape)

    # First convolutional block with MaxPooling
    x = Conv2D(nb_filters, kernel_size, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)

    # Second convolutional block with MaxPooling
    x = Conv2D(nb_filters * 2, kernel_size, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)

    # Third convolutional block with MaxPooling
    x = Conv2D(nb_filters * 4, kernel_size, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)

    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Output layer for classification
    output = Dense(2, activation='softmax')(x)  # 2 classes: 'Left' and 'Right'
    model = Model(inputs=input_layer, outputs=output)
    return model

# Initialize the classification model
classification_model = create_classification_model(input_shape=(128, 96, 1))

# Compile the model with categorical crossentropy loss and Adam optimizer
classification_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# Display the updated model summary
classification_model.summary()

# Training parameters
batch_size = 16
nb_epoch = 10

# Check if output folder exists and create it if not
output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define callbacks for training
checkpointer = ModelCheckpoint(filepath=os.path.join(output_folder, model_name + '.hdf5'), save_best_only=True)
early_stopping = EarlyStopping(patience=8)
tensorboard = TensorBoard()

# Use ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    fill_mode='constant'
)

# No augmentation for validation data, only normalization
val_datagen = ImageDataGenerator()

# Create data generators for training and validation
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

# Train the model using the data generator
classification_model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=nb_epoch,
    validation_data=val_generator,
    validation_steps=len(X_val) // batch_size,
    verbose=1,
    callbacks=[checkpointer, early_stopping, tensorboard]
)
