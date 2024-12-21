from __future__ import print_function
import os
import sys
import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# Add project directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom functions and utilities
from utils import angle_error_regression, load_and_pair_data3 

# Set paths for image directory and CSV file
image_dir = r"C:\Users\13051\Desktop\RotationAngle\RotationAngle\DATASET\STANDARDIZED\cut_masks"
csv_path = r"C:\Users\13051\Desktop\RotationAngle\RotationAngle\DATASET\dataset_image_rotation_data.csv"

# Load and prepare the data
images, angles, labels = load_and_pair_data3(image_dir, csv_path, target_size=(96, 128))

# Convert labels to binary: 'Left' -> 0, 'Right' -> 1
labels = np.array(labels, dtype=str)
binary_labels = np.where(labels == 'Left', 0, 1)
one_hot_labels = to_categorical(binary_labels, num_classes=2)

# Split data into training and testing sets for both tasks
X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val = train_test_split(
    images, angles, one_hot_labels, test_size=0.2, random_state=42
)

# Ensure data is in float32 format
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')


## Split training set further into training and validation sets
##X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val = train_test_split(
 ##   X_train, y_reg_train, y_cls_train, test_size=0.2, random_state=42
#)

# Display data shapes for verification
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Training labels shape (regression): {y_reg_train.shape}")
print(f"Training labels shape (classification): {y_cls_train.shape}")

# Define the multi-head model
def create_multihead_model(input_shape=(128, 96, 3), nb_filters=128, kernel_size=(3, 3), dropout_rate=0.2):
    input_layer = Input(shape=input_shape)

    # Shared convolutional base
    x = Conv2D(nb_filters, kernel_size, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(nb_filters * 2, kernel_size, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)


    # Flatten the output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Regression head for predicting angles
    reg_output = Dense(1, activation='linear', name='regression_output')(x)

    # Classification head for predicting Left/Right
    cls_output = Dense(2, activation='softmax', name='classification_output')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=[reg_output, cls_output])
    return model

# Initialize the multi-head model
multihead_model = create_multihead_model(input_shape=(128, 96, 3))

# Compile the model
optimizer = Adam(learning_rate=1e-4)
multihead_model.compile(
    loss={'regression_output': angle_error_regression, 'classification_output': 'categorical_crossentropy'},
    optimizer=optimizer,
    metrics={'classification_output': 'accuracy'}
)

# Display the model summary
multihead_model.summary()

# Training parameters
batch_size = 32
nb_epoch = 60

# Check if output folder exists and create it if not
output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define callbacks for training
checkpointer = ModelCheckpoint(filepath=os.path.join(output_folder, 'multihead_model_standardized_cutmusk_STANDARDIZED.hdf5'), save_best_only=True)
early_stopping = EarlyStopping(patience=10)
tensorboard = TensorBoard()

# Train the model
multihead_model.fit(
    X_train,
    {'regression_output': y_reg_train, 'classification_output': y_cls_train},
    validation_data=(X_val, {'regression_output': y_reg_val, 'classification_output': y_cls_val}),
    epochs=nb_epoch,
    batch_size=batch_size,
    callbacks=[checkpointer, early_stopping, tensorboard],
    verbose=1
)
