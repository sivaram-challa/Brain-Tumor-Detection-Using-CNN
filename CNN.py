import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split


def build_cnn(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def load_and_preprocess_images(base_folder, img_size):
    images = []
    labels = []
    # Iterate over 'Training' and 'Testing' folders
    for dataset_type in ['Training', 'Testing']:
        dataset_folder = os.path.join(base_folder, dataset_type)
        for label in os.listdir(dataset_folder):
            label_folder = os.path.join(dataset_folder, label)
            if not os.path.isdir(label_folder):  # Skip non-directory files
                continue
            for img_name in os.listdir(label_folder):
                img_path = os.path.join(label_folder, img_name)
                if os.path.isfile(img_path):  # Ensure that img_path is a file
                    try:
                        img = load_img(img_path, target_size=img_size)
                        img_array = img_to_array(img) / 255.0
                        images.append(img_array)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    # Convert labels to numeric values
    label_dict = {label: idx for idx, label in enumerate(np.unique(labels))}
    labels = np.vectorize(label_dict.get)(labels)
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_dict))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

    return X_train, X_test, y_train, y_test


def train_model(cnn_model, X_train, X_test, y_train, y_test, epochs=20, batch_size=32):
    cnn_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss=losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
    history = cnn_model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test))
    return history


# Paths and parameters
base_folder = 'C:\\Users\\Sivaram\\PycharmProjects\\BTD\\archive'
input_shape = (224, 224, 3)
num_classes = 4

# Load and preprocess images
X_train, X_test, y_train, y_test = load_and_preprocess_images(base_folder, input_shape[:2])

# Define and compile the CNN model
cnn_model = build_cnn(input_shape, num_classes)

# Train the model
history = train_model(cnn_model, X_train, X_test, y_train, y_test, epochs=20, batch_size=32)
