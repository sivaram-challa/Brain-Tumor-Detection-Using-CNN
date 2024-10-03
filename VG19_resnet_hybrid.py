import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split


def build_vgg19(input_tensor):
    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    return x


def resnet_block(input_tensor, filters, kernel_size=3, stride=1):
    x = layers.Conv2D(filters, kernel_size, padding='same', strides=stride)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if stride != 1 or input_tensor.shape[-1] != filters:
        input_tensor = layers.Conv2D(filters, (1, 1), padding='same', strides=stride)(input_tensor)
        input_tensor = layers.BatchNormalization()(input_tensor)
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def build_resnet(input_tensor):
    x = layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Define a few ResNet blocks
    for filters in [64, 128, 256, 512]:
        strides = 1 if filters == 64 else 2
        x = resnet_block(x, filters, stride=strides)

    return x


def build_hybrid_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    vgg19_output = build_vgg19(inputs)
    resnet_output = build_resnet(inputs)

    combined_output = layers.concatenate([vgg19_output, resnet_output], axis=-1)
    x = layers.GlobalAveragePooling2D()(combined_output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    hybrid_model = models.Model(inputs=inputs, outputs=output_layer)
    return hybrid_model


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


def train_model(hybrid_model, X_train, X_test, y_train, y_test, epochs=20, batch_size=32):
    hybrid_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                         loss=losses.CategoricalCrossentropy(),
                         metrics=['accuracy'])
    history = hybrid_model.fit(X_train, y_train,
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

# Define and compile the hybrid model
hybrid_model = build_hybrid_model(input_shape, num_classes)

# Train the model
history = train_model(hybrid_model, X_train, X_test, y_train, y_test, epochs=20, batch_size=32)
