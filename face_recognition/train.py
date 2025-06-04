import os
import shutil
import uuid
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "images")
MODEL_PATH = os.path.join(BASE_DIR, "models")


def create_binary_dataset(username):
    temp_dir = os.path.join(BASE_DIR, "temp_dataset", username)
    pos_dir = os.path.join(temp_dir, "1")
    neg_dir = os.path.join(temp_dir, "0")

    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    user_images = []
    random_images = []

    for user_folder in os.listdir(DATA_PATH):
        user_path = os.path.join(DATA_PATH, user_folder, "processed")
        if not os.path.isdir(user_path):
            continue

        for img_file in os.listdir(user_path):
            src = os.path.join(user_path, img_file)
            if user_folder == username:
                user_images.append(src)
            else:
                random_images.append(src)

    min_count = min(len(user_images), len(random_images))
    user_images = random.sample(user_images, min_count)
    random_images = random.sample(random_images, min_count)

    for src in user_images:
        shutil.copy(src, os.path.join(pos_dir, f"{uuid.uuid4().hex}_{os.path.basename(src)}"))

    for src in random_images:
        shutil.copy(src, os.path.join(neg_dir, f"{uuid.uuid4().hex}_{os.path.basename(src)}"))

    return temp_dir


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_binary_model(username):
    print(f"[train] Training model for: {username}")
    temp_data_dir = create_binary_dataset(username)

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        temp_data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        temp_data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=True
    )

    model = build_model()
    os.makedirs(MODEL_PATH, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_PATH, f"{username}_model.h5"),
            save_best_only=True,
            monitor='val_loss'
        )
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )

    shutil.rmtree(temp_data_dir)
    print(f"[train] Model saved and temp data cleaned for: {username}")

    return history
