import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

from getImageLoaders import get_image_loaders


if len(sys.argv) < 2:
    print("Usage: python train.py USERNAME")
    sys.exit(1)

USER = sys.argv[1]
USERS = [USER, "Randoms"]
MODEL_SAVE_PATH = os.path.join("models", f"{USER}_model.h5")

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50

train_loader, val_loader = get_image_loaders(
    debug=False,
    augment=True,
    batch_size=BATCH_SIZE,
    users=USERS,
    image_size=IMAGE_SIZE
)
NUM_CLASSES = len(USERS)


# Plot 5 sample images from the training set with their class names
plt.figure(figsize=(15, 3))
for i in range(5):
    img, label = train_loader[i]
    class_idx = np.argmax(label)
    class_name = USERS[class_idx]
    plt.subplot(1, 5, i + 1)
    plt.imshow(img[0])  # <-- remove .astype(np.uint8)
    plt.title(f"Class: {class_idx}")
    plt.axis('off')
plt.suptitle("Sample Training Images and Their Classes")
plt.show()

def build_transfer_model(input_shape=(128, 128, 3), num_classes=2):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_transfer_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=NUM_CLASSES)
model.summary()

callbacks = [
    EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True),
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1)
]

history = model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=EPOCHS,
    callbacks=callbacks
)




# Plot loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.tight_layout()
plt.show()

x_vals, y_vals = [], []
for x_batch, y_batch in val_loader:
    x_vals.append(x_batch)
    y_vals.append(y_batch)
x_val = np.concatenate(x_vals, axis=0)
y_val = np.concatenate(y_vals, axis=0)

y_pred = np.argmax(model.predict(x_val), axis=1)
y_true = np.argmax(y_val, axis=1)
cm = confusion_matrix(y_true, y_pred, labels=range(len(USERS)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=USERS)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
