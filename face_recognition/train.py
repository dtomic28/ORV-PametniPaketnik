import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from getImageLoaders import get_image_loaders

# --- Parse command line argument ---
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

print("Train class distribution:", Counter(train_loader.labels))
print("Val class distribution:", Counter(val_loader.labels))

def build_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()
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