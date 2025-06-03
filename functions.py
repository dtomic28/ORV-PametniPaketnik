import os
from matplotlib import pyplot as plt
import mediapipe as mp
import cv2
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import tensorflow as tf

from face_recognition.getImageLoaders import get_image_loaders
from face_recognition.ImageLoader import Loader, SubLoader

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

#TODO NE POZABI NEKAK DODAT DA SE ORIGINALS POPULIRA S SLIKAMI
def preprocess(ime):
    original_img_dir = f"face_recognition/images/{ime}/originals"
    processed_img_dir = f"face_recognition/images/{ime}/processed"

    os.makedirs(original_img_dir, exist_ok=True)
    os.makedirs(processed_img_dir, exist_ok=True)

    #TODO populate the originals here 

    print("Looking for images in:", os.path.abspath(original_img_dir))

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    for j in range (0,3):
        for filename in os.listdir(original_img_dir):
            print("Checking file:", filename)
            file_path = os.path.join(original_img_dir, filename)
            if filename.lower().endswith((".jpg", ".jpeg", ".jpg", ".JPEG")):
                try:
                    image = cv2.imread(file_path)
                    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    with mp_face_detection.FaceDetection(
                        model_selection=1, min_detection_confidence=0.5
                    ) as face_detection:
                        results = face_detection.process(image_rgb)

                        if results.detections:
                            print(f"Detected {len(results.detections)} face(s) in {filename}")
                            for i, detection in enumerate(results.detections):
                                bboxC = detection.location_data.relative_bounding_box
                                ih, iw, _ = image.shape
                                x, y, w, h = (
                                    int(bboxC.xmin * iw),
                                    int(bboxC.ymin * ih),
                                    int(bboxC.width * iw),
                                    int(bboxC.height * ih),
                                )
                                face = image[y : y + h, x : x + w]
                                face = cv2.resize(face, (128, 128))
                                cv2.imwrite(os.path.join(processed_img_dir, (str)(j)+"_"+(str)(filename)), face)
                        else:
                            print(f"No face detected in {filename}")
                except Exception as e:
                    print(f"Failed to open {filename}: {e}")

def train(user):
    USERS = [user, "Randoms"]
    MODEL_SAVE_PATH = os.path.join("face_recognition/models", f"{user}_model.h5")

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
    """
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
    """

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
    """
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
    """

def predict_image(image_path, model_path):
    model = tf.keras.models.load_model(model_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0) 
    pred = model.predict(img)
    class_id = np.argmax(pred)
    return class_id

if __name__ == '__main__':
    preprocess("Tilen")  #primer 
    train("Tilen") # primer
    print(predict_image("face_recognition/images/Tilen/originals/IMG_20250527_194319.jpg", "face_recognition/models/Tilen_model.h5"))  # primer
    #vrne 0 ce je pravilno 1 če je napačno