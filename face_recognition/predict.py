import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import mediapipe as mp

def predict_image(image_path, model_path, threshold=0.5):
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Detect face (same as training preprocessing)
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if not results.detections:
            print("[predict] ❌ No face detected.")
            return -1  # Could not detect face

        # Process the first face only
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        x, y, w, h = (
            int(bboxC.xmin * iw),
            int(bboxC.ymin * ih),
            int(bboxC.width * iw),
            int(bboxC.height * ih),
        )
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, (128, 128))
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_rgb = face_rgb.astype(np.float32) / 255.0
        face_rgb = np.expand_dims(face_rgb, axis=0)

        # Prediction
        pred = model.predict(face_rgb)
        prob = pred[0][0]
        result = int(prob > threshold)

        print(f"[predict] 🔍 Probability: {prob:.4f}")
        print(f"[predict] ✅ Predicted: {'USER' if result == 1 else 'RANDOM'} (threshold = {threshold})")

        return result