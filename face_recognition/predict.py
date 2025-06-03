import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def predict_image(image_path, model_path="face_recognition\\_model.h5"):
    print("Current working directory:", os.getcwd())
    
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
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py IMAGE_PATH [MODEL_PATH]")
        sys.exit(1)
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    print(predict_image(image_path, model_path))