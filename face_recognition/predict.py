import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def predict_image(image_path, model_path="face_recognition\\best_model.h5"):
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
    confidence = np.max(pred)
    
    plt.imshow(img[0])  
    plt.title(f"Class: {class_id}, Confidence: {confidence:.2f}")
    plt.axis('off')
    plt.show()
    

predict_image("face_recognition\\images\\Tadej\\originals\\20250529_092030.jpg")