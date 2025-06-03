import os
import sys
import mediapipe as mp
import cv2

# SPREMENTA IME NA SVOJ
ime = sys.argv[1]  # Tilen Danijel Tadej

original_img_dir = f"images/{ime}/originals"
processed_img_dir = f"images/{ime}/processed"

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
