import numpy as np
import cv2
import os
from keras.utils import Sequence

class Loader(Sequence):
    def __init__(self, data_dir = "face_recognition\images", users = ["Tilen", "Tadej", "Tadej", "Randoms"], class_ids = range(4), batch_size=32, augment=False, debug=False, **kwargs):
        if debug: print("Loader(): __init__")
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.class_ids = class_ids
        self.users = users
        self.batch_size = batch_size
        self.augment = augment
        self.debug = debug
        self.image_paths = []
        self.labels = []

        if debug: print("Loader(): getting image paths and labels")
        for class_id in class_ids:
            class_dir = os.path.join(data_dir, f"{self.users[class_id]}", "processed")
            if os.path.exists(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith('.jpg'):
                        self.image_paths.append(os.path.join(class_dir, file))
                        self.labels.append(class_id)
                        print(f"Image: {class_dir}, Label: {class_id}")
        
        self.labels = np.array(self.labels)
        

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_labels = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]
        
        if self.debug: print(f"Loading batch {idx+1}/{len(self)} with {len(batch_paths)} images.")

        batch_images = []
        for path in batch_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64))
            img = img.astype(np.float32) / 255.0

            if False: #self.augment: #gonna fix later
                if np.random.rand() < 0.2:
                    h, w = img.shape[:2]
                    rain = np.zeros((h, w), dtype=np.float32)
                    for _ in range(50):
                        x1 = np.random.randint(0, w)
                        y1 = np.random.randint(0, h//2)
                        x2 = x1 + np.random.randint(-5, 5)
                        y2 = y1 + np.random.randint(h//2, h)
                        cv2.line(rain, (x1,y1), (x2,y2), 1.0, 1)
                    rain = cv2.GaussianBlur(rain, (5,5), 0)
                    img = img * (1 - 0.5*np.expand_dims(rain, -1))
                
                img = self.aug_transform(image=img)['image']
                
            batch_images.append(img)
            
        return np.array(batch_images), np.eye(len(self.class_ids))[batch_labels]
    
class SubLoader(Loader):
    def __init__(self, parent_gen, indices, augment=False):
        super().__init__(
            data_dir = parent_gen.data_dir, 
            users = parent_gen.users,
            class_ids = parent_gen.class_ids, 
            batch_size = parent_gen.batch_size, 
            augment=parent_gen.augment, 
            debug=parent_gen.debug
            )
        self.image_paths = [parent_gen.image_paths[i] for i in indices]
        self.labels = parent_gen.labels[indices]