import numpy as np
import cv2
import os
from keras.utils import Sequence

def rain_augment(image, chance=0.2, intensity=0.5, debug=False):
    if np.random.rand() < chance:
        if debug: print("Applying rain augmentation")
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        for _ in range(10):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h//2)
            x2 = x1 + np.random.randint(-5, 5)
            y2 = y1 + np.random.randint(h//2, h)
            cv2.line(mask, (x1,y1), (x2,y2), 1.0, 1)
        mask = cv2.GaussianBlur(mask, (5,5), 0)
        image = image * (1 - intensity*np.expand_dims(mask, -1))
        if debug: cv2.imwrite("rain_augment_mask.jpg", (mask * 255).astype(np.uint8))
    return image
def vaseline_augment(image, chance=0.2, intensity=0.5, debug=False):
    if np.random.rand() < chance:
        if debug: print("Applying vaseline augmentation")
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        for _ in range(5):
            circleMask = np.zeros((h, w), dtype=np.float32)
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            cv2.circle(circleMask, (x1, y1), 15, (1.0, 1.0, 1.0), -1)
            mask = cv2.addWeighted(circleMask, 0.1, mask, 1 - 0.1, 0)
        image = image * (1 - intensity*np.expand_dims(mask, -1))
        if debug: cv2.imwrite("vaseline_augment_mask.jpg", (mask * 255).astype(np.uint8))
    return image
def broken_camera_augment(image, chance=0.2, intensity=0.5, debug=False):	
    if np.random.rand() < chance:
        if debug: print("Applying broken camera augmentation")
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        for _ in range(10):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            cv2.circle(mask, (x1, y1), 2, 1, -1)
        image = image * (1 - intensity*np.expand_dims(mask, -1))
        if debug: cv2.imwrite("broken_camera_augment_mask.jpg", (mask * 255).astype(np.uint8))
    return image
def covid_mask_augment(image, chance=0.2, debug=False):
    if np.random.rand() < chance:
        if debug: print("Applying COVID mask augmentation")
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, (h//3)*2), (w, h), 1.0, -1)
        
        if debug: 
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(mask, (0, (h//3)*2), (w, h), 1.0, -1)
            cv2.imwrite("covid_mask_augment_mask.jpg", (mask * 255).astype(np.uint8))
    return image

class Loader(Sequence):
    def __init__(self, data_dir = "face_recognition\images", users = ["Tilen", "Tadej", "Tadej", "Randoms"], class_ids = range(4), batch_size=32, augment=False, debug=False, image_size=64, **kwargs):
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
        self.image_size = image_size

        if debug: print("Loader(): getting image paths and labels")
        for class_id in class_ids:
            class_dir = os.path.join(data_dir, f"{self.users[class_id]}", "processed")
            if os.path.exists(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith('.jpg'):
                        self.image_paths.append(os.path.join(class_dir, file))
                        self.labels.append(class_id)
        
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
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img.astype(np.float32) / 255.0

            if self.augment:
                img = rain_augment(img, chance=0.2, intensity=0.5, debug = self.debug)
                img = vaseline_augment(img, chance=0.2, intensity=0.5, debug = self.debug)
                img = broken_camera_augment(img, chance=0.2, intensity=0.5, debug = self.debug)
                img = covid_mask_augment(img, chance=0.2, debug = self.debug)
                
            batch_images.append(img)
            
        return np.array(batch_images), np.eye(len(self.class_ids))[batch_labels]
    
class SubLoader(Loader):
    def __init__(self, parent_gen, indices):
        super().__init__(
            data_dir = parent_gen.data_dir, 
            users = parent_gen.users,
            class_ids = parent_gen.class_ids, 
            batch_size = parent_gen.batch_size, 
            augment=parent_gen.augment, 
            debug=parent_gen.debug,
            image_size=parent_gen.image_size
            )
        self.image_paths = [parent_gen.image_paths[i] for i in indices]
        self.labels = parent_gen.labels[indices]