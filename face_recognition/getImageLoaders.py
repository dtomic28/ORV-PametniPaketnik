import os
import numpy as np
from sklearn.model_selection import train_test_split
from ImageLoader import Loader, SubLoader

def get_image_loaders(debug=False, augment=False, batch_size=32, users=["Tilen", "Danijel", "Tadej", "Randoms"],):
    data_dir = os.path.join(os.path.dirname(__file__), 'images')
    
    if debug: print("get_image_loaders(): initializing data loader")
    full_gen = Loader(
        data_dir, 
        class_ids=range(users.__len__()),
        users=users,
        debug=debug, 
        batch_size=batch_size
        )
    
    if debug: print("get_image_loaders(): splitting data")
    indices = np.arange(len(full_gen.image_paths))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=full_gen.labels,
        random_state=42
    )
    
    if debug: print("get_image_loaders(): creating sub-loaders")
    return SubLoader(full_gen, train_idx), \
           SubLoader(full_gen, val_idx)

if __name__ == "__main__":
    print("----------------------------------------------------------------------------------------------------")
    train_loader, val_loader = get_image_loaders(debug=True, augment=True, batch_size=32)
    print(f"Train loader size: {len(train_loader)} batches")
    print(f"Validation loader size: {len(val_loader)} batches")
    
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        break  # Just to test the first batch