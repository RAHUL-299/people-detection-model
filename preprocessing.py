#After downloading raw images 
import os
import numpy as np
import cv2

# Paths for images and masks
images_dir = "path/for/images folder"  # Replace with the local path
masks_dir = "path/for/masks folder"    # Replace with the local path

# Hyperparameters
IMG_SIZE = (256, 256)  # Resize to 256x256
MAX_SAMPLES = 4200     # Limit dataset to 4200 samples

# Data preprocessing function
def preprocess_image(img_path, mask_path):
    # Load image and mask
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    # Resize image and mask to fixed size
    image = cv2.resize(image, IMG_SIZE)
    mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    
    # Normalize image
    image = image / 255.0
    
    # Expand mask dimensions for segmentation (e.g., add channel dimension)
    mask = np.expand_dims(mask, axis=-1)  # Shape: (height, width, 1)
    
    return image, mask

# Prepare dataset
def prepare_dataset(images_dir, masks_dir, max_samples):
    images = sorted(os.listdir(images_dir))[:max_samples]  # Limit to max_samples
    masks = sorted(os.listdir(masks_dir))[:max_samples]   # Limit to max_samples
    
    image_list = []
    mask_list = []
    
    for img_name, msk_name in zip(images, masks):
        img_path = os.path.join(images_dir, img_name)
        msk_path = os.path.join(masks_dir, msk_name)
        img, mask = preprocess_image(img_path, msk_path)
        image_list.append(img)
        mask_list.append(mask)
    
    return np.array(image_list), np.array(mask_list)

# Load data with the limit of 4200 samples
X, y = prepare_dataset(images_dir, masks_dir, MAX_SAMPLES)

# Save the preprocessed data locally in a compressed format
np.savez_compressed('preprocessed_data.npz', X=X, y=y)
print("Data preprocessing complete and saved as 'preprocessed_data.npz'")
