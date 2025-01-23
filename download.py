#To download raw images and masks from coco
import os
import requests
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
from io import BytesIO
from shutil import make_archive
from shapely.geometry import Polygon
from PIL import ImageDraw

# Setup paths and COCO API
annFile = 'path/for-coco/annotations/instances_train2017.json'  # Update this path
coco = COCO(annFile)

# Define class IDs for car and human (person)
car_id = coco.getCatIds(catNms=['car'])[0]
human_id = coco.getCatIds(catNms=['person'])[0]

# Get image IDs containing cars and humans
filtered_img_ids = coco.getImgIds(catIds=[car_id, human_id])

# Create directories for images and masks
output_images_dir = './resources/images'
output_masks_dir = './resources/masks'
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

# Iterate over the filtered image IDs
for img_id in filtered_img_ids:  # Limiting to 2000 images
    img_info = coco.loadImgs(img_id)[0]
    img_url = img_info['coco_url']
    img_name = img_info['file_name']

    # Download image
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))

    # Save the image
    img.save(os.path.join(output_images_dir, img_name))

    # Get annotations (segmentation masks) for the image
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[car_id, human_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    # Create an empty mask (background = 0)
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

    # Create masks for cars and humans
    for ann in anns:
        if ann['category_id'] == car_id:
            mask_polygon = ann['segmentation']
            for segment in mask_polygon:
                # Convert segmentation to polygon and draw on mask
                polygon = Polygon(zip(segment[::2], segment[1::2]))
                draw = ImageDraw.Draw(Image.fromarray(mask))
                draw.polygon(polygon.exterior.coords, fill=1)  # Car = 1
        elif ann['category_id'] == human_id:
            mask_polygon = ann['segmentation']
            for segment in mask_polygon:
                # Convert segmentation to polygon and draw on mask
                polygon = Polygon(zip(segment[::2], segment[1::2]))
                draw = ImageDraw.Draw(Image.fromarray(mask))
                draw.polygon(polygon.exterior.coords, fill=2)  # Human = 2

    # Save the mask
    mask_img = Image.fromarray(mask)
    mask_img.save(os.path.join(output_masks_dir, img_name.replace('.jpg', '_mask.png')))

print("Download and mask extraction complete!")
