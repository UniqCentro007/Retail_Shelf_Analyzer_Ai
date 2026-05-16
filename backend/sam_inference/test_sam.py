import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load image
image = cv2.imread("../../sample_images/shelf_image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load SAM model
sam_checkpoint = "models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device="cpu")  # later you can change to "cuda"

# Automatic mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Generate masks
masks = mask_generator.generate(image)

print(f"Total masks generated: {len(masks)}")

# Visualize first 5 masks
plt.figure(figsize=(10, 10))
plt.imshow(image)

for mask in masks[:5]:
    m = mask['segmentation']
    plt.imshow(m, alpha=0.5)

plt.axis("off")
plt.show()
