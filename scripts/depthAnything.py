from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# visualize the prediction
#output = prediction.squeeze().cpu().numpy()
#formatted = (output * 255 / np.max(output)).astype("uint8")
#depth = Image.fromarray(formatted)

#######################

# Visualize the input image and depth image side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Display the input image
axes[0].imshow(image)
axes[0].set_title("Input Image")
axes[0].axis("off")

# Display the depth image
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth_image = Image.fromarray(formatted)
axes[1].imshow(depth_image, cmap='gray')
axes[1].set_title("Depth Image")
axes[1].axis("off")

plt.show()