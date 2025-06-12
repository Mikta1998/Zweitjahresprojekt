import os

cam_dir = "EfficientNetB3/cam_visuals"
image_count = len([f for f in os.listdir(cam_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Number of images in cam_visuals: {image_count}")