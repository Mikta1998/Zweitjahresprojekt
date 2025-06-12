import os
import torch

cam_dir = "EfficientNetB3/cam_visuals"
image_count = len([f for f in os.listdir(cam_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Number of images in cam_visuals: {image_count}")


checkpoint = torch.load("EfficientNetB3/train6/best_model_loss.pt", map_location="cpu")
print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("Checkpoint keys:", checkpoint.keys())