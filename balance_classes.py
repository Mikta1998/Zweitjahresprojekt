import os
import random
from PIL import Image, ImageOps
from tqdm import tqdm
from pathlib import Path
import shutil

SOURCE_DIR = 'dataset/split/train'
TARGET_DIR = 'dataset/split/train_balanced'
TARGET_SIZE = 500

def augment_image(image):
    # Return a list of augmentations
    return [
        image,
        image.transpose(Image.FLIP_LEFT_RIGHT),
        image.rotate(15),
        image.rotate(-15),
        ImageOps.mirror(image),
    ]

def balance_class(class_name, src_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    images = list(Path(src_path).glob("*.jpg")) + list(Path(src_path).glob("*.png")) + list(Path(src_path).glob("*.jpeg"))

    if len(images) >= TARGET_SIZE:
        # Undersample
        sampled = random.sample(images, TARGET_SIZE)
        for i, img_path in enumerate(sampled):
            img = Image.open(img_path)
            img.save(os.path.join(dst_path, f"{class_name}_{i}.jpg"))
    else:
        # Use original + augment to reach 500
        count = 0
        while count < TARGET_SIZE:
            for img_path in images:
                img = Image.open(img_path)
                for aug in augment_image(img):
                    if count >= TARGET_SIZE:
                        break
                    aug.save(os.path.join(dst_path, f"{class_name}_{count}.jpg"))
                    count += 1

def balance_dataset(source_dir, target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    class_names = os.listdir(source_dir)
    for cls in tqdm(class_names, desc="Balancing classes"):
        src_cls_path = os.path.join(source_dir, cls)
        dst_cls_path = os.path.join(target_dir, cls)
        balance_class(cls, src_cls_path, dst_cls_path)

balance_dataset(SOURCE_DIR, TARGET_DIR)
