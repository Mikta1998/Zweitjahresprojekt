import os
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm
from pathlib import Path
import shutil

# Input/output
SOURCE_DIR = 'dataset/split/train'
TARGET_DIR = 'dataset/split/train_balanced'

# Custom target sizes per class
TARGET_SIZES = {
    "vasc": 1000,
    "df": 1000,
    "mel": 1000,
    "nv": 1000,
    "bcc": 1000,
    "akiec": 1000,
    "bkl": 1000
}

def augment_image(image):
    augmented = []

    # Original
    augmented.append(image)

    # Horizontal flip
    augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))

    # Rotations (±15°)
    augmented.append(image.rotate(15))
    augmented.append(image.rotate(-15))

    # Slight zooms
    zoom_factors = [0.9, 1.1]
    for factor in zoom_factors:
        w, h = image.size
        new_w, new_h = int(w * factor), int(h * factor)
        resized = image.resize((new_w, new_h), Image.BICUBIC)
        cropped = resized.crop((0, 0, w, h))
        augmented.append(cropped)

    # Brightness adjustment
    brightness = ImageEnhance.Brightness(image).enhance(1.2)
    dimmed = ImageEnhance.Brightness(image).enhance(0.8)
    augmented.extend([brightness, dimmed])

    # Contrast adjustment
    contrast_up = ImageEnhance.Contrast(image).enhance(1.3)
    contrast_down = ImageEnhance.Contrast(image).enhance(0.7)
    augmented.extend([contrast_up, contrast_down])

    return augmented

def balance_class(class_name, src_path, dst_path, target_size):
    os.makedirs(dst_path, exist_ok=True)
    images = list(Path(src_path).glob("*.jpg")) + list(Path(src_path).glob("*.png")) + list(Path(src_path).glob("*.jpeg"))

    if len(images) >= target_size:
        # Undersample
        sampled = random.sample(images, target_size)
        for i, img_path in enumerate(sampled):
            img = Image.open(img_path).convert('RGB')
            img.save(os.path.join(dst_path, f"{class_name}_{i}.jpg"))
    else:
        # Use original + augment to reach target_size
        count = 0
        while count < target_size:
            for img_path in images:
                img = Image.open(img_path).convert('RGB')
                for aug in augment_image(img):
                    if count >= target_size:
                        break
                    aug.save(os.path.join(dst_path, f"{class_name}_{count}.jpg"))
                    count += 1
                if count >= target_size:
                    break

def balance_dataset(source_dir, target_dir, target_sizes):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    class_names = os.listdir(source_dir)
    for cls in tqdm(class_names, desc="Balancing classes"):
        if cls not in target_sizes:
            print(f"Skipping class '{cls}' (no target size defined)")
            continue
        src_cls_path = os.path.join(source_dir, cls)
        dst_cls_path = os.path.join(target_dir, cls)
        balance_class(cls, src_cls_path, dst_cls_path, target_sizes[cls])

if __name__ == "__main__":
    balance_dataset(SOURCE_DIR, TARGET_DIR, TARGET_SIZES)