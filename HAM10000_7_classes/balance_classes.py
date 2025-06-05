import os
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm
from pathlib import Path
import shutil

# Input/output
SOURCE_DIR = 'dataset/split/train'
TARGET_DIR = 'dataset/split/train_balanced'

# Target samples per class
TARGET_SIZES = {
    "vasc": 200,
    "df": 150,
    "mel": 1000,
    "nv": 1500,
    "bcc": 500,
    "akiec": 500,
    "bkl": 1000
}

def random_augment(image):
    w, h = image.size
    img = image.copy()

    # Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random rotation
    angle = random.uniform(-20, 20)
    img = img.rotate(angle)

    # Random zoom (center crop)
    zoom_factor = random.uniform(0.9, 1.1)
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    left = max(0, (new_w - w) // 2)
    top = max(0, (new_h - h) // 2)
    img = resized.crop((left, top, left + w, top + h))

    # Random brightness
    brightness_factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Random contrast
    contrast_factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    return img

def balance_class(class_name, src_path, dst_path, target_size):
    os.makedirs(dst_path, exist_ok=True)
    images = list(Path(src_path).glob("*.jpg")) + list(Path(src_path).glob("*.png")) + list(Path(src_path).glob("*.jpeg"))

    orig_len = len(images)

    if orig_len >= target_size:
        sampled = random.sample(images, target_size)
        for i, img_path in enumerate(sampled):
            img = Image.open(img_path).convert('RGB')
            img.save(os.path.join(dst_path, f"{class_name}_{i}.jpg"))
        return orig_len, target_size

    # Augment
    count = 0
    while count < target_size:
        for img_path in images:
            if count >= target_size:
                break
            img = Image.open(img_path).convert('RGB')
            img.save(os.path.join(dst_path, f"{class_name}_{count}.jpg"))
            count += 1

            while count < target_size:
                aug = random_augment(img)
                aug.save(os.path.join(dst_path, f"{class_name}_{count}.jpg"))
                count += 1
    return orig_len, count

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
        orig, final = balance_class(cls, src_cls_path, dst_cls_path, target_sizes[cls])
        print(f"Class '{cls}': original={orig}, final={final}")

if __name__ == "__main__":
    balance_dataset(SOURCE_DIR, TARGET_DIR, TARGET_SIZES)