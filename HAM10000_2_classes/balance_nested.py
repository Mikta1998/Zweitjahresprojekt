import os
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm
from pathlib import Path
import shutil

SOURCE_DIR = 'dataset/split_nested/train'
TARGET_DIR = 'dataset/split_nested/train_balanced'
TARGET_SIZE = 1000

def augment_image(image):
    augmented = [image]

    augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    augmented.append(image.rotate(15))
    augmented.append(image.rotate(-15))

    zoom_factors = [0.9, 1.1]
    for factor in zoom_factors:
        w, h = image.size
        new_w, new_h = int(w * factor), int(h * factor)
        resized = image.resize((new_w, new_h), Image.BICUBIC)
        cropped = resized.crop((0, 0, w, h))
        augmented.append(cropped)

    brightness = ImageEnhance.Brightness(image).enhance(1.2)
    dimmed = ImageEnhance.Brightness(image).enhance(0.8)
    augmented.extend([brightness, dimmed])

    contrast_up = ImageEnhance.Contrast(image).enhance(1.3)
    contrast_down = ImageEnhance.Contrast(image).enhance(0.7)
    augmented.extend([contrast_up, contrast_down])

    return augmented

def balance_class(category, class_name, src_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    images = list(Path(src_path).glob("*.jpg")) + list(Path(src_path).glob("*.png")) + list(Path(src_path).glob("*.jpeg"))

    if len(images) >= TARGET_SIZE:
        sampled = random.sample(images, TARGET_SIZE)
        for i, img_path in enumerate(sampled):
            img = Image.open(img_path).convert('RGB')
            img.save(os.path.join(dst_path, f"{category}_{class_name}_{i}.jpg"))
    else:
        count = 0
        while count < TARGET_SIZE:
            for img_path in images:
                img = Image.open(img_path).convert('RGB')
                for aug in augment_image(img):
                    if count >= TARGET_SIZE:
                        break
                    aug.save(os.path.join(dst_path, f"{category}_{class_name}_{count}.jpg"))
                    count += 1

def balance_dataset_nested(source_dir, target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    categories = [cat for cat in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, cat))]

    for category in tqdm(categories, desc="Processing categories"):
        category_path = os.path.join(source_dir, category)
        class_names = [cls for cls in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, cls))]

        for class_name in tqdm(class_names, desc=f"Balancing {category}", leave=False):
            src_cls_path = os.path.join(category_path, class_name)
            dst_cls_path = os.path.join(target_dir, category, class_name)
            balance_class(category, class_name, src_cls_path, dst_cls_path)

if __name__ == "__main__":
    balance_dataset_nested(SOURCE_DIR, TARGET_DIR)