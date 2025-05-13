import os
import shutil
import random
from glob import glob

def split_nested_dataset(source_dir, output_dir, split_ratio=(0.7, 0.2, 0.1), seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    categories = [cat for cat in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, cat))]

    for category in categories:
        category_path = os.path.join(source_dir, category)
        class_names = [cls for cls in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, cls))]

        for class_name in class_names:
            class_dir = os.path.join(category_path, class_name)

            # Match both .jpg and .jpeg, any case
            images = glob(os.path.join(class_dir, "*.[jJ][pP][gG]")) + \
                     glob(os.path.join(class_dir, "*.[jJ][pP][eE][gG]"))

            print(f"Found {len(images)} images in {category}/{class_name}")

            if not images:
                continue

            # Shuffle and split
            random.shuffle(images)
            total = len(images)
            train_end = int(split_ratio[0] * total)
            val_end = train_end + int(split_ratio[1] * total)

            subsets = {
                'train': images[:train_end],
                'val': images[train_end:val_end],
                'test': images[val_end:]
            }

            for split_name, image_list in subsets.items():
                split_class_dir = os.path.join(output_dir, split_name, category, class_name)
                os.makedirs(split_class_dir, exist_ok=True)

                for img_path in image_list:
                    dest_path = os.path.join(split_class_dir, os.path.basename(img_path))
                    shutil.copy2(img_path, dest_path)

    print("\nNested dataset split complete!")

# Run it
source_dir = 'dataset/organized_nested'
output_dir = 'dataset/split_nested'
split_nested_dataset(source_dir, output_dir)