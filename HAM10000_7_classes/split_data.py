import os
import shutil
import random
from glob import glob

def split_dataset(source_dir, output_dir, split_ratio=(0.7, 0.2, 0.1), seed=42):
    random.seed(seed)

    # Loop over each class folder
    class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in class_names:
        class_dir = os.path.join(source_dir, class_name)
        images = glob(os.path.join(class_dir, "*.jpg"))

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

        # Copies files to new split directories if they don't already exist
        for split_name, image_list in subsets.items():
            split_class_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for img_path in image_list:
                dest_path = os.path.join(split_class_dir, os.path.basename(img_path))
                if not os.path.exists(dest_path):
                    shutil.copy2(img_path, dest_path)

    print("Dataset split complete (existing files skipped).")


source_dir = 'dataset/organized'  
output_dir = 'dataset/split'       

split_dataset(source_dir, output_dir)