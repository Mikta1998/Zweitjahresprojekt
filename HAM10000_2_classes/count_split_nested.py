import os

def count_images_per_class_nested(split_root):
    for split_name in ['train_balanced', 'val', 'test']:
        split_path = os.path.join(split_root, split_name)
        print(f"\n{split_name.upper()}:")

        total_images = 0
        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            if not os.path.isdir(category_path):
                continue

            for class_name in os.listdir(category_path):
                class_path = os.path.join(category_path, class_name)
                if not os.path.isdir(class_path):
                    continue

                count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                print(f"  {category}/{class_name}: {count}")
                total_images += count

        print(f"Total in {split_name}: {total_images}")

# Set the path
split_folder = 'dataset/split_nested'
count_images_per_class_nested(split_folder)