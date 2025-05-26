import os

def count_images_per_class(split_root):
    for split_name in ['train_balanced', 'val', 'test']:
        split_path = os.path.join(split_root, split_name)
        print(f"\n {split_name.upper()}:")

        total_images = 0
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))])
            print(f"  {class_name}: {count}")
            total_images += count

        print(f"Total in {split_name}: {total_images}")


split_folder = 'dataset/split'
count_images_per_class(split_folder)