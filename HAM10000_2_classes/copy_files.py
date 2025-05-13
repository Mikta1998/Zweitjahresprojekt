import pandas as pd
import os
import shutil

def copy_photos_nested_by_malignancy(base_image_dir, csv_file, organized_dir):
    metadata = pd.read_csv(csv_file)
    malignant_labels = {'mel', 'bcc', 'akiec'}
    already_copied = True

    for _, row in metadata.iterrows():
        image_id = row['image_id']
        lesion_type = row['dx']
        image_file_name = image_id + '.jpg'

        source_path = os.path.join(base_image_dir, image_file_name)
        if not os.path.exists(source_path):
            continue

        # Determine category
        category = 'malignant' if lesion_type in malignant_labels else 'benign'

        # Destination path: organized/category/lesion_type/
        destination_dir = os.path.join(organized_dir, category, lesion_type)
        os.makedirs(destination_dir, exist_ok=True)

        destination_path = os.path.join(destination_dir, image_file_name)

        if not os.path.exists(destination_path):
            shutil.copy2(source_path, destination_path)
            already_copied = False

    if already_copied:
        print("Photos are already organized.")
    else:
        print("Photos copied successfully.")

# Set your paths
base_image_directory = 'dataset/HAM10000'
csv_file_path = 'dataset/HAM10000_metadata.csv'
organized_directory = 'dataset/organized_nested'

copy_photos_nested_by_malignancy(base_image_directory, csv_file_path, organized_directory)