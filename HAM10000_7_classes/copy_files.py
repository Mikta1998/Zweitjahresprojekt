import pandas as pd
import os
import shutil

def copy_photos_by_type(base_image_dir, csv_file, organized_dir):
    metadata = pd.read_csv(csv_file)
    already_copied = True  # Assume all files are already there

    for _, row in metadata.iterrows():
        lesion_type = row['dx']
        image_file_name = row['image_id'] + '.jpg'  

        source_path = os.path.join(base_image_dir, image_file_name)

        if not os.path.exists(source_path):
            continue

        destination_dir = os.path.join(organized_dir, lesion_type)
        os.makedirs(destination_dir, exist_ok=True)

        destination_path = os.path.join(destination_dir, image_file_name)

        if not os.path.exists(destination_path):
            shutil.copy2(source_path, destination_path)
            already_copied = False  # At least one file was copied

    if already_copied:
        print("Photos are already in organized.")
    else:
        print("Photos copied successfully.")

# My paths
base_image_directory = 'dataset/HAM10000'
csv_file_path = 'dataset/HAM10000_metadata.csv'
organized_directory = 'dataset/organized'

copy_photos_by_type(base_image_directory, csv_file_path, organized_directory)