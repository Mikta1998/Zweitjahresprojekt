import pandas as pd
import os
import shutil

"""




DO NOT RUN THIS CODE!!!





"""

def copy_photos_by_type(base_image_dir, csv_file, organized_dir):
    metadata = pd.read_csv(csv_file)

    for _, row in metadata.iterrows():
        lesion_type = row['dx']
        image_file_name = row['image_id'] + '.jpg'  

        source_path = os.path.join(base_image_dir, image_file_name)

        if not os.path.exists(source_path):
            continue

        # Manually check each class (your original structure)
        if lesion_type == 'akiec':
            destination_dir = os.path.join(organized_dir, 'akiec')
        elif lesion_type == 'bcc':
            destination_dir = os.path.join(organized_dir, 'bcc')
        elif lesion_type == 'bkl':
            destination_dir = os.path.join(organized_dir, 'bkl')
        elif lesion_type == 'df':
            destination_dir = os.path.join(organized_dir, 'df')
        elif lesion_type == 'mel':
            destination_dir = os.path.join(organized_dir, 'mel')
        elif lesion_type == 'nv':
            destination_dir = os.path.join(organized_dir, 'nv')
        elif lesion_type == 'vasc':
            destination_dir = os.path.join(organized_dir, 'vasc')
        else:
            continue 

        # Creates the class folder if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)

        # Copies photos
        destination_path = os.path.join(destination_dir, image_file_name)
        shutil.copy2(source_path, destination_path)

# My paths
base_image_directory = 'dataset/HAM10000'
csv_file_path = 'dataset/HAM10000_metadata.csv'
organized_directory = 'dataset/organized'

copy_photos_by_type(base_image_directory, csv_file_path, organized_directory)