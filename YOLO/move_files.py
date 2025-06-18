import pandas as pd
import os
import shutil

def move_photos_by_type(base_image_dir, csv_file, train_dir):

    metadata = pd.read_csv(csv_file)


    for _, row in metadata.iterrows():
        lesion_type = row['dx']
        image_file_name = row['image_id'] + '.jpg'  

        source_path = os.path.join(base_image_dir, image_file_name)
        
        if not os.path.exists(source_path):
            continue

        if lesion_type == 'akiec':
            destination_dir = os.path.join(train_dir, 'akiec')
        elif lesion_type == 'bcc':
            destination_dir = os.path.join(train_dir, 'bcc')
        elif lesion_type == 'bkl':
            destination_dir = os.path.join(train_dir, 'bkl')
        elif lesion_type == 'df':
            destination_dir = os.path.join(train_dir, 'df')
        elif lesion_type == 'mel':
            destination_dir = os.path.join(train_dir, 'mel')
        elif lesion_type == 'nv':
            destination_dir = os.path.join(train_dir, 'nv')
        elif lesion_type == 'vasc':
            destination_dir = os.path.join(train_dir, 'vasc')
        else:
            continue  


        os.makedirs(destination_dir, exist_ok=True)

        destination_path = os.path.join(destination_dir, image_file_name)
        shutil.move(source_path, destination_path)

base_image_directory = r'F:\KI in den Life Sciences\HAM10000_images'
csv_file_path = r'F:\KI in den Life Sciences\HAM10000_metadata.csv'
train_directory = r'F:\KI in den Life Sciences\dataset4\train'

move_photos_by_type(base_image_directory, csv_file_path, train_directory)