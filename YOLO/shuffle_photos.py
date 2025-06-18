import os
import random
import time

def shuffle_images_in_folder(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Temporarily rename files to randomize their order
    temp_filenames = []
    for filename in files:
        old_path = os.path.join(folder_path, filename)
        temp_filename = f"{random.randint(100000, 999999)}_{filename}"
        temp_path = os.path.join(folder_path, temp_filename)
        os.rename(old_path, temp_path)
        temp_filenames.append(temp_filename)
    
    # Shuffle the temporary filenames
    random.shuffle(temp_filenames)
    
    # Rename files back to their original names in the new order
    for i, temp_filename in enumerate(temp_filenames):
        old_path = os.path.join(folder_path, temp_filename)
        new_filename = f"{i:04d}_{temp_filename.split('_', 1)[1]}"
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)

if __name__ == "__main__":
    
    train_directory = r'F:\KI in den Life Sciences\dataset3\val'
    
   
    subfolders = ['gutartig', 'boesartig']
    
    # Shuffle images in each subfolder
    for subfolder in subfolders:
        folder_path = os.path.join(train_directory, subfolder)
        shuffle_images_in_folder(folder_path)
        print(f"Shuffling and renaming completed for {subfolder}.")