import os
import shutil
import random
from PIL import Image, ImageOps, ImageEnhance

def augment_image(image):
    """Apply random augmentations to an image."""
    augmentations = [
        ImageOps.mirror,
        ImageOps.flip,
        lambda img: img.rotate(random.uniform(-15, 15)),
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2)),
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    ]
    augmentation = random.choice(augmentations)
    return augmentation(image)

def augment_photos_in_directory(train_dir, target_count=1500):

    for lesion_type in os.listdir(train_dir):
        lesion_dir = os.path.join(train_dir, lesion_type)
        
        if not os.path.isdir(lesion_dir):
            continue

        image_files = [f for f in os.listdir(lesion_dir) if f.endswith('.jpg')]
        num_images = len(image_files)

        if num_images < target_count:
            print(f'Augmenting images in {lesion_dir}. Current count: {num_images}')
            while num_images < target_count:
                image_file = random.choice(image_files)
                image_path = os.path.join(lesion_dir, image_file)
                
                with Image.open(image_path) as img:
                    augmented_img = augment_image(img)
                    new_image_name = f"{os.path.splitext(image_file)[0]}_aug_{num_images}.jpg"
                    new_image_path = os.path.join(lesion_dir, new_image_name)
                    
                    augmented_img.save(new_image_path)
                
                num_images += 1
            print(f'Completed augmentation for {lesion_dir}. New count: {num_images}')


train_directory = r'F:\KI in den Life Sciences\dataset3\train'


augment_photos_in_directory(train_directory)