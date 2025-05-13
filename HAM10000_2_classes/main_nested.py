from copy_files import copy_photos_nested_by_malignancy
from split_data import split_nested_dataset
from balance_nested import balance_dataset_nested

print("\nCopying files into organized_nested folder...")
copy_photos_nested_by_malignancy(
    base_image_dir='dataset/HAM10000',
    csv_file='dataset/HAM10000_metadata.csv',
    organized_dir='dataset/organized_nested'
)

print("\nSplitting dataset into train, val, and test (nested)...")
split_nested_dataset(
    source_dir='dataset/organized_nested',
    output_dir='dataset/split_nested'
)

print("\nBalancing training data via augmentation (nested)...")
balance_dataset_nested(
    source_dir='dataset/split_nested/train',
    target_dir='dataset/split_nested/train_balanced'
)