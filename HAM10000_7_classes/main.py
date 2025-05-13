from copy_files import copy_photos_by_type
from split_data import split_dataset
from balance_classes import balance_dataset


print("\nCopying files into organized folder...")
copy_photos_by_type('dataset/HAM10000', 'dataset/HAM10000_metadata.csv', 'dataset/organized')

print("\nSplitting dataset into train, val, and test...")
split_dataset('dataset/organized', 'dataset/split')

"""print("\nBalancing training data via augmentation...")
balance_dataset('dataset/split/train', 'dataset/split/train_balanced')"""

