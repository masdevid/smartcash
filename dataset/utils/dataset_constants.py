"""
File: smartcash/dataset/utils/dataset_constants.py
Deskripsi: Konstanta yang digunakan dalam operasi dataset
"""

# Konstanta untuk format file gambar
IMG_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

# Konstanta untuk split dataset
DEFAULT_SPLITS = ['train', 'valid', 'test']
DEFAULT_SPLIT_RATIOS = {'train': 0.7, 'valid': 0.15, 'test': 0.15}

# Konstanta untuk random seed
DEFAULT_RANDOM_SEED = 42

# Konstanta untuk prefix file
DEFAULT_ORIGINAL_PREFIX = 'rp'
DEFAULT_AUGMENTATION_PREFIX = 'aug'

# Konstanta untuk path direktori
DEFAULT_DATA_DIR = 'data'
DEFAULT_PREPROCESSED_DIR = 'data/preprocessed'
DEFAULT_AUGMENTED_DIR = 'data/augmented'
DEFAULT_BACKUP_DIR = 'data/backup'
DEFAULT_INVALID_DIR = 'data/invalid'
DEFAULT_VISUALIZATION_DIR = 'visualizations'

# Konstanta untuk ukuran gambar
DEFAULT_IMAGE_SIZE = [640, 640]