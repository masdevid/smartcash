"""
File: smartcash/dataset/utils/dataset_constants.py
Deskripsi: Konstanta yang digunakan dalam operasi dataset
"""
from smartcash.common.constants.core import APP_NAME
from smartcash.common.constants.paths import (
    DEFAULT_DATA_DIR, PREPROCESSED_DIR as DEFAULT_PREPROCESSED_DIR,
    AUGMENTED_DIR as DEFAULT_AUGMENTED_DIR, BACKUP_DIR as DEFAULT_BACKUP_DIR,
    INVALID_DIR as DEFAULT_INVALID_DIR, VISUALIZATION_DIR as DEFAULT_VISUALIZATION_DIR,
    CHECKPOINT_DIR as DEFAULT_CHECKPOINT_DIR, TRAIN_DIR as DEFAULT_TRAIN_DIR,
    VALID_DIR as DEFAULT_VALID_DIR, TEST_DIR as DEFAULT_TEST_DIR,
    COLAB_PATH, COLAB_DATASET_PATH, COLAB_PREPROCESSED_PATH, COLAB_AUGMENTED_PATH,
    COLAB_BACKUP_PATH, COLAB_INVALID_PATH, COLAB_VISUALIZATION_PATH,
    DRIVE_PATH, DRIVE_DATASET_PATH, DRIVE_PREPROCESSED_PATH, DRIVE_AUGMENTED_PATH,
    DRIVE_BACKUP_PATH, DRIVE_INVALID_PATH, DRIVE_VISUALIZATION_PATH
)
from smartcash.common.constants.dataset import DEFAULT_TRAIN_SPLIT, DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT

# Konstanta untuk format file gambar
IMG_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

# Konstanta untuk split dataset
DEFAULT_SPLITS = ['train', 'valid', 'test']
DEFAULT_SPLIT_RATIOS = {'train': DEFAULT_TRAIN_SPLIT, 'valid': DEFAULT_VAL_SPLIT, 'test': DEFAULT_TEST_SPLIT}

# Konstanta untuk random seed
DEFAULT_RANDOM_SEED = 42

# Konstanta untuk prefix file
DEFAULT_ORIGINAL_PREFIX = 'rp'
DEFAULT_AUGMENTATION_PREFIX = 'aug'

# Konstanta untuk ukuran gambar
DEFAULT_IMG_SIZE = [640, 640]
DENOMINATION_CLASS_MAP = {
    '0': '1k',
    '1': '2k',
    '2': '5k',
    '3': '10k',
    '4': '20k',
    '5': '50k',
    '6': '100k',
    '7': '1k',
    '8': '2k',
    '9': '5k',
    '10': '10k',
    '11': '20k',
    '12': '50k',
    '13': '100k',
    '14': 'all',
    '15': 'all',
    '16': 'all'
}