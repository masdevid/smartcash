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

DRIVE_PATH = '/content/drive/MyDrive/SmartCash'
DRIVE_DATASET_PATH = f'{DRIVE_PATH}/{DEFAULT_DATA_DIR}'
DRIVE_PREPROCESSED_PATH = f'{DRIVE_PATH}/{DEFAULT_PREPROCESSED_DIR}'
DRIVE_AUGMENTED_PATH = f'{DRIVE_PATH}/{DEFAULT_AUGMENTED_DIR}'
DRIVE_BACKUP_PATH = f'{DRIVE_PATH}/{DEFAULT_BACKUP_DIR}'
DRIVE_INVALID_PATH = f'{DRIVE_PATH}/{DEFAULT_INVALID_DIR}'
DRIVE_VISUALIZATION_PATH = f'{DRIVE_PATH}/{DEFAULT_VISUALIZATION_DIR}'

COLAB_PATH = '/content'
COLAB_DATASET_PATH = f'{COLAB_PATH}/{DEFAULT_DATA_DIR}'
COLAB_PREPROCESSED_PATH = f'{COLAB_PATH}/{DEFAULT_PREPROCESSED_DIR}'
COLAB_AUGMENTED_PATH = f'{COLAB_PATH}/{DEFAULT_AUGMENTED_DIR}'
COLAB_BACKUP_PATH = f'{COLAB_PATH}/{DEFAULT_BACKUP_DIR}'
COLAB_INVALID_PATH = f'{COLAB_PATH}/{DEFAULT_INVALID_DIR}'
COLAB_VISUALIZATION_PATH = f'{COLAB_PATH}/{DEFAULT_VISUALIZATION_DIR}'
# Konstanta untuk ukuran gambar
DEFAULT_IMAGE_SIZE = [640, 640]