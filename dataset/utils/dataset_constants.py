"""
File: smartcash/dataset/utils/dataset_constants.py
Deskripsi: Konstanta yang digunakan dalam operasi dataset
"""
from smartcash.common.constants import APP_NAME
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
DEFAULT_PREPROCESSED_DIR = f'{DEFAULT_DATA_DIR}/preprocessed'
DEFAULT_AUGMENTED_DIR = f'{DEFAULT_DATA_DIR}/augmented'
DEFAULT_BACKUP_DIR = f'{DEFAULT_DATA_DIR}/backup'
DEFAULT_INVALID_DIR = f'{DEFAULT_DATA_DIR}/invalid'
DEFAULT_VISUALIZATION_DIR = f'{DEFAULT_DATA_DIR}/visualizations'
DEFAULT_CHECKPOINT_DIR = f'/runs/train/weights'
DEFAULT_TRAIN_DIR = f'{DEFAULT_DATA_DIR}/train'
DEFAULT_VALID_DIR = f'{DEFAULT_DATA_DIR}/valid'
DEFAULT_TEST_DIR = f'{DEFAULT_DATA_DIR}/test'

COLAB_PATH = '/content'
COLAB_DATASET_PATH = f'{COLAB_PATH}/{DEFAULT_DATA_DIR}'
COLAB_PREPROCESSED_PATH = f'{COLAB_PATH}/{DEFAULT_PREPROCESSED_DIR}'
COLAB_AUGMENTED_PATH = f'{COLAB_PATH}/{DEFAULT_AUGMENTED_DIR}'
COLAB_BACKUP_PATH = f'{COLAB_PATH}/{DEFAULT_BACKUP_DIR}'
COLAB_INVALID_PATH = f'{COLAB_PATH}/{DEFAULT_INVALID_DIR}'
COLAB_VISUALIZATION_PATH = f'{COLAB_PATH}/{DEFAULT_VISUALIZATION_DIR}'


DRIVE_PATH = f'{COLAB_PATH}/drive/MyDrive/{APP_NAME}'
DRIVE_DATASET_PATH = f'{DRIVE_PATH}/{DEFAULT_DATA_DIR}'
DRIVE_PREPROCESSED_PATH = f'{DRIVE_PATH}/{DEFAULT_PREPROCESSED_DIR}'
DRIVE_AUGMENTED_PATH = f'{DRIVE_PATH}/{DEFAULT_AUGMENTED_DIR}'
DRIVE_BACKUP_PATH = f'{DRIVE_PATH}/{DEFAULT_BACKUP_DIR}'
DRIVE_INVALID_PATH = f'{DRIVE_PATH}/{DEFAULT_INVALID_DIR}'
DRIVE_VISUALIZATION_PATH = f'{DRIVE_PATH}/{DEFAULT_VISUALIZATION_DIR}'

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