"""
File: smartcash/common/constants/paths.py
Deskripsi: Path constants yang diperbaiki dengan struktur download dan data yang konsisten
"""

from smartcash.common.constants.core import APP_NAME, DEFAULT_DATA_DIR

# Base paths
BASE_PATH = "."
COLAB_PATH = "/content"  # Base dir di Colab
DEFAULT_CONFIG_DIR = "configs"  # Nama direktori config

# Main data directories
DATA_ROOT = f"{DEFAULT_DATA_DIR}"
DOWNLOADS_DIR = f"{DATA_ROOT}/downloads"  # Temporary download location
PREPROCESSED_DIR = f"{DATA_ROOT}/preprocessed"
AUGMENTED_DIR = f"{DATA_ROOT}/augmented"
BACKUP_DIR = f"{DATA_ROOT}/backup"
INVALID_DIR = f"{DATA_ROOT}/invalid"
VISUALIZATION_DIR = f"{DATA_ROOT}/visualizations"
CHECKPOINT_DIR = "/runs/train/weights"

# Dataset split directories (final locations)
TRAIN_DIR = f"{DATA_ROOT}/train"
VALID_DIR = f"{DATA_ROOT}/valid"
TEST_DIR = f"{DATA_ROOT}/test"

# Colab paths
COLAB_DATASET_PATH = f"{COLAB_PATH}/{DATA_ROOT}"  # Legacy compatibility
COLAB_DATA_ROOT = f"{COLAB_PATH}/{DATA_ROOT}"
COLAB_CONFIG_PATH = f"{COLAB_PATH}/{DEFAULT_CONFIG_DIR}"
COLAB_DOWNLOADS_PATH = f"{COLAB_PATH}/{DOWNLOADS_DIR}"
COLAB_PREPROCESSED_PATH = f"{COLAB_PATH}/{PREPROCESSED_DIR}"
COLAB_AUGMENTED_PATH = f"{COLAB_PATH}/{AUGMENTED_DIR}"
COLAB_BACKUP_PATH = f"{COLAB_PATH}/{BACKUP_DIR}"
COLAB_INVALID_PATH = f"{COLAB_PATH}/{INVALID_DIR}"
COLAB_VISUALIZATION_PATH = f"{COLAB_PATH}/{VISUALIZATION_DIR}"
COLAB_TRAIN_PATH = f"{COLAB_PATH}/{TRAIN_DIR}"
COLAB_VALID_PATH = f"{COLAB_PATH}/{VALID_DIR}"
COLAB_TEST_PATH = f"{COLAB_PATH}/{TEST_DIR}"

# Drive paths
DRIVE_PATH = f"{COLAB_PATH}/drive/MyDrive/{APP_NAME}"
DRIVE_DATA_ROOT = f"{DRIVE_PATH}/{DATA_ROOT}"
DRIVE_CONFIG_PATH = f"{DRIVE_PATH}/{DEFAULT_CONFIG_DIR}"
DRIVE_DOWNLOADS_PATH = f"{DRIVE_PATH}/{DOWNLOADS_DIR}"
DRIVE_PREPROCESSED_PATH = f"{DRIVE_PATH}/{PREPROCESSED_DIR}"
DRIVE_AUGMENTED_PATH = f"{DRIVE_PATH}/{AUGMENTED_DIR}"
DRIVE_BACKUP_PATH = f"{DRIVE_PATH}/{BACKUP_DIR}"
DRIVE_INVALID_PATH = f"{DRIVE_PATH}/{INVALID_DIR}"
DRIVE_VISUALIZATION_PATH = f"{DRIVE_PATH}/{VISUALIZATION_DIR}"
DRIVE_TRAIN_PATH = f"{DRIVE_PATH}/{TRAIN_DIR}"
DRIVE_VALID_PATH = f"{DRIVE_PATH}/{VALID_DIR}"
DRIVE_TEST_PATH = f"{DRIVE_PATH}/{TEST_DIR}"

# Path mapping untuk environment detection
def get_paths_for_environment(is_colab: bool = False, is_drive_mounted: bool = False) -> dict:
    """
    Dapatkan path mapping berdasarkan environment.
    
    Args:
        is_colab: Apakah running di Colab
        is_drive_mounted: Apakah Drive terpasang
        
    Returns:
        Dictionary berisi path mapping
    """
    if is_colab and is_drive_mounted:
        return {
            'data_root': DRIVE_DATA_ROOT,
            'downloads': DRIVE_DOWNLOADS_PATH,
            'train': DRIVE_TRAIN_PATH,
            'valid': DRIVE_VALID_PATH,
            'test': DRIVE_TEST_PATH,
            'backup': DRIVE_BACKUP_PATH,
            'config': DRIVE_CONFIG_PATH
        }
    elif is_colab:
        return {
            'data_root': COLAB_DATA_ROOT,
            'downloads': COLAB_DOWNLOADS_PATH,
            'train': COLAB_TRAIN_PATH,
            'valid': COLAB_VALID_PATH,
            'test': COLAB_TEST_PATH,
            'backup': COLAB_BACKUP_PATH,
            'config': COLAB_CONFIG_PATH
        }
    else:
        return {
            'data_root': DATA_ROOT,
            'downloads': DOWNLOADS_DIR,
            'train': TRAIN_DIR,
            'valid': VALID_DIR,
            'test': TEST_DIR,
            'backup': BACKUP_DIR,
            'config': DEFAULT_CONFIG_DIR
        }