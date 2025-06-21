"""
File: smartcash/common/constants/dataset.py
Deskripsi: Konstanta terkait dataset, pengolahan data, dan pengaturan split
"""

# Split dataset
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_TEST_SPLIT = 0.15

# Preprocessing
DEFAULT_IMAGE_SIZE = (640, 640)
DEFAULT_BATCH_SIZE = 16

# Augmentation 
DEFAULT_AUGMENTATION_MULTIPLIER = 2
DEFAULT_AUGMENTATIONS = [
    'horizontal_flip',
    'rotate',
    'brightness_contrast',
    'hue_saturation'
]

# Data versioning
DATA_VERSION_FORMAT = "v{major}.{minor}"
DEFAULT_DATA_VERSION = "v1.0"

# Path template
PATH_TEMPLATES = {
    'images': '{split}/images',
    'labels': '{split}/labels',
    'visualizations': 'visualizations/{split}',
    'preprocessed': 'preprocessed/{split}',
    'augmented': 'augmented/{split}'
}