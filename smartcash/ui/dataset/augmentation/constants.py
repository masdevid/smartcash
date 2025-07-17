"""
File: smartcash/ui/dataset/augmentation/constants.py
Description: Constants for the augmentation module.
"""

# UI Configuration
UI_CONFIG = {
    'title': 'Data Augmentation',
    'subtitle': 'Configure image augmentation for dataset enhancement',
    'icon': '🎨',
    'module_name': 'augmentation',
    'parent_module': 'dataset',
    'version': '1.0.0'
}

# Button Configuration
BUTTON_CONFIG = {
    'augment': {
        'text': '🎨 Augment Dataset',
        'style': 'primary',
        'tooltip': 'Apply augmentation to the dataset',
        'order': 1
    },
    'preview': {
        'text': '👁️ Preview',
        'style': 'info',
        'tooltip': 'Preview augmentation effects',
        'order': 2
    },
    'check': {
        'text': '🔍 Check Status',
        'style': 'secondary',
        'tooltip': 'Check augmentation status',
        'order': 3
    },
    'cleanup': {
        'text': '🧹 Cleanup',
        'style': 'warning',
        'tooltip': 'Clean up augmented files',
        'order': 4
    }
}

# Validation rules for form fields
VALIDATION_RULES = {
    'augmentation_type': {'required': True},
    'intensity': {'required': True, 'min': 0.0, 'max': 1.0}
}

# Default augmentation pipeline phases
AUGMENTATION_PHASES = {
    'init': {'text': 'Menginisialisasi...', 'style': 'primary'},
    'load': {'text': 'Memuat dataset...', 'style': 'primary'},
    'process': {'text': 'Memproses augmentasi...', 'style': 'primary'},
    'save': {'text': 'Menyimpan hasil...', 'style': 'primary'},
    'verify': {'text': 'Memverifikasi hasil...', 'style': 'primary'},
    'complete': {'text': 'Augmentasi Selesai!', 'style': 'success'},
    'error': {'text': 'Terjadi Kesalahan', 'style': 'danger'}
}