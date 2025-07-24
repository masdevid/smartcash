"""
File: smartcash/ui/dataset/augmentation/constants.py
Description: Constants for the augmentation module.
"""

# UI Configuration
UI_CONFIG = {
    'title': 'Augmentasi Data',
    'subtitle': 'Konfigurasi augmentasi gambar untuk peningkatan dataset',
    'icon': '🎨',
    'module_name': 'augmentation',
    'parent_module': 'dataset',
    'version': '1.0.0'
}

# Button Configuration
BUTTON_CONFIG = {
    'augment': {
        'text': '🎨 Augmentasi Dataset',
        'style': 'success',  # Changed from 'primary' to avoid conflict
        'tooltip': 'Terapkan augmentasi pada dataset',
        'order': 1
    },
    'status': {
        'text': '🔍 Periksa Status',
        'style': 'info',
        'tooltip': 'Periksa status augmentasi',
        'order': 2
    },
    'cleanup': {
        'text': '🧹 Bersihkan',
        'style': 'danger',
        'tooltip': 'Bersihkan file hasil augmentasi',
        'order': 3
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