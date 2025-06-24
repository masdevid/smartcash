"""
File: smartcash/ui/setup/env_config/constants.py
Deskripsi: Konstanta untuk environment configuration
"""

# Required folders for SmartCash
REQUIRED_FOLDERS = [
    '/content/smartcash',
    '/content/smartcash/data',
    '/content/smartcash/data/download',
    '/content/smartcash/data/backup',
    '/content/smartcash/data/train',
    '/content/smartcash/data/test',
    '/content/smartcash/data/valid',
    '/content/smartcash/data/invalid',
    '/content/smartcash/data/preprocessed',
    '/content/smartcash/data/augmented',
    '/content/smartcash/models',
    '/content/smartcash/configs',
    '/content/smartcash/outputs',
    '/content/smartcash/logs'
]

# Source directories that need to be created in Google Drive
SOURCE_DIRECTORIES = [
    '/content/drive/MyDrive/SmartCash/data',
    '/content/drive/MyDrive/SmartCash/models',
    '/content/drive/MyDrive/SmartCash/configs',
    '/content/drive/MyDrive/SmartCash/outputs',
    '/content/drive/MyDrive/SmartCash/logs'
]

# Symlink mapping: source -> target
SYMLINK_MAP = {
    '/content/drive/MyDrive/SmartCash/data': '/content/data',
    '/content/drive/MyDrive/SmartCash/models': '/content/models',
    '/content/drive/MyDrive/SmartCash/configs': '/content/configs',
    '/content/drive/MyDrive/SmartCash/outputs': '/content/outputs',
    '/content/drive/MyDrive/SmartCash/logs': '/content/logs'
}

# Progress steps
PROGRESS_STEPS = [
    "Mounting Google Drive",
    "Creating directories", 
    "Syncing configurations",
    "Verifying setup"
]

# Status messages
STATUS_MESSAGES = {
    'ready': "Siap untuk setup environment",
    'running': "Setup sedang berjalan...",
    'success': "Setup berhasil diselesaikan!",
    'failed': "Setup gagal, silakan coba lagi"
}