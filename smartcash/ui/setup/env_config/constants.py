"""
File: smartcash/ui/setup/env_config/constants.py
Deskripsi: Constants untuk environment config dengan struktur yang konsisten
"""

# Required folders untuk setup environment
REQUIRED_FOLDERS = ['data', 'configs', 'exports', 'logs', 'models', 'output']

# Config templates yang akan di-clone ke Drive
CONFIG_TEMPLATES = [
    'base_config.yaml', 'colab_config.yaml', 'dataset_config.yaml',
    'model_config.yaml', 'training_config.yaml', 'augmentation_config.yaml',
    'preprocessing_config.yaml', 'hyperparameters_config.yaml', 
    'backbone_config.yaml', 'split_config.yaml', 'evaluation_config.yaml',
    'pretrained_config.yaml','analysis_config.yaml', 'strategy_config.yaml'
]

# Essential configs yang minimal harus ada
ESSENTIAL_CONFIGS = ['base_config.yaml', 'model_config.yaml', 'training_config.yaml', 'dataset_config.yaml', 'backbone_config.yaml']

# Progress ranges untuk setiap operation
PROGRESS_RANGES = {
    'start': (1, 5),
    'analysis': (5, 10),
    'drive_mount': (10, 25),
    'folders': (30, 45),
    'configs': (50, 65),
    'symlinks': (70, 90),
    'validation': (95, 100)
}

# Drive paths
DRIVE_MOUNT_POINT = '/content/drive/MyDrive'
SMARTCASH_DRIVE_PATH = '/content/drive/MyDrive/SmartCash'
REPO_CONFIG_PATH = '/content/smartcash/configs'

# Status messages
STATUS_MESSAGES = {
    'ready': "✅ Environment sudah terkonfigurasi",
    'setup_needed': "🔧 Environment perlu dikonfigurasi", 
    'setup_start': "🚀 Memulai konfigurasi environment SmartCash...",
    'setup_success': "🎉 Setup environment berhasil selesai!",
    'setup_failed': "❌ Setup gagal - Coba lagi",
    'drive_ready': "✅ Drive siap digunakan dan telah divalidasi",
    'drive_error': "❌ Drive tidak dapat diakses"
}

# Progress messages templates
PROGRESS_MESSAGES = {
    'start': "Memulai setup...",
    'refresh': "🔄 Refreshing environment state...",
    'analysis': "🔍 Analyzing environment...", 
    'drive_connect': "📱 Menghubungkan Google Drive...",
    'drive_mount': "📱 Connecting to Google Drive...",
    'folders_create': "📁 Creating Drive folders...",
    'configs_clone': "📋 Cloning config templates...",
    'symlinks_create': "🔗 Creating symlinks...",
    'validation': "✅ Validating setup...",
    'complete': "✅ Setup completed successfully"
}

# Retry configurations
RETRY_CONFIG = {
    'drive_ready_attempts': 3,
    'drive_ready_delay': 0.5,
    'symlink_attempts': 3,
    'symlink_delay': 0.5,
    'status_check_retries': 3,
    'drive_mount_timeout': 45
}