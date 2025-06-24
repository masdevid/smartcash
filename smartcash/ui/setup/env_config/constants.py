"""
File: smartcash/ui/setup/env_config/constants.py
Deskripsi: Konstanta untuk environment configuration dengan struktur konsisten
"""

# Required folders untuk setup environment
REQUIRED_FOLDERS = [
    'data', 'configs', 'exports', 'logs', 'models', 'output'
]

# Config discovery settings
CONFIG_SOURCE_PATH = '/content/smartcash/configs'
CONFIG_EXTENSIONS = ['.yaml', '.yml', '.json', '.toml']

# Essential configs minimal yang harus ada (auto-detected)
ESSENTIAL_CONFIG_PATTERNS = [
    'base_config', 'model_config', 'training_config', 
    'dataset_config', 'backbone_config'
]

# Progress step definitions
PROGRESS_STEPS = {
    'start': {'range': (0, 5), 'label': 'Memulai setup...'},
    'analysis': {'range': (5, 15), 'label': '🔍 Menganalisis environment...'},
    'drive_mount': {'range': (15, 30), 'label': '📱 Menghubungkan Google Drive...'},
    'folders': {'range': (30, 50), 'label': '📁 Membuat folder di Drive...'},
    'configs': {'range': (50, 70), 'label': '📋 Menyalin template konfigurasi...'},
    'symlinks': {'range': (70, 85), 'label': '🔗 Membuat symlink...'},
    'validation': {'range': (85, 95), 'label': '✅ Memvalidasi setup...'},
    'complete': {'range': (95, 100), 'label': '🎉 Setup selesai!'}
}

# Drive path constants
DRIVE_MOUNT_POINT = '/content/drive/MyDrive'
SMARTCASH_DRIVE_PATH = '/content/drive/MyDrive/SmartCash'
REPO_CONFIG_PATH = '/content/smartcash/configs'

# Status message templates
STATUS_MESSAGES = {
    'checking': "🔍 Memeriksa status environment...",
    'ready': "✅ Environment sudah terkonfigurasi dengan baik",
    'setup_needed': "🔧 Environment perlu dikonfigurasi",
    'setup_running': "⚙️ Sedang mengkonfigurasi environment...",
    'setup_success': "🎉 Konfigurasi environment berhasil!",
    'setup_failed': "❌ Konfigurasi gagal - Silakan coba lagi",
    'drive_connecting': "📱 Menghubungkan ke Google Drive...",
    'drive_ready': "✅ Google Drive siap dan tervalidasi",
    'drive_error': "❌ Google Drive tidak dapat diakses"
}

# UI element IDs untuk konsistensi
UI_ELEMENTS = {
    'setup_button': 'env_setup_button',
    'status_panel': 'env_status_panel', 
    'progress_bar': 'env_progress_bar',
    'progress_text': 'env_progress_text',
    'log_accordion': 'env_log_accordion',
    'summary_panel': 'env_summary_panel'
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

# Color scheme untuk status
STATUS_COLORS = {
    'success': '#4CAF50',
    'warning': '#FF9800', 
    'error': '#F44336',
    'info': '#2196F3',
    'neutral': '#9E9E9E'
}