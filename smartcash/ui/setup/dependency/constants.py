"""
File: smartcash/ui/setup/dependency/constants.py
Deskripsi: Constants untuk dependency module
"""

from typing import Dict, Any

# Package status constants
PACKAGE_STATUS = {
    'INSTALLED': 'installed',
    'NOT_INSTALLED': 'not_installed',
    'CHECKING': 'checking',
    'INSTALLING': 'installing',
    'UPDATING': 'updating',
    'UNINSTALLING': 'uninstalling',
    'UPDATE_AVAILABLE': 'update_available',
    'ERROR': 'error'
}

# Button action constants
BUTTON_ACTIONS = {
    'INSTALL': 'install',
    'UPDATE': 'update',
    'UNINSTALL': 'uninstall',
    'CHECK': 'check_status'
}

# UI settings constants
UI_SETTINGS = {
    'DEFAULT_TAB': 0,
    'CATEGORIES_TAB': 0,
    'CUSTOM_TAB': 1,
    'MIN_HEIGHT': '600px',
    'GRID_COLUMNS': 'repeat(auto-fit, minmax(400px, 1fr))',
    'GRID_GAP': '20px'
}

# Installation constants
INSTALL_DEFAULTS = {
    'TIMEOUT': 300,
    'RETRIES': 3,
    'PARALLEL_WORKERS': 4,
    'PYTHON_PATH': 'python',
    'PACKAGE_MANAGER': 'pip',
    'UPGRADE_STRATEGY': 'eager'
}

# UI Configuration
UI_CONFIG = {
    'title': "Pengelola Dependensi",
    'subtitle': "Instal, perbarui, dan kelola paket Python",
    'icon': "📦",
    'module_name': "dependency",
    'parent_module': "setup",
    'version': "1.0.0"
}

# Button configuration - Using multiple action buttons pattern
BUTTON_CONFIG = {
    'install': {
        'text': 'Instal',
        'style': 'success',
        'icon': 'download',
        'tooltip': 'Instal paket yang dipilih',
        'order': 1
    },
    'check_updates': {
        'text': 'Cek & Perbarui',
        'style': 'info',
        'icon': 'refresh',
        'tooltip': 'Periksa status paket dan pembaruan yang tersedia',
        'order': 2
    },
    'uninstall': {
        'text': 'Uninstal',
        'style': 'danger',
        'icon': 'trash',
        'tooltip': 'Uninstal paket yang dipilih',
        'order': 3
    }
}

# Color scheme
COLORS = {
    'SUCCESS': '#4CAF50',
    'WARNING': '#FF9800',
    'ERROR': '#F44336',
    'INFO': '#2196F3',
    'SECONDARY': '#9C27B0',
    'BORDER': '#ddd',
    'BACKGROUND': '#f8f9fa',
    'WHITE': '#ffffff'
}

# Icons
ICONS = {
    'PACKAGE': '📦',
    'INSTALL': '📥',
    'UPDATE': '⬆️',
    'UNINSTALL': '🗑️',
    'CHECK': '🔍',
    'SUCCESS': '✅',
    'ERROR': '❌',
    'WARNING': '⚠️',
    'INFO': 'ℹ️',
    'LOADING': '🔄',
    'STAR': '⭐',
    'CUSTOM': '🛠️',
    'CATEGORIES': '📦'
}

# Error messages
ERROR_MESSAGES = {
    'NO_PACKAGES_SELECTED': 'Tidak ada packages yang dipilih',
    'INSTALL_FAILED': 'Gagal menginstall package',
    'UPDATE_FAILED': 'Gagal mengupdate package',
    'UNINSTALL_FAILED': 'Gagal menguninstall package',
    'CHECK_FAILED': 'Gagal mengecek status package',
    'CONFIG_SAVE_FAILED': 'Gagal menyimpan konfigurasi',
    'CONFIG_LOAD_FAILED': 'Gagal memuat konfigurasi',
    'INVALID_PACKAGE_FORMAT': 'Format package tidak valid'
}

# Success messages
SUCCESS_MESSAGES = {
    'INSTALL_SUCCESS': 'Package berhasil diinstall',
    'UPDATE_SUCCESS': 'Package berhasil diupdate',
    'UNINSTALL_SUCCESS': 'Package berhasil diuninstall',
    'CHECK_SUCCESS': 'Status package berhasil dicek',
    'CONFIG_SAVE_SUCCESS': 'Konfigurasi berhasil disimpan',
    'CONFIG_LOAD_SUCCESS': 'Konfigurasi berhasil dimuat'
}

# Log messages dengan emoji
LOG_MESSAGES = {
    'INIT_START': '🚀 Memulai inisialisasi dependency module...',
    'INIT_SUCCESS': '✅ Dependency module berhasil diinisialisasi',
    'INIT_FAILED': '❌ Gagal menginisialisasi dependency module',
    'INSTALL_START': '📥 Memulai instalasi packages...',
    'INSTALL_SUCCESS': '✅ Instalasi packages berhasil',
    'INSTALL_FAILED': '❌ Instalasi packages gagal',
    'UPDATE_START': '⬆️ Memulai update packages...',
    'UPDATE_SUCCESS': '✅ Update packages berhasil',
    'UPDATE_FAILED': '❌ Update packages gagal',
    'UNINSTALL_START': '🗑️ Memulai uninstall packages...',
    'UNINSTALL_SUCCESS': '✅ Uninstall packages berhasil',
    'UNINSTALL_FAILED': '❌ Uninstall packages gagal',
    'CHECK_START': '🔍 Memulai pengecekan status packages...',
    'CHECK_SUCCESS': '✅ Pengecekan status packages berhasil',
    'CHECK_FAILED': '❌ Pengecekan status packages gagal',
    'CONFIG_SYNC': '🔄 Sinkronisasi konfigurasi...',
    'CLEANUP_START': '🧹 Membersihkan resources...',
    'CLEANUP_SUCCESS': '✅ Resources berhasil dibersihkan'
}

def get_status_config(status: str) -> Dict[str, Any]:
    """Get configuration untuk package status"""
    status_configs = {
        PACKAGE_STATUS['INSTALLED']: {
            'icon': ICONS['SUCCESS'],
            'color': COLORS['SUCCESS'],
            'text': 'Terinstal',
            'bg_color': '#E8F5E8'
        },
        PACKAGE_STATUS['NOT_INSTALLED']: {
            'icon': ICONS['ERROR'],
            'color': COLORS['ERROR'],
            'text': 'Tidak Terinstal',
            'bg_color': '#FFEBEE'
        },
        PACKAGE_STATUS['CHECKING']: {
            'icon': ICONS['LOADING'],
            'color': COLORS['INFO'],
            'text': 'Memeriksa...',
            'bg_color': '#E3F2FD'
        },
        PACKAGE_STATUS['INSTALLING']: {
            'icon': ICONS['INSTALL'],
            'color': COLORS['INFO'],
            'text': 'Menginstal...',
            'bg_color': '#E3F2FD'
        },
        PACKAGE_STATUS['UPDATING']: {
            'icon': ICONS['UPDATE'],
            'color': COLORS['SECONDARY'],
            'text': 'Memperbarui...',
            'bg_color': '#F3E5F5'
        },
        PACKAGE_STATUS['UNINSTALLING']: {
            'icon': ICONS['UNINSTALL'],
            'color': COLORS['ERROR'],
            'text': 'Menguninstal...',
            'bg_color': '#FFEBEE'
        },
        PACKAGE_STATUS['UPDATE_AVAILABLE']: {
            'icon': ICONS['WARNING'],
            'color': COLORS['WARNING'],
            'text': 'Pembaruan Tersedia',
            'bg_color': '#FFF3E0'
        },
        PACKAGE_STATUS['ERROR']: {
            'icon': ICONS['ERROR'],
            'color': COLORS['ERROR'],
            'text': 'Kesalahan',
            'bg_color': '#FFEBEE'
        }
    }
    
    return status_configs.get(status, status_configs[PACKAGE_STATUS['ERROR']])

def get_button_config(action: str) -> Dict[str, Any]:
    """Get configuration untuk button actions"""
    button_configs = {
        BUTTON_ACTIONS['INSTALL']: {
            'text': 'Instal',
            'icon': ICONS['INSTALL'],
            'color': COLORS['SUCCESS'],
            'variant': 'primary'
        },
        BUTTON_ACTIONS['UPDATE']: {
            'text': 'Perbarui',
            'icon': ICONS['UPDATE'],
            'color': COLORS['WARNING'],
            'variant': 'warning'
        },
        BUTTON_ACTIONS['UNINSTALL']: {
            'text': 'Uninstal',
            'icon': ICONS['UNINSTALL'],
            'color': COLORS['ERROR'],
            'variant': 'danger'
        },
        BUTTON_ACTIONS['CHECK']: {
            'text': 'Periksa',
            'icon': ICONS['CHECK'],
            'color': COLORS['INFO'],
            'variant': 'info'
        }
    }
    
    return button_configs.get(action, button_configs[BUTTON_ACTIONS['CHECK']])