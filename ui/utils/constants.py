"""
File: smartcash/ui/utils/constants.py
Author: Refactored
Deskripsi: Konstanta untuk komponen UI dengan struktur yang lebih terorganisir dan warna konsisten
"""

# Color palette
COLORS = {
    'primary': '#3498db',
    'secondary': '#2c3e50',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#C62300',
    'muted': '#6c757d',
    'highlight': '#e65100',
    'background': '#ffffff',
    'card': '#f8f9fa',
    'border': '#dee2e6',
    'header_bg': '#f0f8ff',
    'header_border': '#3498db',
    
    # Alert colors
    'alert_info_bg': '#d1ecf1',
    'alert_info_text': '#0c5460',
    'alert_success_bg': '#d4edda',
    'alert_success_text': '#155724',
    'alert_warning_bg': '#fff3cd',
    'alert_warning_text': '#856404',
    'alert_danger_bg': '#f8d7da',
    'alert_danger_text': '#721c24',
}

# Tema
THEMES = {
    'default': {
        **COLORS
    },
    'dark': {
        'primary': '#0d6efd',
        'secondary': '#6c757d',
        'success': '#198754',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#0dcaf0',
        'light': '#212529',
        'dark': '#f8f9fa',
        'muted': '#adb5bd',
        'highlight': '#fd7e14',
        'background': '#212529',
        'card': '#343a40',
        'border': '#495057',
        'header_bg': '#343a40',
        'header_border': '#0d6efd',
        'alert_info_bg': '#1a3a42',
        'alert_info_text': '#8edaec',
        'alert_success_bg': '#0e3a1d',
        'alert_success_text': '#8eda99',
        'alert_warning_bg': '#3a3000',
        'alert_warning_text': '#ffe066',
        'alert_danger_bg': '#3a0a0d',
        'alert_danger_text': '#ea868f',
    }
}

# Active theme (can be changed at runtime)
ACTIVE_THEME = 'default'

# Font Awesome icon mapping for ipywidgets
FA_ICONS = {
    'check': 'check',
    'times': 'times',
    'warning': 'exclamation-triangle',
    'info': 'info-circle',
    'error': 'exclamation-circle',
    'success': 'check-circle',
    'refresh': 'sync',
    'save': 'save',
    'upload': 'upload',
    'download': 'download',
    'folder': 'folder',
    'file': 'file',
    'plus': 'plus',
    'minus': 'minus',
    'edit': 'edit',
    'delete': 'trash',
    'search': 'search',
    'settings': 'cog',
    'link': 'link',
    'folder-plus': 'folder-plus',
    'folder-open': 'folder-open',
    'chart': 'chart-bar',
    'camera': 'camera',
    'play': 'play',
    'pause': 'pause',
    'stop': 'stop',
    'code': 'code',
    'home': 'home',
    'cog': 'cog',
    'arrow-up': 'arrow-up',
    'arrow-down': 'arrow-down',
    'arrow-left': 'arrow-left',
    'arrow-right': 'arrow-right',
}

# Emoji icons
ICONS = {
    # Status icons
    'success': '✅',
    'warning': '⚠️',
    'error': '❌',
    'info': 'ℹ️',
    
    # Action icons
    'config': '⚙️',
    'data': '📊',
    'processing': '🔄',
    'start': '🚀',
    'download': '📥',
    'upload': '📤',
    'save': '💾',
    'add': '➕',
    'remove': '➖',
    'edit': '✏️',
    'delete': '🗑️',
    'search': '🔍',
    
    # Object icons
    'folder': '📁',
    'file': '📄',
    'model': '🧠',
    'time': '⏱️',
    'calendar': '📅',
    'metric': '📈',
    'stats': '📊',
    'chart': '📊',
    
    # Domain-specific icons
    'settings': '🔧',
    'tools': '🛠️',
    'split': '✂️',
    'augmentation': '🎨',
    'training': '🏋️',
    'evaluation': '🔍',
    'cleanup': '🧹',
    'dataset': '📚',
    'cache': '💽',
    
    # Control icons
    'stop': '🛑',
    'pause': '⏸️',
    'resume': '▶️',
    'play': '▶️',
    'next': '⏭️',
    'prev': '⏮️',
    'medal': '🏆',
    'check': '✓',
    'times': '✗',
}

# Alert styles
ALERT_STYLES = {
    'info': {
        'bg_color': COLORS['alert_info_bg'],
        'text_color': COLORS['alert_info_text'],
        'border_color': COLORS['alert_info_text'],
        'icon': ICONS['info']
    },
    'success': {
        'bg_color': COLORS['alert_success_bg'],
        'text_color': COLORS['alert_success_text'],
        'border_color': COLORS['alert_success_text'],
        'icon': ICONS['success']
    },
    'warning': {
        'bg_color': COLORS['alert_warning_bg'],
        'text_color': COLORS['alert_warning_text'],
        'border_color': COLORS['alert_warning_text'],
        'icon': ICONS['warning']
    },
    'error': {
        'bg_color': COLORS['alert_danger_bg'],
        'text_color': COLORS['alert_danger_text'],
        'border_color': COLORS['alert_danger_text'],
        'icon': ICONS['error']
    }
}

# Button styles
BUTTON_STYLES = {
    'primary': 'primary',
    'success': 'success',
    'info': 'info',
    'warning': 'warning',
    'danger': 'danger',
    'default': ''
}

# Font Config
FONTS = {
    'default': '"Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    'monospace': 'Consolas, Menlo, Monaco, "Courier New", monospace',
    'header': '"Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
}

# Size Config
SIZES = {
    'xs': '0.75rem',   # 12px
    'sm': '0.875rem',  # 14px
    'md': '1rem',      # 16px
    'lg': '1.25rem',   # 20px
    'xl': '1.5rem',    # 24px
    '2xl': '2rem',     # 32px
}

# Layout constants
PADDINGS = {
    'none': '0',
    'small': '5px',
    'medium': '10px',
    'large': '15px',
    'xl': '20px'
}

MARGINS = {
    'none': '0',
    'small': '5px',
    'medium': '10px',
    'large': '15px',
    'xl': '20px'
}

# File Related
IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']
VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv']
DOCUMENT_EXTENSIONS = ['pdf', 'doc', 'docx', 'txt', 'csv', 'xls', 'xlsx', 'ppt', 'pptx']
CODE_EXTENSIONS = ['py', 'js', 'java', 'cpp', 'c', 'h', 'html', 'css', 'json', 'xml']

# For file size formatting
FILE_SIZE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']