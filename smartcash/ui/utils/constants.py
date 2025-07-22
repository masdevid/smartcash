"""
File: smartcash/ui/utils/constants.py
Deskripsi: Updated constants dengan missing icons untuk preprocessing UI - merged dengan existing
"""

# Import existing constants untuk merge
try:
    from smartcash.common.constants.ui import (
        STATUS_ICONS, ACTION_ICONS, DOMAIN_ICONS, UI_COLORS
    )
    
    # Use existing color structure
    COLORS = {
        'primary': UI_COLORS.get('primary', '#007bff'),
        'secondary': UI_COLORS.get('secondary', '#6c757d'),
        'success': UI_COLORS.get('success', '#28a745'),
        'warning': UI_COLORS.get('warning', '#ffc107'),
        'danger': UI_COLORS.get('danger', '#dc3545'),
        'info': UI_COLORS.get('info', '#17a2b8'),
        'light': UI_COLORS.get('light', '#f8f9fa'),
        'dark': UI_COLORS.get('dark', '#343a40'),
        'muted': UI_COLORS.get('muted', '#6c757d'),
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
        
        # Text colors
        "text": "#212529",
        "text_muted": "#6c757d",
        
        # Background colors
        "bg_light": "#f8f9fa",
        "bg_white": "#ffffff",
        "bg_dark": "#343a40"
    }
    
    # Merge existing icons dengan missing icons
    ICONS = {
        # From existing constants
        'success': STATUS_ICONS.get('success', '✅'),
        'warning': STATUS_ICONS.get('warning', '⚠️'),
        'error': STATUS_ICONS.get('error', '❌'),
        'info': STATUS_ICONS.get('info', 'ℹ️'),
        'debug': STATUS_ICONS.get('debug', '🔍'),
        'processing': STATUS_ICONS.get('processing', '🔧'),
        'waiting': STATUS_ICONS.get('waiting', '⏳'),
        'complete': STATUS_ICONS.get('complete', '✅'),
        
        # Action icons from existing
        'config': ACTION_ICONS.get('config', '⚙️'),
        'start': ACTION_ICONS.get('start', '▶️'),
        'download': ACTION_ICONS.get('download', '📥'),
        'upload': ACTION_ICONS.get('upload', '📤'),
        'save': ACTION_ICONS.get('save', '💾'),
        'add': ACTION_ICONS.get('add', '➕'),
        'remove': ACTION_ICONS.get('remove', '➖'),
        'edit': ACTION_ICONS.get('edit', '✏️'),
        'delete': ACTION_ICONS.get('delete', '🗑️'),
        'search': ACTION_ICONS.get('search', '🔍'),
        'refresh': ACTION_ICONS.get('refresh', '🔄'),
        
        # Domain icons from existing
        'model': DOMAIN_ICONS.get('model', '🧠'),
        'dataset': DOMAIN_ICONS.get('dataset', '📊'),
        'augmentation': DOMAIN_ICONS.get('augmentation', '🔄'),
        'training': DOMAIN_ICONS.get('training', '🚀'),
        'evaluation': DOMAIN_ICONS.get('evaluation', '📊'),
        
        # Missing icons yang dibutuhkan preprocessing UI
        "action": "▶️",  # Fixed: added missing 'action' icon
        "play": "▶️",
        "stop": "⏹️",
        "pause": "⏸️",
        "reset": "🔄",
        
        # Settings & Configuration
        "settings": "⚙️",
        "gear": "⚙️",
        
        # Data & Files
        "folder": "📁",
        "file": "📄",
        "image": "🖼️",
        "data": "💾",
        
        # Status & Feedback
        "question": "❓",
        "check": "✅",
        "cross": "❌",
        
        # Search & Analysis
        "analyze": "📊",
        "stats": "📊",
        "chart": "📈",
        "graph": "📉",
        
        # Navigation & UI
        "arrow_right": "➡️",
        "arrow_left": "⬅️",
        "arrow_up": "⬆️",
        "arrow_down": "⬇️",
        "expand": "🔽",
        "collapse": "🔼",
        
        # Tools & Utilities
        "clean": "🧹",
        "cleanup": "🧹",
        "trash": "🗑️",
        "copy": "📋",
        "link": "🔗",
        
        # Progress & Status
        "progress": "📊",
        "loading": "⏳",
        "clock": "🕐",
        "timer": "⏱️",
        
        # Validation & Testing
        "validate": "✅",
        "test": "🧪",
        "debug": "🐛",
        "fix": "🔧",
        
        # AI & Model
        "ai": "🤖",
        "train": "🚀",
        "predict": "🎯",
        
        # Dataset & Preprocessing
        "preprocess": "🔧",
        "augment": "🔄",
        "split": "✂️",
        
        # System & Environment
        "system": "💻",
        "cloud": "☁️",
        "drive": "💽",
        "sync": "🔄",
        
        # Additional missing icons
        'time': '⏱️',
        'calendar': '📅',
        'metric': '📈',
        'compare': '🔄',
        'idle': '🏁',
        'running': '🚀',
        'tools': '🛠️',
        'cache': '💽',
        'visualize': '👁️',
        'next': '⏭️',
        'prev': '⏮️',
        'medal': '🏆',
        'times': '✗',
        'resume': '▶️'
    }

except ImportError:
    # Fallback jika common.constants.ui tidak tersedia
    COLORS = {
        "primary": "#007bff",
        "secondary": "#6c757d", 
        "success": "#28a745",
        "danger": "#dc3545",
        "warning": "#ffc107",
        "info": "#17a2b8",
        "light": "#f8f9fa",
        "dark": "#343a40",
        "muted": "#6c757d",
        "text": "#212529",
        "text_muted": "#6c757d",
        "bg_light": "#f8f9fa",
        "bg_white": "#ffffff",
        "bg_dark": "#343a40",
        "header_bg": "#f0f8ff",
        "header_border": "#3498db"
    }
    
    # Fallback icons
    ICONS = {
        "action": "▶️",
        "processing": "🔧",
        "settings": "⚙️",
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️",
        "play": "▶️",
        "stop": "⏹️",
        "pause": "⏸️",
        "refresh": "🔄",
        "reset": "🔄",
        "download": "📥",
        "upload": "📤",
        "save": "💾",
        "search": "🔍",
        "cleanup": "🧹",
        "dataset": "📊",
        "model": "🧠"
    }

# Default style values for fallback
DEFAULT_STYLE = {
    'bg_color': '#f8f9fa',
    'text_color': '#212529',
    'border_color': '#6c757d',
    'icon': 'ℹ️'
}

# Alert styles untuk berbagai tipe status
ALERT_STYLES = {
    'info': {
        'bg_color': '#d1ecf1',
        'text_color': '#0c5460',
        'border_color': '#17a2b8',
        'icon': ICONS.get('info', 'ℹ️')
    },
    'success': {
        'bg_color': '#d4edda',
        'text_color': '#155724',
        'border_color': '#28a745',
        'icon': ICONS.get('success', '✅')
    },
    'warning': {
        'bg_color': '#fff3cd',
        'text_color': '#856404',
        'border_color': '#ffc107',
        'icon': ICONS.get('warning', '⚠️')
    },
    'error': {
        'bg_color': '#f8d7da',
        'text_color': '#721c24',
        'border_color': '#dc3545',
        'icon': ICONS.get('error', '❌')
    },
    'debug': {
        'bg_color': '#e2e3e5',
        'text_color': '#383d41',
        'border_color': '#6c757d',
        'icon': '🔍'
    }
}

def get_alert_style(style_type: str) -> dict:
    """Get alert style with fallback to default if style_type is invalid"""
    return ALERT_STYLES.get(style_type.lower(), DEFAULT_STYLE)

# File extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}

# Button styles mapping
BUTTON_STYLES = {
    'primary': 'primary',
    'secondary': '',
    'success': 'success',
    'danger': 'danger',
    'warning': 'warning',
    'info': 'info'
}

# Layout constants
LAYOUT = {
    'default_width': '100%',
    'default_padding': '10px',
    'default_margin': '5px 0',
    'container_padding': '15px',
    'section_margin': '20px 0',
    'button_height': '35px',
    'input_height': '32px'
}