"""
File: smartcash/ui/setup/dependency_installer/utils/constants.py
Deskripsi: Consolidated constants untuk dependency installer dengan DRY approach
"""

from typing import Dict, Any
from smartcash.ui.utils.constants import COLORS, ICONS

# Consolidated status configurations dengan one-liner approach
STATUS_CONFIGS = {
    level: {
        "emoji": ICONS.get(icon_key, fallback_emoji),
        "color": COLORS.get(color_key, fallback_color),
        "bg_color": bg_color,
        "border_color": COLORS.get(color_key, fallback_color),
        "text": text
    }
    for level, (icon_key, fallback_emoji, color_key, fallback_color, bg_color, text) in {
        "info": ("info", "ℹ️", "info", "#17a2b8", "#d1ecf1", "Informasi"),
        "success": ("success", "✅", "success", "#28a745", "#d4edda", "Terinstall"),
        "warning": ("warning", "⚠️", "warning", "#ffc107", "#fff3cd", "Perlu update"),
        "error": ("error", "❌", "danger", "#dc3545", "#f8d7da", "Error"),
        "checking": ("debug", "🔍", "info", "#17a2b8", "#e3f2fd", "Checking..."),
        "installing": ("processing", "⏳", "info", "#17a2b8", "#e3f2fd", "Installing...")
    }.items()
}

# Package status mapping untuk UI - one-liner
PACKAGE_STATUS_MAPPING = {
    status: {"icon": config["emoji"], "color": config["color"], "text": config["text"]}
    for status, config in STATUS_CONFIGS.items()
}

# Installation constants
INSTALLATION_DEFAULTS = {
    'parallel_workers': 3,
    'force_reinstall': False,
    'use_cache': True,
    'timeout': 300,
    'max_retries': 2,
    'retry_delay': 1.0
}

# Analysis constants
ANALYSIS_DEFAULTS = {
    'check_compatibility': True,
    'include_dev_deps': False,
    'batch_size': 10,
    'detailed_info': True
}

# Progress step constants yang ter-consolidated
PROGRESS_STEPS = {
    # Installation flow
    'install': {
        'init': 5,
        'analysis': 15,
        'start': 20,
        'end': 90,
        'finalize': 100
    },
    # Analysis flow
    'analysis': {
        'init': 10,
        'get_packages': 30,
        'categories': 50,
        'check': 60,
        'update_ui': 90,
        'complete': 100
    },
    # Status check flow
    'status_check': {
        'init': 10,
        'system_info': 30,
        'package_check': 60,
        'report': 80,
        'ui_update': 90,
        'complete': 100
    }
}

# UI Messages constants - one-liner templates
UI_MESSAGES = {
    'no_packages_selected': "❌ Tidak ada packages yang dipilih",
    'all_packages_installed': "✅ Semua packages sudah terinstall dengan benar",
    'analysis_complete': "📊 Analisis packages selesai",
    'installation_complete': "🎉 Instalasi packages selesai",
    'status_check_complete': "📋 Status check selesai",
    'config_saved': "💾 Konfigurasi tersimpan",
    'config_reset': "🔄 Konfigurasi direset ke default",
    'operation_failed': "💥 Operasi gagal: {error}",
    'system_compatible': "✅ Sistem kompatibel",
    'system_incompatible': "⚠️ Sistem tidak memenuhi requirements"
}

def get_status_config(level: str) -> Dict[str, Any]:
    """Get status configuration dengan fallback - one-liner"""
    return STATUS_CONFIGS.get(level, STATUS_CONFIGS["info"])

def get_package_status_config(status: str) -> Dict[str, Any]:
    """Get package status configuration - one-liner"""
    return PACKAGE_STATUS_MAPPING.get(status, PACKAGE_STATUS_MAPPING["checking"])

def get_progress_step(operation: str, step: str) -> int:
    """Get progress step value - one-liner dengan fallback"""
    return PROGRESS_STEPS.get(operation, {}).get(step, 0)

def get_ui_message(message_key: str, **kwargs) -> str:
    """Get UI message dengan format variables - one-liner"""
    template = UI_MESSAGES.get(message_key, message_key)
    return template.format(**kwargs) if kwargs else template

def get_installation_default(key: str) -> Any:
    """Get installation default value - one-liner"""
    return INSTALLATION_DEFAULTS.get(key)

def get_analysis_default(key: str) -> Any:
    """Get analysis default value - one-liner"""
    return ANALYSIS_DEFAULTS.get(key)

# Package categories constants yang bisa di-override
DEFAULT_PACKAGE_CATEGORIES = [
    {
        'name': 'Core Requirements',
        'icon': '🔧',
        'description': 'Package inti SmartCash',
        'priority': 1
    },
    {
        'name': 'ML/AI Libraries', 
        'icon': '🤖',
        'description': 'Machine Learning frameworks',
        'priority': 2
    },
    {
        'name': 'Data Processing',
        'icon': '📊', 
        'description': 'Data manipulation tools',
        'priority': 3
    }
]

# Status icons untuk berbagai kondisi - consolidated dari multiple files
CONSOLIDATED_STATUS_ICONS = {
    **{status: config["emoji"] for status, config in STATUS_CONFIGS.items()},
    'loading': '⏳',
    'complete': '✅', 
    'failed': '❌',
    'skipped': '⏭️',
    'retry': '🔄',
    'timeout': '⏰'
}

def create_status_badge_html(status: str, message: str = "") -> str:
    """Create status badge HTML - one-liner template"""
    config = get_status_config(status)
    display_message = message or config["text"]
    
    return f"""
    <span style="background:{config['bg_color']};color:{config['color']};
                 padding:2px 6px;border-radius:3px;font-size:11px;
                 border-left:2px solid {config['border_color']};">
        {config['emoji']} {display_message}
    </span>
    """

def create_progress_message(operation: str, step: str, custom_message: str = "") -> str:
    """Create progress message dengan operation context - one-liner"""
    progress_value = get_progress_step(operation, step)
    base_message = custom_message or f"{operation.title()} {step}"
    return f"[{progress_value}%] {base_message}"

# Validation constants
VALIDATION_RULES = {
    'python_min_version': (3, 7),
    'memory_min_gb': 2.0,
    'disk_min_gb': 1.0,
    'supported_platforms': ['Linux', 'Windows', 'Darwin']
}

def check_validation_rule(rule_name: str, value: Any) -> bool:
    """Check validation rule - one-liner dengan type handling"""
    rule = VALIDATION_RULES.get(rule_name)
    
    if rule_name == 'python_min_version' and isinstance(value, str):
        try:
            major, minor = map(int, value.split('.')[:2])
            return (major, minor) >= rule
        except:
            return False
    elif rule_name in ['memory_min_gb', 'disk_min_gb']:
        return value >= rule
    elif rule_name == 'supported_platforms':
        return value in rule
    
    return False