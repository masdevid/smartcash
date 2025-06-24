"""
File: smartcash/ui/setup/env_config/utils.py
Deskripsi: Complete utility functions untuk environment config dengan semua method original
"""

import os
import sys
import time
import warnings
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Import constants dari constants module
try:
    from .constants import (
        REQUIRED_FOLDERS, CONFIG_TEMPLATES, ESSENTIAL_CONFIGS,
        PROGRESS_RANGES, DRIVE_MOUNT_POINT, SMARTCASH_DRIVE_PATH, 
        REPO_CONFIG_PATH, STATUS_MESSAGES, PROGRESS_MESSAGES, RETRY_CONFIG
    )
except ImportError:
    # Fallback constants jika import gagal
    REQUIRED_FOLDERS = ['data', 'models', 'logs', 'config']
    SMARTCASH_DRIVE_PATH = '/content/drive/MyDrive/smartcash'
    REPO_CONFIG_PATH = '/content/drive/MyDrive/smartcash/config'
    DRIVE_MOUNT_POINT = '/content/drive'
    PROGRESS_RANGES = {}
    STATUS_MESSAGES = {}
    PROGRESS_MESSAGES = {}

# === ENVIRONMENT DETECTION ===
def is_colab_environment() -> bool:
    """🔍 Check Colab environment - one-liner"""
    return 'google.colab' in sys.modules or os.path.exists('/content')

def test_drive_readiness() -> bool:
    """🔧 Test Drive readiness - one-liner"""
    try: return Path(DRIVE_MOUNT_POINT).exists() and Path(f'{DRIVE_MOUNT_POINT}/MyDrive').exists()
    except Exception: return False

def refresh_environment_state_silent() -> bool:
    """🔄 Refresh environment state tanpa verbose logging - one-liner"""
    try:
        modules_to_clear = [m for m in sys.modules.keys() if m.startswith('smartcash.ui.setup.env_config')]
        for module in modules_to_clear:
            if hasattr(sys.modules[module], '_cached_state'): delattr(sys.modules[module], '_cached_state')
        return True
    except Exception: return False

def get_system_summary_minimal() -> Dict[str, str]:
    """📊 Get minimal system summary - one-liner"""
    try:
        import platform, psutil
        return {'python_version': platform.python_version(), 'system': platform.system(), 
               'memory_gb': f"{psutil.virtual_memory().total // (1024**3)}GB",
               'drive_status': 'Ready' if test_drive_readiness() else 'Not Ready',
               'environment': 'Colab' if is_colab_environment() else 'Local'}
    except Exception: return {'python_version': 'Unknown', 'system': 'Unknown', 'memory_gb': 'Unknown',
                             'drive_status': 'Unknown', 'environment': 'Colab' if is_colab_environment() else 'Local'}

# === PROGRESS MANAGEMENT ===
def update_progress_safe(ui_components: Dict[str, Any], value: int, message: str = "") -> bool:
    """🔄 Update progress dengan consistent API - one-liner"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'update_overall'): progress_tracker.update_overall(value, message)
        elif 'progress_bar' in ui_components:
            progress_bar = ui_components['progress_bar']
            if hasattr(progress_bar, 'value'): progress_bar.value = value
            if hasattr(progress_bar, 'description'): progress_bar.description = f"{value}%"
            if 'progress_message' in ui_components and message and hasattr(ui_components['progress_message'], 'value'): 
                ui_components['progress_message'].value = message
        return True
    except Exception: return False

def update_step_progress_safe(ui_components: Dict[str, Any], value: int, message: str = "") -> bool:
    """🔄 Update step progress - one-liner"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        return progress_tracker.update_step(value, message) if progress_tracker and hasattr(progress_tracker, 'update_step') else False
    except Exception: return False

def show_progress_safe(ui_components: Dict[str, Any], operation_name: str = None) -> bool:
    """👁️ Show progress tracker - one-liner"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            if hasattr(progress_tracker, 'show_for_operation') and operation_name: progress_tracker.show_for_operation(operation_name)
            elif hasattr(progress_tracker, 'show'): progress_tracker.show()
            return True
        elif 'progress_bar' in ui_components:
            ui_components['progress_bar'].layout.display = 'block'
            return True
        return False
    except Exception: return False

def hide_progress_safe(ui_components: Dict[str, Any]) -> bool:
    """🙈 Hide progress tracker - one-liner"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'hide'): progress_tracker.hide()
        elif 'progress_bar' in ui_components: ui_components['progress_bar'].layout.display = 'none'
        return True
    except Exception: return False

def complete_progress_safe(ui_components: Dict[str, Any], message: str = "✅ Selesai") -> bool:
    """🎉 Complete progress dengan success state - one-liner"""
    return update_progress_safe(ui_components, 100, message)

def error_progress_safe(ui_components: Dict[str, Any], message: str = "❌ Error") -> bool:
    """🚨 Set progress error state - one-liner"""
    return update_progress_safe(ui_components, 0, message)

def reset_progress_safe(ui_components: Dict[str, Any]) -> bool:
    """🔄 Reset progress ke initial state - one-liner"""
    return update_progress_safe(ui_components, 0, "🚀 Siap memulai...")

# === VALIDATION & SETUP ===
def validate_setup_integrity() -> Dict[str, Any]:
    """✅ Validate setup integrity dengan comprehensive check - one-liner"""
    try:
        issues, warnings = [], []
        smartcash_dir = Path(SMARTCASH_DRIVE_PATH)
        
        # Check Drive mount
        if not test_drive_readiness(): issues.append("Google Drive belum mounted atau tidak dapat diakses")
        
        # Check SmartCash directory
        if not smartcash_dir.exists(): issues.append(f"Directory SmartCash tidak ditemukan: {SMARTCASH_DRIVE_PATH}")
        
        # Check required folders
        for folder in REQUIRED_FOLDERS:
            folder_path = smartcash_dir / folder
            if not folder_path.exists(): issues.append(f"Folder tidak ditemukan: {folder}")
        
        # Check repo config directory
        repo_config_dir = Path(REPO_CONFIG_PATH)
        if not repo_config_dir.exists(): warnings.append("Repo configs not available for sync")
        
        return {'valid': len(issues) == 0, 'issues': issues, 'warnings': warnings, 
                'issue_count': len(issues), 'warning_count': len(warnings)}
    except Exception: return {'valid': False, 'issues': ['Validation error'], 'warnings': [], 'issue_count': 1, 'warning_count': 0}

def get_prioritized_missing_items(status: Dict[str, Any] = None) -> List[str]:
    """📋 Get prioritized missing items dari status - one-liner"""
    try:
        if not status: status = validate_setup_integrity()
        missing = status.get('issues', [])
        priority_order = ['drive_mount', 'folders', 'config_files', 'permissions']
        return sorted(missing, key=lambda x: next((i for i, p in enumerate(priority_order) if p in x.lower()), 999))
    except Exception: return []

# === DIRECTORY MANAGEMENT ===
def ensure_directory_exists(path: str) -> bool:
    """📁 Ensure directory exists dengan auto-creation - one-liner"""
    try: Path(path).mkdir(parents=True, exist_ok=True); return True
    except Exception: return False

def setup_directories(required_folders: List[str] = None) -> Tuple[bool, List[str]]:
    """📁 Setup required directories dengan auto-creation - one-liner"""
    try:
        folders = required_folders or REQUIRED_FOLDERS
        base_path = Path(SMARTCASH_DRIVE_PATH)
        created = [folder for folder in folders if ensure_directory_exists(str(base_path / folder))]
        return len(created) == len(folders), created
    except Exception: return False, []

# === STATUS & MESSAGE HELPERS ===
def get_status_message(key: str) -> str:
    """💬 Get localized status message - one-liner"""
    return STATUS_MESSAGES.get(key, f"Status: {key}")

def get_progress_message(key: str) -> str:
    """💬 Get localized progress message - one-liner"""
    return PROGRESS_MESSAGES.get(key, f"Progress: {key}")

def get_progress_range(operation: str) -> Tuple[int, int]:
    """📊 Get progress range untuk operation - one-liner"""
    return PROGRESS_RANGES.get(operation, (0, 100))

def format_log_message(level: str, message: str, emoji: str = "") -> str:
    """📝 Format log message dengan emoji dan level - one-liner"""
    level_emojis = {'INFO': 'ℹ️', 'WARNING': '⚠️', 'ERROR': '❌', 'SUCCESS': '✅', 'DEBUG': '🔍'}
    used_emoji = emoji or level_emojis.get(level.upper(), '📋')
    return f"{used_emoji} {message}"

# === UI STYLING HELPERS ===
def create_flexbox_layout(direction: str = 'row', justify: str = 'space-between', align: str = 'center') -> Dict[str, str]:
    """🎨 Create flexbox layout dengan compact styling - one-liner"""
    return {'display': 'flex', 'flex_direction': direction, 'justify_content': justify, 'align_items': align, 'gap': '10px'}

def create_compact_text_style(font_size: str = '12px', line_height: str = '1.2') -> Dict[str, str]:
    """📝 Create compact text style - one-liner"""
    return {'font_size': font_size, 'line_height': line_height, 'margin': '2px 0', 'padding': '0'}

def create_two_column_flexbox() -> Dict[str, str]:
    """📐 Create two column flexbox layout untuk tips dan requirements - one-liner"""
    return {'display': 'flex', 'flex_direction': 'row', 'justify_content': 'space-between', 'align_items': 'flex-start', 'gap': '20px', 'width': '100%'}

def create_column_style() -> Dict[str, str]:
    """📋 Create individual column style - one-liner"""
    return {'flex': '1', 'min_width': '0', 'padding': '10px', 'background': '#f8f9fa', 'border_radius': '6px', 'border': '1px solid #e9ecef'}

# === FILE & PATH UTILITIES ===
def check_file_exists(filepath: str) -> bool:
    """📄 Check file existence - one-liner"""
    try: return Path(filepath).exists()
    except Exception: return False

def get_safe_path(base_path: str, *parts: str) -> str:
    """🛡️ Get safe path dengan proper joining - one-liner"""
    try: return str(Path(base_path).joinpath(*parts))
    except Exception: return base_path

# === ENVIRONMENT INFO ===
def get_environment_info() -> Dict[str, Any]:
    """🌍 Get comprehensive environment info - one-liner"""
    try:
        colab = is_colab_environment()
        drive_ready = test_drive_readiness()
        system_info = get_system_summary_minimal()
        return {**system_info, 'is_colab': colab, 'drive_ready': drive_ready}
    except Exception: return {'is_colab': False, 'drive_ready': False, 'platform': 'unknown'}

def validate_environment_requirements() -> Dict[str, Any]:
    """✅ Validate environment requirements comprehensive - one-liner"""
    try:
        env_info = get_environment_info()
        setup_valid = validate_setup_integrity()
        return {**env_info, **setup_valid, 'requirements_met': all([env_info.get('is_colab', False), setup_valid.get('drive_ready', False)])}
    except Exception: return {'requirements_met': False, 'error': 'Validation failed'}

# === CONVENIENCE ONE-LINERS ===
is_drive_ready = lambda: test_drive_readiness()
get_smartcash_dir = lambda: Path(SMARTCASH_DRIVE_PATH)
get_repo_config_dir = lambda: Path(REPO_CONFIG_PATH)
setup_needed = lambda: len(get_prioritized_missing_items()) > 0

# === CONSTANTS ===
FLEXBOX_STYLES = {
    'two_column': create_two_column_flexbox(),
    'column': create_column_style(),
    'compact_text': create_compact_text_style()
}

EMOJI_STATUS = {
    'ready': '✅', 'error': '❌', 'warning': '⚠️', 'info': 'ℹ️', 
    'loading': '🔄', 'success': '🎉', 'setup': '🚀', 'check': '🔍'
}

def debug_ui_components(ui_components: Dict[str, Any]) -> str:
    """🔧 Debug UI components structure - one-liner"""
    try: return f"UI Components: {list(ui_components.keys())}" 
    except Exception: return "UI Components: Error getting keys"