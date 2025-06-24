"""
File: smartcash/ui/setup/env_config/utils.py  
Deskripsi: Complete utils untuk environment config dengan semua required functions
"""

import time
from typing import Dict, Any, Tuple
from pathlib import Path
from .constants import (
    REQUIRED_FOLDERS, CONFIG_TEMPLATES, ESSENTIAL_CONFIGS,
    PROGRESS_RANGES, DRIVE_MOUNT_POINT, SMARTCASH_DRIVE_PATH, 
    REPO_CONFIG_PATH, STATUS_MESSAGES, PROGRESS_MESSAGES, RETRY_CONFIG
)

# === PROGRESS UTILS ===
def update_progress_safe(ui_components: Dict[str, Any], value: int, message: str = "") -> None:
    """🔄 Update progress dengan consistent API"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(value, message)
        elif 'progress_bar' in ui_components:
            progress_bar = ui_components['progress_bar']
            if hasattr(progress_bar, 'value'):
                progress_bar.value = value
            if hasattr(progress_bar, 'description'):
                progress_bar.description = f"{value}%"
            if 'progress_message' in ui_components and message:
                message_widget = ui_components['progress_message']
                if hasattr(message_widget, 'value'):
                    message_widget.value = message
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].debug(f"🔍 Progress update error: {str(e)}")

def update_step_progress_safe(ui_components: Dict[str, Any], value: int, message: str = "") -> None:
    """🔄 Update step progress"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'update_step'):
            progress_tracker.update_step(value, message)
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].debug(f"🔍 Step progress error: {str(e)}")

def show_progress_safe(ui_components: Dict[str, Any], operation_name: str = None) -> None:
    """👁️ Show progress tracker"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            if hasattr(progress_tracker, 'show_for_operation') and operation_name:
                progress_tracker.show_for_operation(operation_name)
            elif hasattr(progress_tracker, 'show'):
                progress_tracker.show()
        elif 'progress_container' in ui_components:
            container = ui_components['progress_container']
            if hasattr(container, 'layout'):
                container.layout.visibility = 'visible'
                container.layout.display = 'block'
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].debug(f"🔍 Show progress error: {str(e)}")

def hide_progress_safe(ui_components: Dict[str, Any]) -> None:
    """🙈 Hide progress tracker"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'hide'):
            progress_tracker.hide()
        elif 'progress_container' in ui_components:
            container = ui_components['progress_container']
            if hasattr(container, 'layout'):
                container.layout.visibility = 'hidden'
                container.layout.display = 'none'
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].debug(f"🔍 Hide progress error: {str(e)}")

def complete_progress_safe(ui_components: Dict[str, Any], message: str) -> None:
    """✅ Complete progress operation"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'complete_operation'):
            progress_tracker.complete_operation(message)
        else:
            update_progress_safe(ui_components, 100, message)
            time.sleep(1)
            hide_progress_safe(ui_components)
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].debug(f"🔍 Complete progress error: {str(e)}")

def error_progress_safe(ui_components: Dict[str, Any], message: str) -> None:
    """❌ Error progress operation"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'error_operation'):
            progress_tracker.error_operation(message)
        else:
            hide_progress_safe(ui_components)
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].debug(f"🔍 Error progress error: {str(e)}")

def reset_progress_safe(ui_components: Dict[str, Any]) -> None:
    """🔄 Reset progress state"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
        elif 'progress_bar' in ui_components:
            progress_bar = ui_components['progress_bar']
            if hasattr(progress_bar, 'value'):
                progress_bar.value = 0
            if hasattr(progress_bar, 'description'):
                progress_bar.description = "0%"
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].debug(f"🔍 Reset progress error: {str(e)}")

# === ENVIRONMENT UTILS ===
def is_colab_environment() -> bool:
    """🔍 Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def test_drive_readiness(drive_path: Path = None) -> bool:
    """📱 Test Google Drive readiness"""
    if drive_path is None:
        drive_path = Path(DRIVE_MOUNT_POINT)
    
    try:
        if drive_path.exists() and drive_path.is_dir():
            # Test write access
            test_file = drive_path / 'smartcash_test.tmp'
            test_file.write_text('test')
            test_file.unlink()
            return True
        return False
    except Exception:
        return False

def wait_for_drive_ready(max_wait: int = 30, ui_components: Dict[str, Any] = None) -> Tuple[bool, str]:
    """⏳ Wait for Drive to be ready dengan timeout"""
    for i in range(max_wait):
        if test_drive_readiness():
            return True, STATUS_MESSAGES['drive_ready']
        
        if i % 5 == 0 and i > 0 and ui_components:
            progress = 15 + (i/max_wait) * 10
            update_progress_safe(ui_components, int(progress), f"⏳ Validating Drive state... ({i}s)")
        
        time.sleep(1)
    
    return False, f"Timeout setelah {max_wait}s - Drive tidak dapat divalidasi"

# === FOLDER & CONFIG OPERATIONS ===
def create_folder_with_retry(folder_path: Path, attempts: int = 3) -> bool:
    """📁 Create folder dengan retry mechanism"""
    for attempt in range(attempts):
        try:
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                time.sleep(0.2)
            return folder_path.exists() and folder_path.is_dir()
        except Exception:
            if attempt < attempts - 1:
                time.sleep(0.5)
    return False

def validate_folders(drive_base_path: str) -> Dict[str, bool]:
    """📁 Validate Drive folders"""
    drive_base = Path(drive_base_path)
    return {folder: (folder_path := drive_base / folder).exists() and folder_path.is_dir() 
            for folder in REQUIRED_FOLDERS}

def validate_configs(drive_base_path: str) -> Dict[str, bool]:
    """📋 Validate Drive configs"""
    drive_config_path = Path(drive_base_path) / 'configs'
    return {config: (config_file := drive_config_path / config).exists() and config_file.is_file() 
            for config in CONFIG_TEMPLATES}

def validate_repo_configs() -> Dict[str, bool]:
    """📋 Validate repo configs"""
    repo_config_path = Path(REPO_CONFIG_PATH)
    return {config: (repo_config_path / config).exists() for config in CONFIG_TEMPLATES}

# === SYMLINK OPERATIONS ===
def check_symlink_status(local_path: Path) -> Dict[str, Any]:
    """🔗 Check single symlink status"""
    if not local_path.exists():
        return {'exists': False, 'is_symlink': False, 'valid': False}
    elif local_path.is_symlink():
        try:
            return {'exists': True, 'is_symlink': True, 'valid': local_path.resolve().exists()}
        except Exception:
            return {'exists': True, 'is_symlink': True, 'valid': False}
    else:
        return {'exists': True, 'is_symlink': False, 'valid': True}

def validate_symlinks() -> Dict[str, Dict[str, Any]]:
    """🔗 Validate all symlinks"""
    if not is_colab_environment():
        return {folder: {'exists': True, 'is_symlink': False, 'valid': True} 
                for folder in REQUIRED_FOLDERS}
    
    return {folder: check_symlink_status(Path(f'/content/{folder}')) 
            for folder in REQUIRED_FOLDERS}

def create_symlink_with_retry(source_path: Path, target_path: Path, attempts: int = 3) -> bool:
    """🔗 Create symlink dengan retry"""
    for attempt in range(attempts):
        try:
            if target_path.exists():
                if target_path.is_symlink():
                    target_path.unlink()
                else:
                    return False  # Don't overwrite existing non-symlink
            
            target_path.symlink_to(source_path)
            return target_path.is_symlink() and target_path.resolve().exists()
        except Exception:
            if attempt < attempts - 1:
                time.sleep(RETRY_CONFIG['symlink_delay'])
    return False

# === STATUS EVALUATION ===
def evaluate_readiness(drive_status: Dict[str, Any], repo_configs: Dict[str, bool], 
                      drive_folders: Dict[str, bool], drive_configs: Dict[str, bool], 
                      symlinks: Dict[str, Dict[str, Any]]) -> bool:
    """✅ Evaluate overall readiness"""
    return (drive_status['mounted'] and drive_status.get('ready', False) and
            all(repo_configs.values()) and all(drive_folders.values()) and
            sum(drive_configs.values()) >= len(ESSENTIAL_CONFIGS) and
            all(link['valid'] for link in symlinks.values()))

def get_missing_items(repo_configs: Dict[str, bool], drive_folders: Dict[str, bool], 
                     drive_configs: Dict[str, bool], symlinks: Dict[str, Dict[str, Any]]) -> Dict[str, list]:
    """📋 Get missing items untuk diagnostics"""
    return {
        'missing_repo_configs': [k for k, v in repo_configs.items() if not v],
        'missing_drive_folders': [k for k, v in drive_folders.items() if not v],
        'missing_drive_configs': [k for k, v in drive_configs.items() if not v],
        'invalid_symlinks': [k for k, v in symlinks.items() if not v['valid']]
    }

def get_prioritized_missing_items(env_status: Dict[str, Any] = None) -> list:
    """🎯 Get prioritized missing items untuk display"""
    if env_status is None:
        # Simple fallback
        return []
    
    missing_items = []
    
    if not env_status.get('drive', {}).get('mounted', False):
        missing_items.append("Google Drive")
    
    missing_folders = env_status.get('missing_drive_folders', [])[:2]
    missing_items.extend([f"folder {folder}" for folder in missing_folders])
    
    missing_configs = env_status.get('missing_drive_configs', [])[:2]  
    missing_items.extend([f"config {config.replace('_config.yaml', '')}" for config in missing_configs])
    
    return missing_items

# === SETUP VALIDATION ===
def validate_setup_integrity(drive_base_path: str = None) -> Dict[str, Any]:
    """✅ Validate setup integrity"""
    if drive_base_path is None:
        drive_base_path = SMARTCASH_DRIVE_PATH
    
    issues = []
    warnings = []
    
    smartcash_dir = Path(drive_base_path)
    
    # Check essential directories
    for folder in REQUIRED_FOLDERS:
        folder_path = smartcash_dir / folder
        if not folder_path.exists():
            issues.append(f"Missing folder: {folder}")
    
    # Check essential configs
    config_dir = smartcash_dir / 'configs'
    for config in ESSENTIAL_CONFIGS:
        config_path = config_dir / config
        if not config_path.exists():
            issues.append(f"Missing config: {config}")
    
    # Check repo configs availability
    repo_config_dir = Path(REPO_CONFIG_PATH)
    if not repo_config_dir.exists():
        warnings.append("Repo configs not available for sync")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'issue_count': len(issues),
        'warning_count': len(warnings)
    }

def refresh_environment_state_silent(env_manager=None) -> bool:
    """🔄 Refresh environment state tanpa verbose logging
    
    Args:
        env_manager: Optional environment manager instance (for backward compatibility)
        
    Returns:
        bool: True if refresh was successful, False otherwise
    """
    try:
        # Clear any cached state
        import sys
        modules_to_clear = [
            module for module in sys.modules.keys() 
            if module.startswith('smartcash.ui.setup.env_config')
        ]
        
        for module in modules_to_clear:
            if hasattr(sys.modules[module], '_cached_state'):
                delattr(sys.modules[module], '_cached_state')
        
        return True
    except Exception:
        return False

# === RANGE HELPERS ===
def get_progress_range(operation: str) -> Tuple[int, int]:
    """📊 Get progress range untuk operation"""
    return PROGRESS_RANGES.get(operation, (0, 100))

def get_status_message(key: str) -> str:
    """💬 Get localized status message"""
    return STATUS_MESSAGES.get(key, f"Status: {key}")

def get_progress_message(key: str) -> str:
    """💬 Get localized progress message"""
    return PROGRESS_MESSAGES.get(key, f"Progress: {key}")

# === SYSTEM INFO UTILS ===
def get_system_summary_minimal() -> Dict[str, str]:
    """📊 Get minimal system summary"""
    try:
        import platform
        import psutil
        
        return {
            'python_version': platform.python_version(),
            'system': platform.system(),
            'memory_gb': f"{psutil.virtual_memory().total // (1024**3)}GB",
            'drive_status': 'Ready' if test_drive_readiness() else 'Not Ready',
            'environment': 'Colab' if is_colab_environment() else 'Local'
        }
    except Exception:
        return {
            'python_version': 'Unknown',
            'system': 'Unknown', 
            'memory_gb': 'Unknown',
            'drive_status': 'Unknown',
            'environment': 'Colab' if is_colab_environment() else 'Local'
        }

# === ONE-LINER CONVENIENCE FUNCTIONS ===
is_drive_ready = lambda: test_drive_readiness()
get_smartcash_dir = lambda: Path(SMARTCASH_DRIVE_PATH)
get_repo_config_dir = lambda: Path(REPO_CONFIG_PATH)
setup_needed = lambda: get_prioritized_missing_items().get('setup_needed', True) if get_prioritized_missing_items() else True