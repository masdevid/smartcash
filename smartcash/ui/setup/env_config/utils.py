"""
File: smartcash/ui/setup/env_config/utils.py  
Deskripsi: Fixed progress tracker utils dengan consistent API calls
"""

import time
from typing import Dict, Any, Tuple
from pathlib import Path
from .constants import (
    REQUIRED_FOLDERS, CONFIG_TEMPLATES, ESSENTIAL_CONFIGS,
    PROGRESS_RANGES, DRIVE_MOUNT_POINT, SMARTCASH_DRIVE_PATH, 
    REPO_CONFIG_PATH, STATUS_MESSAGES, PROGRESS_MESSAGES, RETRY_CONFIG
)

# === PROGRESS UTILS - FIXED API ===
def update_progress_safe(ui_components: Dict[str, Any], value: int, message: str = "") -> None:
    """🔄 Update progress dengan consistent API - fixed"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'update_overall'):
            # Use correct API from SHARED_COMPONENTS_README.md
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
        # Log instead of silent fail
        if 'logger' in ui_components:
            ui_components['logger'].debug(f"🔍 Progress update error: {str(e)}")

def update_step_progress_safe(ui_components: Dict[str, Any], value: int, message: str = "") -> None:
    """🔄 Update step progress - new method"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'update_step'):
            progress_tracker.update_step(value, message)
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].debug(f"🔍 Step progress error: {str(e)}")

def show_progress_safe(ui_components: Dict[str, Any], operation_name: str = None) -> None:
    """👁️ Show progress tracker - fixed"""
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
    """🙈 Hide progress tracker - fixed"""
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
    """✅ Complete progress operation - new method"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'complete_operation'):
            progress_tracker.complete_operation(message)
        else:
            # Fallback to 100% progress
            update_progress_safe(ui_components, 100, message)
            time.sleep(1)  # Brief pause to show completion
            hide_progress_safe(ui_components)
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].debug(f"🔍 Complete progress error: {str(e)}")

def error_progress_safe(ui_components: Dict[str, Any], message: str) -> None:
    """❌ Error progress operation - new method"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'error_operation'):
            progress_tracker.error_operation(message)
        else:
            # Fallback to hide progress
            hide_progress_safe(ui_components)
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].debug(f"🔍 Error progress error: {str(e)}")

def reset_progress_safe(ui_components: Dict[str, Any]) -> None:
    """🔄 Reset progress state - fixed"""
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

def test_drive_readiness(max_retries: int = 3) -> Tuple[bool, str]:
    """📱 Test Google Drive readiness dengan retry"""
    drive_path = Path(DRIVE_MOUNT_POINT)
    
    for attempt in range(max_retries):
        try:
            if drive_path.exists() and drive_path.is_dir():
                # Test write access
                test_file = drive_path / 'smartcash_test.tmp'
                test_file.write_text('test')
                test_file.unlink()
                return True, "Drive ready and writable"
        except Exception as e:
            if attempt == max_retries - 1:
                return False, f"Drive test failed: {str(e)}"
            time.sleep(RETRY_CONFIG['drive_ready_delay'])
    
    return False, "Drive not accessible"

def validate_setup_integrity() -> Dict[str, Any]:
    """✅ Validate setup integrity"""
    issues = []
    warnings = []
    
    smartcash_dir = Path(SMARTCASH_DRIVE_PATH)
    
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

def get_prioritized_missing_items() -> Dict[str, Any]:
    """🎯 Get prioritized missing items untuk setup"""
    smartcash_dir = Path(SMARTCASH_DRIVE_PATH)
    
    missing_essential = []
    missing_optional = []
    
    # Check folders
    for folder in REQUIRED_FOLDERS:
        folder_path = smartcash_dir / folder
        if not folder_path.exists():
            if folder in ['data', 'configs']:
                missing_essential.append(f"folder:{folder}")
            else:
                missing_optional.append(f"folder:{folder}")
    
    # Check configs
    config_dir = smartcash_dir / 'configs'
    for config in ESSENTIAL_CONFIGS:
        config_path = config_dir / config
        if not config_path.exists():
            missing_essential.append(f"config:{config}")
    
    return {
        'essential': missing_essential,
        'optional': missing_optional,
        'total_missing': len(missing_essential) + len(missing_optional),
        'setup_needed': len(missing_essential) > 0
    }

def refresh_environment_state_silent() -> bool:
    """🔄 Refresh environment state tanpa verbose logging"""
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

# === ONE-LINER CONVENIENCE FUNCTIONS ===
is_drive_ready = lambda: test_drive_readiness()[0]
get_smartcash_dir = lambda: Path(SMARTCASH_DRIVE_PATH)
get_repo_config_dir = lambda: Path(REPO_CONFIG_PATH)
setup_needed = lambda: get_prioritized_missing_items()['setup_needed']