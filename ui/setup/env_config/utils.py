"""
File: smartcash/ui/setup/env_config/utils.py  
Deskripsi: Utils untuk environment config dengan SmartProgressTracker integration dan one-liner functions
"""

import time
from typing import Dict, Any, Tuple
from pathlib import Path
from .constants import (
    REQUIRED_FOLDERS, CONFIG_TEMPLATES, ESSENTIAL_CONFIGS,
    PROGRESS_RANGES, DRIVE_MOUNT_POINT, SMARTCASH_DRIVE_PATH, 
    REPO_CONFIG_PATH, STATUS_MESSAGES, PROGRESS_MESSAGES, RETRY_CONFIG
)

# === SMARTPROGRESSTRACKER UTILS ===
def update_progress_safe(ui_components: Dict[str, Any], value: int, message: str = "") -> None:
    """Update progress dengan SmartProgressTracker atau fallback - one-liner safe"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'update_progress'):
            # Determine level berdasarkan range progress
            if value <= 30:
                progress_tracker.update_progress('phase', value * 3, message)  # Scale untuk phase
            elif value <= 90:
                progress_tracker.update_progress('step', (value - 30) * 1.67, message)  # Scale untuk step
            else:
                progress_tracker.update_progress('overall', value, message)
        elif 'progress_bar' in ui_components:
            # Legacy fallback
            progress_bar = ui_components['progress_bar']
            setattr(progress_bar, 'value', value) if hasattr(progress_bar, 'value') else None
            setattr(progress_bar, 'description', f"{value}%") if hasattr(progress_bar, 'description') else None
    except Exception:
        pass

def hide_progress_safe(ui_components: Dict[str, Any]) -> None:
    """Hide progress dengan SmartProgressTracker atau fallback - one-liner"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'hide'):
            progress_tracker.hide()
        elif 'progress_container' in ui_components:
            ui_components['progress_container'].layout.visibility = 'hidden'
    except Exception:
        pass

def show_progress_safe(ui_components: Dict[str, Any]) -> None:
    """Show progress dengan SmartProgressTracker atau fallback - one-liner"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'show'):
            progress_tracker.show()
        elif 'progress_container' in ui_components:
            ui_components['progress_container'].layout.visibility = 'visible'
    except Exception:
        pass

def reset_progress_safe(ui_components: Dict[str, Any], message: str = "") -> None:
    """Reset progress dengan SmartProgressTracker atau fallback - one-liner"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
            message and progress_tracker.update_status(message)
        elif 'progress_bar' in ui_components:
            progress_bar = ui_components['progress_bar']
            setattr(progress_bar, 'value', 0) if hasattr(progress_bar, 'value') else None
            setattr(progress_bar, 'description', "0%") if hasattr(progress_bar, 'description') else None
            show_progress_safe(ui_components)
    except Exception:
        pass

def start_progress_phase(ui_components: Dict[str, Any], phase_name: str = "setup") -> None:
    """Start progress phase dengan SmartProgressTracker - one-liner"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'start'):
            progress_tracker.start(f"🚀 Memulai {phase_name}")
        else:
            update_progress_safe(ui_components, 0, f"🚀 Memulai {phase_name}")
    except Exception:
        pass

def next_progress_phase(ui_components: Dict[str, Any], phase_name: str = None) -> None:
    """Next phase dalam SmartProgressTracker - one-liner"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'next_phase'):
            progress_tracker.next_phase(phase_name)
    except Exception:
        pass

def complete_progress_safe(ui_components: Dict[str, Any], message: str = "Selesai") -> None:
    """Complete progress dengan SmartProgressTracker - one-liner"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'complete'):
            progress_tracker.complete(message)
        else:
            update_progress_safe(ui_components, 100, message)
    except Exception:
        pass

def error_progress_safe(ui_components: Dict[str, Any], message: str = "Error") -> None:
    """Set error state dengan SmartProgressTracker - one-liner"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'error'):
            progress_tracker.error(message)
        else:
            reset_progress_safe(ui_components, message)
    except Exception:
        pass

# === ENVIRONMENT DETECTION ===
def is_colab_environment() -> bool:
    """Check Colab environment dengan one-liner"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_drive_paths() -> Dict[str, str]:
    """Get Drive paths untuk environment - one-liner"""
    return {
        'mount_point': DRIVE_MOUNT_POINT,
        'smartcash_path': SMARTCASH_DRIVE_PATH,
        'repo_config_path': REPO_CONFIG_PATH
    }

# === DRIVE OPERATIONS ===
def test_drive_readiness(drive_path: Path) -> bool:
    """Test Drive readiness dengan write-read test - one-liner pattern"""
    try:
        test_file = drive_path / '.smartcash_ready_test'
        test_content = f'ready_{int(time.time())}'
        test_file.write_text(test_content)
        content = test_file.read_text()
        test_file.unlink()
        return content == test_content
    except Exception:
        return False

def check_drive_mount_comprehensive() -> Dict[str, Any]:
    """Comprehensive Drive mount check dengan one-liner evaluation"""
    if not is_colab_environment():
        return {'mounted': True, 'path': None, 'type': 'local', 'ready': True}
    
    drive_mount_point = Path(DRIVE_MOUNT_POINT)
    smartcash_drive_path = Path(SMARTCASH_DRIVE_PATH)
    
    if not drive_mount_point.exists():
        return {'mounted': False, 'path': None, 'type': 'colab', 'ready': False}
    
    ready = test_drive_readiness(drive_mount_point)
    return {
        'mounted': True, 'path': str(smartcash_drive_path),
        'type': 'colab', 'ready': ready
    }

def wait_for_drive_ready(max_wait: int = None, ui_components: Dict[str, Any] = None) -> Tuple[bool, str]:
    """Wait untuk Drive ready dengan SmartProgressTracker updates - one-liner loop"""
    max_wait = max_wait or RETRY_CONFIG['drive_mount_timeout']
    
    for i in range(max_wait):
        if test_drive_readiness(Path(DRIVE_MOUNT_POINT)):
            if ui_components:
                update_progress_safe(ui_components, 25, "✅ Drive connected successfully")
            return True, STATUS_MESSAGES['drive_ready']
        
        if i % 5 == 0 and i > 0 and ui_components:
            progress = 15 + (i/max_wait) * 10
            update_progress_safe(ui_components, int(progress), f"⏳ Validating Drive state... ({i}s)")
        
        time.sleep(1)
    
    return (test_drive_readiness(Path(DRIVE_MOUNT_POINT)), 
            STATUS_MESSAGES['drive_ready'] if test_drive_readiness(Path(DRIVE_MOUNT_POINT)) 
            else f"Timeout setelah {max_wait}s - Drive tidak dapat divalidasi")

# === FOLDER & CONFIG OPERATIONS ===
def create_folder_with_retry(folder_path: Path, attempts: int = 3) -> bool:
    """Create folder dengan retry mechanism - one-liner"""
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
    """Validate Drive folders dengan one-liner"""
    drive_base = Path(drive_base_path)
    return {folder: (folder_path := drive_base / folder).exists() and folder_path.is_dir() 
            for folder in REQUIRED_FOLDERS}

def validate_configs(drive_base_path: str) -> Dict[str, bool]:
    """Validate Drive configs dengan one-liner"""
    drive_config_path = Path(drive_base_path) / 'configs'
    return {config: (config_file := drive_config_path / config).exists() and config_file.is_file() 
            for config in CONFIG_TEMPLATES}

def validate_repo_configs() -> Dict[str, bool]:
    """Validate repo configs dengan one-liner"""
    repo_config_path = Path(REPO_CONFIG_PATH)
    return {config: (repo_config_path / config).exists() for config in CONFIG_TEMPLATES}

# === SYMLINK OPERATIONS ===
def check_symlink_status(local_path: Path) -> Dict[str, Any]:
    """Check single symlink status - one-liner pattern"""
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
    """Validate all symlinks dengan one-liner"""
    if not is_colab_environment():
        return {folder: {'exists': True, 'is_symlink': False, 'valid': True} 
                for folder in REQUIRED_FOLDERS}
    
    return {folder: check_symlink_status(Path(f'/content/{folder}')) 
            for folder in REQUIRED_FOLDERS}

# === STATUS EVALUATION ===
def evaluate_readiness(drive_status: Dict[str, Any], repo_configs: Dict[str, bool], 
                      drive_folders: Dict[str, bool], drive_configs: Dict[str, bool], 
                      symlinks: Dict[str, Dict[str, Any]]) -> bool:
    """Evaluate overall readiness dengan one-liner logic"""
    return (drive_status['mounted'] and drive_status.get('ready', False) and
            all(repo_configs.values()) and all(drive_folders.values()) and
            sum(drive_configs.values()) >= len(ESSENTIAL_CONFIGS) and
            all(link['valid'] for link in symlinks.values()))

def get_missing_items(repo_configs: Dict[str, bool], drive_folders: Dict[str, bool], 
                     drive_configs: Dict[str, bool], symlinks: Dict[str, Dict[str, Any]]) -> Dict[str, list]:
    """Get missing items untuk diagnostics - one-liner"""
    return {
        'missing_repo_configs': [k for k, v in repo_configs.items() if not v],
        'missing_drive_folders': [k for k, v in drive_folders.items() if not v],
        'missing_drive_configs': [k for k, v in drive_configs.items() if not v],
        'invalid_symlinks': [k for k, v in symlinks.items() if not v['valid']]
    }

def get_prioritized_missing_items(env_status: Dict[str, Any]) -> list:
    """Get prioritized missing items untuk display - one-liner pattern"""
    missing_items = []
    
    if not env_status.get('drive', {}).get('mounted', False):
        missing_items.append("Google Drive")
    
    missing_folders = env_status.get('missing_drive_folders', [])[:2]
    missing_items.extend([f"folder {folder}" for folder in missing_folders])
    
    missing_configs = env_status.get('missing_drive_configs', [])[:2]  
    missing_items.extend([f"config {config.replace('_config.yaml', '')}" for config in missing_configs])
    
    return missing_items

# === SETUP VALIDATION ===
def validate_setup_integrity(drive_base_path: str) -> bool:
    """Quick validation untuk setup integrity - one-liner pattern"""
    try:
        drive_base = Path(drive_base_path)
        
        # Critical folders check
        critical_folders_valid = all((drive_base / folder).exists() and (drive_base / folder).is_dir() 
                                   for folder in ['data', 'configs'])
        
        # Critical symlinks check
        critical_symlinks_valid = all(Path(f'/content/{folder}').exists() and Path(f'/content/{folder}').is_symlink() 
                                    for folder in ['data', 'configs'])
        
        # Write access test
        test_file = drive_base / 'data' / '.setup_validation_test'
        test_file.write_text('validation')
        content = test_file.read_text()
        test_file.unlink()
        write_access_valid = content == 'validation'
        
        return critical_folders_valid and critical_symlinks_valid and write_access_valid
        
    except Exception:
        return False

# === ENVIRONMENT MANAGER INTEGRATION ===
def refresh_environment_state_silent(env_manager) -> None:
    """Refresh environment state secara silent - one-liner"""
    try:
        env_manager.refresh_drive_status() if env_manager else None
        time.sleep(0.5)  # Brief settling time
    except Exception:
        pass

def get_system_summary_minimal(env_manager) -> str:
    """Get minimal system summary untuk logging - one-liner"""
    try:
        system_info = env_manager.get_system_info() if env_manager else {}
        summary_parts = [
            system_info.get('environment', 'Unknown'),
            "GPU✅" if system_info.get('cuda_available') else "CPU",
            f"{system_info['available_memory_gb']:.1f}GB" if 'available_memory_gb' in system_info else ""
        ]
        return ' | '.join([part for part in summary_parts if part])
    except Exception:
        return "System info unavailable"

# === CONSTANTS ACCESS ===
get_required_folders = lambda: REQUIRED_FOLDERS.copy()
get_config_templates = lambda: CONFIG_TEMPLATES.copy()
get_essential_configs = lambda: ESSENTIAL_CONFIGS.copy()
get_progress_range = lambda operation: PROGRESS_RANGES.get(operation, (0, 100))
get_status_message = lambda key: STATUS_MESSAGES.get(key, "")
get_progress_message = lambda key: PROGRESS_MESSAGES.get(key, "")
get_retry_config = lambda key: RETRY_CONFIG.get(key, 3)