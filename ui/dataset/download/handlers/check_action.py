"""
File: smartcash/ui/dataset/download/handlers/check_action.py
Deskripsi: Fixed check action dengan progress tracking yang sederhana dan reliable
"""

from typing import Dict, Any
from pathlib import Path

def execute_check_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi pengecekan dataset dengan progress tracking sederhana."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🔍 Memeriksa status dataset")
    
    _disable_buttons(ui_components, True)
    
    try:
        # Reset log output
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        
        # Start progress
        _start_progress(ui_components, "Memeriksa dataset...")
        
        # Dapatkan output directory
        output_dir = _get_output_dir(ui_components)
        _update_progress(ui_components, 20, "Memeriksa direktori...")
        
        # Check dataset
        stats = _check_dataset_status(ui_components, output_dir)
        _update_progress(ui_components, 80, "Menganalisis hasil...")
        
        # Display results
        _display_check_results(ui_components, stats)
        _complete_progress(ui_components, "Pengecekan selesai")
        
    except Exception as e:
        _error_progress(ui_components, f"Error pengecekan: {str(e)}")
        if logger:
            logger.error(f"❌ Error saat memeriksa dataset: {str(e)}")
    finally:
        _disable_buttons(ui_components, False)

def _get_output_dir(ui_components: Dict[str, Any]) -> str:
    """Get output directory dengan fallback."""
    try:
        if 'output_dir' in ui_components and hasattr(ui_components['output_dir'], 'value'):
            output_dir = ui_components['output_dir'].value
            if output_dir:
                return output_dir
    except Exception:
        pass
    
    # Fallback ke default
    env_manager = ui_components.get('env_manager')
    if env_manager and env_manager.is_colab and env_manager.is_drive_mounted:
        return str(env_manager.drive_path / 'downloads')
    return 'data'

def _check_dataset_status(ui_components: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Periksa status dataset dengan progress updates."""
    base_dir = Path(output_dir)
    
    stats = {
        'total_images': 0,
        'splits': {'train': 0, 'valid': 0, 'test': 0},
        'valid': False,
        'base_dir': str(base_dir)
    }
    
    if not base_dir.exists():
        return stats
    
    # Check splits dengan progress updates
    for i, split in enumerate(['train', 'valid', 'test']):
        progress = 30 + (i * 15)
        _update_progress(ui_components, progress, f"Memeriksa {split}...")
        
        split_dir = base_dir / split / 'images'
        if split_dir.exists():
            try:
                img_files = list(split_dir.glob('*.*'))
                img_count = len(img_files)
                stats['splits'][split] = img_count
                stats['total_images'] += img_count
            except Exception:
                stats['splits'][split] = 0
    
    stats['valid'] = stats['total_images'] > 0
    return stats

def _display_check_results(ui_components: Dict[str, Any], stats: Dict[str, Any]) -> None:
    """Tampilkan hasil pengecekan."""
    logger = ui_components.get('logger')
    
    if stats['valid']:
        message = f"✅ Dataset ditemukan: {stats['total_images']} gambar"
        if logger:
            logger.success(message)
            logger.info(f"📁 Lokasi: {stats['base_dir']}")
            for split, count in stats['splits'].items():
                if count > 0:
                    logger.info(f"   • {split}: {count} gambar")
    else:
        message = f"⚠️ Dataset tidak ditemukan di: {stats['base_dir']}"
        if logger:
            logger.warning(message)
            logger.info("💡 Gunakan tombol Download untuk mengunduh dataset")

# Progress helper functions
def _start_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Start progress tracking."""
    try:
        if '_observers' in ui_components and 'progress' in ui_components['_observers']:
            ui_components['_observers']['progress']['start_handler'](message)
    except Exception:
        pass

def _update_progress(ui_components: Dict[str, Any], value: int, message: str) -> None:
    """Update progress."""
    try:
        if '_observers' in ui_components and 'progress' in ui_components['_observers']:
            ui_components['_observers']['progress']['overall_progress'](value, message)
    except Exception:
        pass

def _complete_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Complete progress."""
    try:
        if '_observers' in ui_components and 'progress' in ui_components['_observers']:
            ui_components['_observers']['progress']['complete_handler'](message)
    except Exception:
        pass

def _error_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Error progress."""
    try:
        if '_observers' in ui_components and 'progress' in ui_components['_observers']:
            ui_components['_observers']['progress']['error_handler'](message)
    except Exception:
        pass

def _disable_buttons(ui_components: Dict[str, Any], disabled: bool) -> None:
    """Disable/enable buttons dengan error handling."""
    try:
        from smartcash.ui.dataset.download.utils.button_state import disable_download_buttons
        disable_download_buttons(ui_components, disabled)
    except Exception:
        # Fallback manual
        button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
        for key in button_keys:
            if key in ui_components and hasattr(ui_components[key], 'disabled'):
                ui_components[key].disabled = disabled