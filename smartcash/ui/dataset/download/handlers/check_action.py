"""
File: smartcash/ui/dataset/download/handlers/check_action.py
Deskripsi: Fixed check action yang memeriksa struktur final /data/{train,valid,test}
"""

from typing import Dict, Any
from pathlib import Path
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.environment import get_environment_manager

def execute_check_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi pengecekan dataset dengan focus pada struktur final."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🔍 Memeriksa status dataset di struktur final")
    
    _disable_buttons(ui_components, True)
    
    try:
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        
        _start_progress(ui_components, "Memeriksa dataset...")
        
        # Check struktur final dataset di /data/
        final_stats = _check_final_dataset_structure(ui_components)
        _update_progress(ui_components, 60, "Menganalisis hasil...")
        
        # Check downloads folder sebagai info tambahan
        downloads_stats = _check_downloads_folder(ui_components)
        _update_progress(ui_components, 80, "Menyelesaikan pengecekan...")
        
        # Display comprehensive results
        _display_comprehensive_results(ui_components, final_stats, downloads_stats)
        _complete_progress(ui_components, "Pengecekan selesai")
        
    except Exception as e:
        _error_progress(ui_components, f"Error pengecekan: {str(e)}")
        if logger:
            logger.error(f"❌ Error saat memeriksa dataset: {str(e)}")
    finally:
        _disable_buttons(ui_components, False)

def _check_final_dataset_structure(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Periksa struktur final dataset di /data/{train,valid,test}."""
    env_manager = get_environment_manager()
    paths = get_paths_for_environment(
        is_colab=env_manager.is_colab,
        is_drive_mounted=env_manager.is_drive_mounted
    )
    
    final_stats = {
        'total_images': 0,
        'total_labels': 0,
        'splits': {'train': {}, 'valid': {}, 'test': {}},
        'valid': False,
        'base_dir': paths['data_root'],
        'storage_type': 'Drive' if env_manager.is_drive_mounted else 'Local'
    }
    
    # Check setiap split dengan progress updates
    for i, split in enumerate(['train', 'valid', 'test']):
        progress = 20 + (i * 15)
        _update_progress(ui_components, progress, f"Memeriksa {split}...")
        
        split_path = Path(paths[split])
        split_info = {
            'exists': False,
            'images': 0,
            'labels': 0,
            'path': str(split_path),
            'images_path': str(split_path / 'images'),
            'labels_path': str(split_path / 'labels')
        }
        
        if split_path.exists():
            images_dir = split_path / 'images'
            labels_dir = split_path / 'labels'
            
            if images_dir.exists():
                try:
                    img_files = list(images_dir.glob('*.*'))
                    split_info['images'] = len(img_files)
                    split_info['exists'] = True
                except Exception:
                    split_info['images'] = 0
            
            if labels_dir.exists():
                try:
                    label_files = list(labels_dir.glob('*.txt'))
                    split_info['labels'] = len(label_files)
                except Exception:
                    split_info['labels'] = 0
        
        final_stats['splits'][split] = split_info
        final_stats['total_images'] += split_info['images']
        final_stats['total_labels'] += split_info['labels']
    
    final_stats['valid'] = final_stats['total_images'] > 0
    return final_stats

def _check_downloads_folder(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check downloads folder sebagai info tambahan."""
    env_manager = get_environment_manager()
    paths = get_paths_for_environment(
        is_colab=env_manager.is_colab,
        is_drive_mounted=env_manager.is_drive_mounted
    )
    
    downloads_stats = {
        'exists': False,
        'total_files': 0,
        'path': paths['downloads']
    }
    
    downloads_path = Path(paths['downloads'])
    if downloads_path.exists():
        try:
            files = list(downloads_path.rglob('*.*'))
            downloads_stats['total_files'] = len(files)
            downloads_stats['exists'] = len(files) > 0
        except Exception:
            downloads_stats['total_files'] = 0
    
    return downloads_stats

def _display_comprehensive_results(ui_components: Dict[str, Any], 
                                 final_stats: Dict[str, Any], 
                                 downloads_stats: Dict[str, Any]) -> None:
    """Tampilkan hasil pengecekan yang comprehensive."""
    logger = ui_components.get('logger')
    
    if not logger:
        return
    
    # Header info
    storage_info = f"📁 Storage: {final_stats['storage_type']}"
    if final_stats['storage_type'] == 'Drive':
        env_manager = get_environment_manager()
        storage_info += f" ({env_manager.drive_path})"
    
    logger.info(f"🔍 Hasil Pengecekan Dataset - {storage_info}")
    
    # Final dataset structure results
    if final_stats['valid']:
        logger.success(f"✅ Dataset ditemukan di struktur final: {final_stats['total_images']} gambar")
        logger.info(f"📊 Base directory: {final_stats['base_dir']}")
        
        # Detail per split
        for split, split_info in final_stats['splits'].items():
            if split_info['exists'] and split_info['images'] > 0:
                logger.info(f"   📁 {split}:")
                logger.info(f"      • Gambar: {split_info['images']} file")
                logger.info(f"      • Label: {split_info['labels']} file")
                logger.info(f"      • Path: {split_info['path']}")
        
        # Dataset ready message
        logger.success("🎉 Dataset siap untuk training!")
        
    else:
        logger.warning(f"⚠️ Dataset tidak ditemukan di struktur final: {final_stats['base_dir']}")
        
        # Check individual splits
        for split, split_info in final_stats['splits'].items():
            if Path(split_info['path']).exists():
                if split_info['images'] == 0:
                    logger.info(f"   📁 {split}: folder ada tapi kosong")
                else:
                    logger.info(f"   📁 {split}: {split_info['images']} gambar")
            else:
                logger.info(f"   📁 {split}: folder tidak ada")
    
    # Downloads folder info
    if downloads_stats['exists']:
        logger.info(f"📥 Downloads folder: {downloads_stats['total_files']} file di {downloads_stats['path']}")
    else:
        logger.info(f"📥 Downloads folder: kosong atau tidak ada")
    
    # Guidance
    if not final_stats['valid']:
        logger.info("💡 Gunakan tombol 'Download Dataset' untuk mengunduh dan mengorganisir dataset")
        if downloads_stats['exists']:
            logger.info("💡 Ada file di downloads folder - mungkin perlu diorganisir ulang")

# Progress helper functions (tidak berubah)
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
        button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
        for key in button_keys:
            if key in ui_components and hasattr(ui_components[key], 'disabled'):
                ui_components[key].disabled = disabled