"""
File: smartcash/ui/dataset/download/handlers/check_action.py
Deskripsi: Fixed check action dengan proper button state management dan progress tracking
"""

from typing import Dict, Any
from pathlib import Path
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.environment import get_environment_manager
from smartcash.ui.dataset.download.utils.button_state_manager import get_button_state_manager

def execute_check_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi pengecekan dataset dengan proper state management."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    # Use context manager untuk proper state management
    with button_manager.operation_context('check'):
        try:
            if logger:
                logger.info("🔍 Memeriksa status dataset di struktur final")
            
            _clear_ui_outputs(ui_components)
            
            # Single progress bar untuk check operation
            _update_check_progress(ui_components, 20, "Memeriksa struktur dataset...")
            
            # Check struktur final dataset
            final_stats = _check_final_dataset_structure(ui_components)
            _update_check_progress(ui_components, 60, "Menganalisis hasil...")
            
            # Check downloads folder
            downloads_stats = _check_downloads_folder(ui_components)
            _update_check_progress(ui_components, 80, "Menyelesaikan pengecekan...")
            
            # Display results
            _display_comprehensive_results(ui_components, final_stats, downloads_stats)
            _update_check_progress(ui_components, 100, "Pengecekan selesai")
            
        except Exception as e:
            if logger:
                logger.error(f"❌ Error saat memeriksa dataset: {str(e)}")
            raise

def _update_check_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update progress untuk check operation - hanya overall progress."""
    try:
        # Update overall progress (single progress bar untuk check)
        if 'overall_progress' in ui_components and ui_components['overall_progress']:
            ui_components['overall_progress'].value = progress
            ui_components['overall_progress'].description = f"Progress: {progress}%"
        elif 'progress_bar' in ui_components and ui_components['progress_bar']:
            ui_components['progress_bar'].value = progress
            ui_components['progress_bar'].description = f"Progress: {progress}%"
        
        # Update label
        if 'overall_label' in ui_components and ui_components['overall_label']:
            ui_components['overall_label'].value = f"<div style='color: #495057; font-weight: bold;'>🔍 {message}</div>"
    except Exception:
        pass

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
    
    # Check setiap split
    for split in ['train', 'valid', 'test']:
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

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs sebelum check."""
    try:
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
    except Exception:
        pass