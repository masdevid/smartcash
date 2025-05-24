"""
File: smartcash/ui/dataset/download/handlers/check_action.py
Deskripsi: Check action dengan tqdm progress tracking
"""

from typing import Dict, Any
from pathlib import Path
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.environment import get_environment_manager
from smartcash.ui.dataset.download.utils.button_state_manager import get_button_state_manager

def execute_check_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Check dataset dengan tqdm progress."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    with button_manager.operation_context('check'):
        try:
            logger and logger.info("🔍 Memeriksa status dataset")
            
            _clear_ui_outputs(ui_components)
            
            _update_check_progress(ui_components, 20, "Memeriksa struktur dataset...")
            final_stats = _check_final_dataset_structure(ui_components)
            
            _update_check_progress(ui_components, 60, "Menganalisis hasil...")
            downloads_stats = _check_downloads_folder(ui_components)
            
            _update_check_progress(ui_components, 80, "Menyelesaikan pengecekan...")
            _display_comprehensive_results(ui_components, final_stats, downloads_stats)
            
            _update_check_progress(ui_components, 100, "Pengecekan selesai")
            
        except Exception as e:
            logger and logger.error(f"❌ Error check: {str(e)}")
            raise

def _update_check_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update check progress dengan tqdm."""
    if 'update_progress' in ui_components:
        ui_components['update_progress']('overall', progress, message)

def _check_final_dataset_structure(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check struktur final dataset."""
    env_manager = get_environment_manager()
    paths = get_paths_for_environment(
        is_colab=env_manager.is_colab,
        is_drive_mounted=env_manager.is_drive_mounted
    )
    
    final_stats = {
        'total_images': 0, 'total_labels': 0,
        'splits': {'train': {}, 'valid': {}, 'test': {}},
        'valid': False, 'base_dir': paths['data_root'],
        'storage_type': 'Drive' if env_manager.is_drive_mounted else 'Local'
    }
    
    for split in ['train', 'valid', 'test']:
        split_path = Path(paths[split])
        split_info = {
            'exists': False, 'images': 0, 'labels': 0, 'path': str(split_path),
            'images_path': str(split_path / 'images'), 'labels_path': str(split_path / 'labels')
        }
        
        if split_path.exists():
            images_dir, labels_dir = split_path / 'images', split_path / 'labels'
            
            try:
                if images_dir.exists():
                    split_info['images'] = len(list(images_dir.glob('*.*')))
                    split_info['exists'] = True
                if labels_dir.exists():
                    split_info['labels'] = len(list(labels_dir.glob('*.txt')))
            except Exception:
                pass
        
        final_stats['splits'][split] = split_info
        final_stats['total_images'] += split_info['images']
        final_stats['total_labels'] += split_info['labels']
    
    final_stats['valid'] = final_stats['total_images'] > 0
    return final_stats

def _check_downloads_folder(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check downloads folder."""
    env_manager = get_environment_manager()
    paths = get_paths_for_environment(
        is_colab=env_manager.is_colab,
        is_drive_mounted=env_manager.is_drive_mounted
    )
    
    downloads_stats = {'exists': False, 'total_files': 0, 'path': paths['downloads']}
    downloads_path = Path(paths['downloads'])
    
    if downloads_path.exists():
        try:
            files = list(downloads_path.rglob('*.*'))
            downloads_stats['total_files'] = len(files)
            downloads_stats['exists'] = len(files) > 0
        except Exception:
            pass
    
    return downloads_stats

def _display_comprehensive_results(ui_components: Dict[str, Any], 
                                 final_stats: Dict[str, Any], 
                                 downloads_stats: Dict[str, Any]) -> None:
    """Display hasil pengecekan."""
    logger = ui_components.get('logger')
    if not logger:
        return
    
    storage_info = f"📁 Storage: {final_stats['storage_type']}"
    if final_stats['storage_type'] == 'Drive':
        env_manager = get_environment_manager()
        storage_info += f" ({env_manager.drive_path})"
    
    logger.info(f"🔍 Hasil Pengecekan Dataset - {storage_info}")
    
    if final_stats['valid']:
        logger.success(f"✅ Dataset ditemukan: {final_stats['total_images']} gambar")
        logger.info(f"📊 Base directory: {final_stats['base_dir']}")
        
        for split, split_info in final_stats['splits'].items():
            if split_info['exists'] and split_info['images'] > 0:
                logger.info(f"   📁 {split}: {split_info['images']} gambar, {split_info['labels']} label")
        
        logger.success("🎉 Dataset siap untuk training!")
    else:
        logger.warning(f"⚠️ Dataset tidak ditemukan di: {final_stats['base_dir']}")
        
        for split, split_info in final_stats['splits'].items():
            if Path(split_info['path']).exists():
                msg = f"   📁 {split}: " + ("folder kosong" if split_info['images'] == 0 else f"{split_info['images']} gambar")
                logger.info(msg)
            else:
                logger.info(f"   📁 {split}: tidak ada")
    
    if downloads_stats['exists']:
        logger.info(f"📥 Downloads: {downloads_stats['total_files']} file")
    else:
        logger.info("📥 Downloads: kosong")
    
    if not final_stats['valid']:
        logger.info("💡 Gunakan 'Download Dataset' untuk mengunduh dataset")

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs."""
    try:
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
    except Exception:
        pass