
"""
File: smartcash/ui/dataset/download/handlers/check_action.py
Deskripsi: Updated check action dengan observer progress
"""

from pathlib import Path
from smartcash.ui.dataset.download.utils.button_state import disable_download_buttons
from smartcash.components.observer import notify

def execute_check_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi pengecekan dataset dengan observer progress."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🔍 Memeriksa status dataset")
    
    disable_download_buttons(ui_components, True)
    
    try:
        # Reset log output
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        
        # Start via observer
        notify('DOWNLOAD_START', ui_components,
               message="Memeriksa dataset...", namespace="check")
        
        # Dapatkan output directory
        output_dir = ui_components.get('output_dir', {}).value or 'data'
        
        notify('DOWNLOAD_PROGRESS', ui_components,
               progress=20, message="Memeriksa direktori...", namespace="check")
        
        # Check dataset
        stats = _check_dataset_status(ui_components, output_dir)
        
        notify('DOWNLOAD_PROGRESS', ui_components,
               progress=80, message="Menganalisis hasil...", namespace="check")
        
        # Display results
        _display_check_results(ui_components, stats)
        
        notify('DOWNLOAD_COMPLETE', ui_components,
               message="Pengecekan selesai", namespace="check")
        
    except Exception as e:
        notify('DOWNLOAD_ERROR', ui_components,
               message=f"Error pengecekan: {str(e)}", namespace="check")
        if logger:
            logger.error(f"❌ Error saat memeriksa dataset: {str(e)}")
    finally:
        disable_download_buttons(ui_components, False)

def _check_dataset_status(ui_components: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Periksa status dataset sederhana."""
    base_dir = Path(output_dir)
    
    stats = {
        'total_images': 0,
        'splits': {'train': 0, 'valid': 0, 'test': 0},
        'valid': False
    }
    
    if not base_dir.exists():
        return stats
    
    # Check splits dengan progress updates
    for i, split in enumerate(['train', 'valid', 'test']):
        progress = 30 + (i * 15)
        notify('DOWNLOAD_PROGRESS', ui_components,
               progress=progress, message=f"Memeriksa {split}...", namespace="check")
        
        split_dir = base_dir / split / 'images'
        if split_dir.exists():
            img_count = len(list(split_dir.glob('*.*')))
            stats['splits'][split] = img_count
            stats['total_images'] += img_count
    
    stats['valid'] = stats['total_images'] > 0
    return stats

def _display_check_results(ui_components: Dict[str, Any], stats: Dict[str, Any]) -> None:
    """Tampilkan hasil pengecekan."""
    logger = ui_components.get('logger')
    
    if stats['valid']:
        message = f"✅ Dataset ditemukan: {stats['total_images']} gambar"
        if logger:
            logger.success(message)
            for split, count in stats['splits'].items():
                if count > 0:
                    logger.info(f"  • {split}: {count} gambar")
    else:
        message = "⚠️ Dataset tidak ditemukan atau kosong"
        if logger:
            logger.warning(message)