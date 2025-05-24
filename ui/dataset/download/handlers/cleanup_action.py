"""
File: smartcash/ui/dataset/download/handlers/cleanup_action.py
Deskripsi: Fixed cleanup action dengan progress tracking yang terintegrasi dengan dual progress system
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.utils.confirmation_dialog import show_cleanup_confirmation
from smartcash.ui.dataset.download.utils.button_state import disable_download_buttons
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer

def execute_cleanup_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi cleanup dataset dengan progress tracking yang benar."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🧹 Memulai cleanup dataset")
    
    disable_download_buttons(ui_components, True)
    
    try:
        _clear_ui_outputs(ui_components)
        
        organizer = DatasetOrganizer(logger=logger)
        dataset_stats = organizer.check_organized_dataset()
        
        if not dataset_stats['is_organized'] or dataset_stats['total_images'] == 0:
            if logger:
                logger.info("ℹ️ Tidak ada dataset yang perlu dihapus")
            disable_download_buttons(ui_components, False)
            return
        
        total_files = dataset_stats['total_images'] + dataset_stats['total_labels']
        show_cleanup_confirmation(ui_components, "Dataset Final Structure", total_files)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error persiapan cleanup: {str(e)}")
        disable_download_buttons(ui_components, False)

def execute_cleanup_confirmed(ui_components: Dict[str, Any], output_dir: str = None) -> None:
    """Eksekusi cleanup dengan dual progress tracking yang benar."""
    logger = ui_components.get('logger')
    
    try:
        # Initialize progress tracking
        from smartcash.ui.dataset.download.handlers.progress_handlers import start_progress, update_step_progress, complete_progress, error_progress
        
        start_progress(ui_components, "Memulai cleanup dataset")
        
        organizer = DatasetOrganizer(logger=logger)
        organizer.set_progress_callback(lambda step, curr, total, msg: _cleanup_progress_callback(ui_components, curr, msg))
        
        if logger:
            logger.info("🗑️ Menghapus dataset dan downloads")
        
        result = organizer.cleanup_all_dataset_folders()
        
        if result['status'] == 'success':
            complete_progress(ui_components, result['message'])
            
            if logger:
                logger.success(f"✅ {result['message']}")
                stats = result['stats']
                if stats['folders_cleaned']:
                    for folder in stats['folders_cleaned']:
                        logger.info(f"   • {folder}")
        elif result['status'] == 'empty':
            complete_progress(ui_components, result['message'])
            if logger:
                logger.info(f"ℹ️ {result['message']}")
        else:
            error_progress(ui_components, result.get('message', 'Cleanup gagal'))
            if logger:
                logger.error(f"❌ {result.get('message', 'Cleanup gagal')}")
        
    except Exception as e:
        error_progress(ui_components, f"Cleanup error: {str(e)}")
        if logger:
            logger.error(f"❌ Error cleanup: {str(e)}")
    finally:
        disable_download_buttons(ui_components, False)

def _cleanup_progress_callback(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Callback untuk update step progress saja selama cleanup."""
    from smartcash.ui.dataset.download.handlers.progress_handlers import update_step_progress
    
    # Update step progress dengan pesan cleanup
    update_step_progress(ui_components, progress, message, "Cleanup")

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs sebelum cleanup."""
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        ui_components['log_output'].clear_output(wait=True)
    
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()