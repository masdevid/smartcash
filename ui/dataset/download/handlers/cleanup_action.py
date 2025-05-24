"""
File: smartcash/ui/dataset/download/handlers/cleanup_action.py
Deskripsi: Updated cleanup action dengan progress tracking yang match progress_tracking.py
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.utils.confirmation_dialog import show_cleanup_confirmation
from smartcash.ui.dataset.download.utils.button_state import disable_download_buttons
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer

def execute_cleanup_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi cleanup dataset dengan progress tracking yang match progress_tracking.py."""
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
    """Eksekusi cleanup dengan progress tracking yang match progress_tracking.py."""
    logger = ui_components.get('logger')
    
    try:
        # Start progress tracking
        _start_cleanup_progress(ui_components, "Memulai cleanup dataset")
        
        organizer = DatasetOrganizer(logger=logger)
        organizer.set_progress_callback(lambda step, curr, total, msg: _cleanup_progress_callback(ui_components, curr, msg))
        
        if logger:
            logger.info("🗑️ Menghapus dataset dan downloads")
        
        result = organizer.cleanup_all_dataset_folders()
        
        if result['status'] == 'success':
            _complete_cleanup_progress(ui_components, result['message'])
            
            if logger:
                logger.success(f"✅ {result['message']}")
                stats = result['stats']
                if stats['folders_cleaned']:
                    for folder in stats['folders_cleaned']:
                        logger.info(f"   • {folder}")
        elif result['status'] == 'empty':
            _complete_cleanup_progress(ui_components, result['message'])
            if logger:
                logger.info(f"ℹ️ {result['message']}")
        else:
            _error_cleanup_progress(ui_components, result.get('message', 'Cleanup gagal'))
            if logger:
                logger.error(f"❌ {result.get('message', 'Cleanup gagal')}")
        
    except Exception as e:
        _error_cleanup_progress(ui_components, f"Cleanup error: {str(e)}")
        if logger:
            logger.error(f"❌ Error cleanup: {str(e)}")
    finally:
        disable_download_buttons(ui_components, False)

def _cleanup_progress_callback(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Callback untuk update current progress selama cleanup."""
    # Update current progress menggunakan progress_tracking.py functions
    from smartcash.ui.components.progress_tracking import update_current_progress
    update_current_progress(ui_components, progress, 100, message)

def _start_cleanup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Start cleanup progress menggunakan progress_tracking.py."""
    from smartcash.ui.components.progress_tracking import update_overall_progress, update_step_progress
    
    # Show progress container
    if 'progress_container' in ui_components:
        ui_components['progress_container']['show_container']()
    
    # Initialize progress
    update_overall_progress(ui_components, 0, 100, message)
    update_step_progress(ui_components, 1, 1, "Cleanup")

def _complete_cleanup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Complete cleanup progress menggunakan progress_tracking.py."""
    from smartcash.ui.components.progress_tracking import update_overall_progress, update_current_progress
    
    # Set semua progress ke 100%
    update_overall_progress(ui_components, 100, 100, message)
    update_current_progress(ui_components, 100, 100, "Selesai")
    
    # Set success bar style
    if 'overall_progress' in ui_components and ui_components['overall_progress']:
        ui_components['overall_progress'].bar_style = 'success'
    if 'current_progress' in ui_components and ui_components['current_progress']:
        ui_components['current_progress'].bar_style = 'success'

def _error_cleanup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Error cleanup progress menggunakan progress_tracking.py."""
    from smartcash.ui.components.progress_tracking import update_overall_progress, update_current_progress
    
    # Reset progress dan set error messages
    update_overall_progress(ui_components, 0, 100, f"❌ {message}")
    update_current_progress(ui_components, 0, 100, "Error")
    
    # Set danger bar style
    if 'overall_progress' in ui_components and ui_components['overall_progress']:
        ui_components['overall_progress'].bar_style = 'danger'
    if 'current_progress' in ui_components and ui_components['current_progress']:
        ui_components['current_progress'].bar_style = 'danger'

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs sebelum cleanup."""
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        ui_components['log_output'].clear_output(wait=True)
    
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()