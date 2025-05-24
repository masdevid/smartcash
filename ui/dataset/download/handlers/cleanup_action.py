"""
File: smartcash/ui/dataset/download/handlers/cleanup_action.py
Deskripsi: Cleanup action dengan tqdm progress tracking
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.utils.confirmation_dialog import show_cleanup_confirmation
from smartcash.ui.dataset.download.utils.button_state_manager import get_button_state_manager
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer

def execute_cleanup_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Execute cleanup dengan tqdm progress."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    try:
        button_manager.disable_buttons('cleanup')
        logger and logger.info("🧹 Memulai cleanup dataset")
        _clear_ui_outputs(ui_components)
        
        organizer = DatasetOrganizer(logger=logger)
        dataset_stats = organizer.check_organized_dataset()
        
        if not dataset_stats['is_organized'] or dataset_stats['total_images'] == 0:
            logger and logger.info("ℹ️ Tidak ada dataset untuk dihapus")
            button_manager.enable_buttons('all')
            return
        
        total_files = dataset_stats['total_images'] + dataset_stats['total_labels']
        show_cleanup_confirmation(ui_components, "Dataset Final Structure", total_files)
        
    except Exception as e:
        logger and logger.error(f"❌ Error cleanup: {str(e)}")
        button_manager.enable_buttons('all')

def execute_cleanup_confirmed(ui_components: Dict[str, Any], output_dir: str = None) -> None:
    """Execute cleanup dengan tqdm progress tracking."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    with button_manager.operation_context('cleanup'):
        try:
            logger and logger.info("🗑️ Menghapus dataset")
            
            _update_cleanup_progress(ui_components, 10, "Memulai cleanup...")
            
            organizer = DatasetOrganizer(logger=logger)
            organizer.set_progress_callback(lambda step, curr, total, msg: _cleanup_progress_callback(ui_components, curr, total, msg))
            
            _update_cleanup_progress(ui_components, 20, "Menghapus file dataset...")
            result = organizer.cleanup_all_dataset_folders()
            
            if result['status'] == 'success':
                if logger:
                    logger.success(f"✅ {result['message']}")
                    stats = result['stats']
                    if stats['folders_cleaned']:
                        for folder in stats['folders_cleaned']:
                            logger.info(f"   • {folder}")
            elif result['status'] == 'empty':
                logger and logger.info(f"ℹ️ {result['message']}")
            else:
                raise Exception(result.get('message', 'Cleanup gagal'))
                
        except Exception as e:
            logger and logger.error(f"❌ Error cleanup: {str(e)}")
            raise

def _update_cleanup_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update cleanup progress dengan tqdm."""
    if 'update_progress' in ui_components:
        ui_components['update_progress']('overall', progress, message)

def _cleanup_progress_callback(ui_components: Dict[str, Any], current: int, total: int, message: str) -> None:
    """Callback untuk current progress cleanup."""
    if 'update_progress' in ui_components:
        percentage = int((current / max(total, 1)) * 100)
        ui_components['update_progress']('current', percentage, message)

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs."""
    try:
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
    except Exception:
        pass