"""
File: smartcash/ui/dataset/download/handlers/cleanup_action.py
Deskripsi: Fixed cleanup action dengan proper progress tracking dan button state management
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.utils.confirmation_dialog import show_cleanup_confirmation
from smartcash.ui.dataset.download.utils.button_state_manager import get_button_state_manager
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer

def execute_cleanup_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi cleanup dataset dengan proper state management."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    try:
        # Disable buttons temporarily untuk initial check
        button_manager.disable_buttons('cleanup')
        
        if logger:
            logger.info("🧹 Memulai cleanup dataset")
        
        _clear_ui_outputs(ui_components)
        
        organizer = DatasetOrganizer(logger=logger)
        dataset_stats = organizer.check_organized_dataset()
        
        if not dataset_stats['is_organized'] or dataset_stats['total_images'] == 0:
            if logger:
                logger.info("ℹ️ Tidak ada dataset yang perlu dihapus")
            button_manager.enable_buttons('all')
            return
        
        total_files = dataset_stats['total_images'] + dataset_stats['total_labels']
        show_cleanup_confirmation(ui_components, "Dataset Final Structure", total_files)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error persiapan cleanup: {str(e)}")
        button_manager.enable_buttons('all')

def execute_cleanup_confirmed(ui_components: Dict[str, Any], output_dir: str = None) -> None:
    """Eksekusi cleanup confirmed dengan proper progress tracking."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    # Use context manager untuk cleanup operation
    with button_manager.operation_context('cleanup'):
        try:
            if logger:
                logger.info("🗑️ Menghapus dataset dan downloads")
            
            # Setup progress untuk cleanup (overall + current)
            _update_cleanup_progress_overall(ui_components, 10, "Memulai cleanup dataset")
            
            organizer = DatasetOrganizer(logger=logger)
            organizer.set_progress_callback(lambda step, curr, total, msg: _cleanup_progress_callback(ui_components, curr, msg))
            
            result = organizer.cleanup_all_dataset_folders()
            
            if result['status'] == 'success':
                _update_cleanup_progress_overall(ui_components, 100, result['message'])
                
                if logger:
                    logger.success(f"✅ {result['message']}")
                    stats = result['stats']
                    if stats['folders_cleaned']:
                        for folder in stats['folders_cleaned']:
                            logger.info(f"   • {folder}")
            elif result['status'] == 'empty':
                _update_cleanup_progress_overall(ui_components, 100, result['message'])
                if logger:
                    logger.info(f"ℹ️ {result['message']}")
            else:
                raise Exception(result.get('message', 'Cleanup gagal'))
                
        except Exception as e:
            if logger:
                logger.error(f"❌ Error cleanup: {str(e)}")
            raise

def _cleanup_progress_callback(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Callback untuk update current progress selama cleanup."""
    try:
        # Update current progress bar untuk detail cleanup
        if 'current_progress' in ui_components and ui_components['current_progress']:
            ui_components['current_progress'].value = progress
            ui_components['current_progress'].description = f"Current: {progress}%"
        
        if 'current_label' in ui_components and ui_components['current_label']:
            ui_components['current_label'].value = f"<div style='color: #868e96; font-size: 12px;'>⚡ {message}</div>"
    except Exception:
        pass

def _update_cleanup_progress_overall(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update overall progress untuk cleanup operation."""
    try:
        # Update overall progress
        if 'overall_progress' in ui_components and ui_components['overall_progress']:
            ui_components['overall_progress'].value = progress
            ui_components['overall_progress'].description = f"Overall: {progress}%"
        elif 'progress_bar' in ui_components and ui_components['progress_bar']:
            ui_components['progress_bar'].value = progress
            ui_components['progress_bar'].description = f"Progress: {progress}%"
        
        # Update label
        if 'overall_label' in ui_components and ui_components['overall_label']:
            ui_components['overall_label'].value = f"<div style='color: #495057; font-weight: bold;'>🧹 {message}</div>"
    except Exception:
        pass

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs sebelum cleanup."""
    try:
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
    except Exception:
        pass