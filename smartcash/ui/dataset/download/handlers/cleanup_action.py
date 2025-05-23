"""
File: smartcash/ui/dataset/download/handlers/cleanup_action.py
Deskripsi: Fixed cleanup action dengan dataset organizer yang benar
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.utils.confirmation_dialog import show_cleanup_confirmation
from smartcash.ui.dataset.download.utils.button_state import disable_download_buttons
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer

def execute_cleanup_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi cleanup dataset dengan organizer yang benar."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🧹 Memulai cleanup dataset")
    
    disable_download_buttons(ui_components, True)
    
    try:
        # Clear outputs sebelum mulai
        _clear_ui_outputs(ui_components)
        
        # Create organizer untuk check dan cleanup
        organizer = DatasetOrganizer(logger=logger)
        
        # Check existing dataset
        dataset_stats = organizer.check_organized_dataset()
        
        if not dataset_stats['is_organized'] or dataset_stats['total_images'] == 0:
            if logger:
                logger.info("ℹ️ Tidak ada dataset yang perlu dihapus")
            disable_download_buttons(ui_components, False)
            return
        
        # Show confirmation dengan stats yang benar
        total_files = dataset_stats['total_images'] + dataset_stats['total_labels']
        show_cleanup_confirmation(ui_components, "Dataset Final Structure", total_files)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error persiapan cleanup: {str(e)}")
        disable_download_buttons(ui_components, False)

def execute_cleanup_confirmed(ui_components: Dict[str, Any], output_dir: str = None) -> None:
    """Eksekusi cleanup setelah konfirmasi."""
    logger = ui_components.get('logger')
    
    try:
        # Initialize organizer dengan progress tracking
        organizer = DatasetOrganizer(logger=logger)
        organizer.set_progress_callback(lambda step, curr, total, msg: _update_cleanup_progress(ui_components, curr, msg))
        
        # Initialize progress tracking
        _start_cleanup_progress(ui_components, "Memulai cleanup dataset")
        
        if logger:
            logger.info("🗑️ Menghapus dataset dan downloads")
        
        # Execute cleanup menggunakan organizer
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

def _start_cleanup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Start cleanup progress tracking."""
    # Show progress container
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'block'
        ui_components['progress_container'].layout.visibility = 'visible'
    
    # Reset progress widgets
    _update_progress_widgets(ui_components, 0, message)
    
    # Send observer notification
    try:
        from smartcash.components.observer import notify
        notify('DOWNLOAD_START', ui_components, 
               progress=0, message=message, namespace="cleanup")
    except Exception:
        pass

def _update_cleanup_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update cleanup progress."""
    # Clamp progress
    progress = max(0, min(100, progress))
    
    # Update UI widgets directly
    _update_progress_widgets(ui_components, progress, message)
    
    # Send observer notification
    try:
        from smartcash.components.observer import notify
        notify('DOWNLOAD_PROGRESS', ui_components,
               progress=progress, message=message, namespace="cleanup")
    except Exception:
        pass

def _complete_cleanup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Complete cleanup progress."""
    # Update ke 100%
    _update_progress_widgets(ui_components, 100, message)
    
    # Send observer notification
    try:
        from smartcash.components.observer import notify
        notify('DOWNLOAD_COMPLETE', ui_components,
               progress=100, message=message, namespace="cleanup")
    except Exception:
        pass

def _error_cleanup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Set cleanup progress ke error state."""
    # Reset progress ke 0 untuk indicate error
    _update_progress_widgets(ui_components, 0, f"❌ {message}", error=True)
    
    # Send observer notification
    try:
        from smartcash.components.observer import notify
        notify('DOWNLOAD_ERROR', ui_components,
               progress=0, message=message, namespace="cleanup")
    except Exception:
        pass

def _update_progress_widgets(ui_components: Dict[str, Any], progress: int, message: str, error: bool = False) -> None:
    """Update progress widgets directly untuk immediate feedback."""
    # Update main progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = progress
        description = "Error" if error else f"Progress: {progress}%"
        ui_components['progress_bar'].description = description
        
        if hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].layout.visibility = 'visible'
    
    # Update current progress (step progress) 
    if 'current_progress' in ui_components:
        ui_components['current_progress'].value = progress
        ui_components['current_progress'].description = f"Cleanup: {progress}%"
        
        if hasattr(ui_components['current_progress'], 'layout'):
            ui_components['current_progress'].layout.visibility = 'visible'
    
    # Update labels
    if 'overall_label' in ui_components:
        ui_components['overall_label'].value = message
        if hasattr(ui_components['overall_label'], 'layout'):
            ui_components['overall_label'].layout.visibility = 'visible'
    
    if 'step_label' in ui_components:
        ui_components['step_label'].value = f"Cleanup: {message}"
        if hasattr(ui_components['step_label'], 'layout'):
            ui_components['step_label'].layout.visibility = 'visible'

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs sebelum cleanup."""
    # Clear log output
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        ui_components['log_output'].clear_output(wait=True)
    
    # Clear confirmation area
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()