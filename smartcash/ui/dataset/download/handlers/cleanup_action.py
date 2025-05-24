"""
File: smartcash/ui/dataset/download/handlers/cleanup_action.py
Deskripsi: Fixed cleanup action dengan explicit progress bar updates
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.utils.confirmation_dialog import show_cleanup_confirmation
from smartcash.ui.dataset.download.utils.button_state_manager import get_button_state_manager
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer

def execute_cleanup_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Execute cleanup action dengan proper state management."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    try:
        button_manager.disable_buttons('cleanup')
        
        if logger:
            logger.info("🧹 Memulai cleanup dataset")
        
        _clear_ui_outputs(ui_components)
        
        organizer = DatasetOrganizer(logger=logger)
        dataset_stats = organizer.check_organized_dataset()
        
        if not dataset_stats['is_organized'] or dataset_stats['total_images'] == 0:
            if logger:
                logger.info("ℹ️ Tidak ada dataset untuk dihapus")
            button_manager.enable_buttons('all')
            return
        
        total_files = dataset_stats['total_images'] + dataset_stats['total_labels']
        show_cleanup_confirmation(ui_components, "Dataset Final Structure", total_files)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error cleanup: {str(e)}")
        button_manager.enable_buttons('all')

def execute_cleanup_confirmed(ui_components: Dict[str, Any], output_dir: str = None) -> None:
    """Execute cleanup dengan explicit progress bar updates."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    with button_manager.operation_context('cleanup'):
        try:
            if logger:
                logger.info("🗑️ Menghapus dataset")
            
            _force_show_cleanup_progress(ui_components)
            _update_cleanup_progress_bars(ui_components, 10, "Memulai cleanup...")
            
            organizer = DatasetOrganizer(logger=logger)
            organizer.set_progress_callback(lambda step, curr, total, msg: _cleanup_progress_callback(ui_components, curr, msg))
            
            _update_cleanup_progress_bars(ui_components, 20, "Menghapus file dataset...")
            result = organizer.cleanup_all_dataset_folders()
            
            if result['status'] == 'success':
                _update_cleanup_progress_bars(ui_components, 100, "Cleanup selesai")
                if logger:
                    logger.success(f"✅ {result['message']}")
                    stats = result['stats']
                    if stats['folders_cleaned']:
                        for folder in stats['folders_cleaned']:
                            logger.info(f"   • {folder}")
            elif result['status'] == 'empty':
                _update_cleanup_progress_bars(ui_components, 100, "Tidak ada file untuk dihapus")
                if logger:
                    logger.info(f"ℹ️ {result['message']}")
            else:
                raise Exception(result.get('message', 'Cleanup gagal'))
                
        except Exception as e:
            _update_cleanup_progress_bars(ui_components, 0, f"❌ Error: {str(e)}")
            if logger:
                logger.error(f"❌ Error cleanup: {str(e)}")
            raise

def _force_show_cleanup_progress(ui_components: Dict[str, Any]) -> None:
    """Force show progress bars untuk cleanup operation."""
    try:
        # Show container
        if 'progress_container' in ui_components:
            container = ui_components['progress_container']
            if isinstance(container, dict) and 'show_container' in container:
                container['show_container']()
            elif hasattr(container, 'layout'):
                container.layout.visibility = 'visible'
                container.layout.display = 'block'
        
        # Show overall progress
        for widget_key in ['overall_progress', 'progress_bar']:
            if widget_key in ui_components and ui_components[widget_key]:
                widget = ui_components[widget_key]
                if hasattr(widget, 'layout'):
                    widget.layout.visibility = 'visible'
                    widget.layout.display = 'block'
                    widget.layout.width = '100%'
                    widget.layout.height = '25px'
                if hasattr(widget, 'value'):
                    widget.value = 0
                if hasattr(widget, 'bar_style'):
                    widget.bar_style = 'info'
        
        # Show current progress untuk detail cleanup
        if 'current_progress' in ui_components and ui_components['current_progress']:
            widget = ui_components['current_progress']
            if hasattr(widget, 'layout'):
                widget.layout.visibility = 'visible'
                widget.layout.display = 'block'
                widget.layout.width = '100%'
                widget.layout.height = '15px'
            if hasattr(widget, 'value'):
                widget.value = 0
            if hasattr(widget, 'bar_style'):
                widget.bar_style = 'warning'
        
        # Hide step progress
        for widget_key in ['step_progress', 'step_label']:
            if widget_key in ui_components and ui_components[widget_key]:
                widget = ui_components[widget_key]
                if hasattr(widget, 'layout'):
                    widget.layout.visibility = 'hidden'
                    widget.layout.display = 'none'
                    
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"📊 Error showing cleanup progress: {str(e)}")

def _update_cleanup_progress_bars(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update cleanup progress bars dengan explicit updates."""
    try:
        # Update overall progress widgets
        for widget_key in ['overall_progress', 'progress_bar']:
            if widget_key in ui_components and ui_components[widget_key]:
                widget = ui_components[widget_key]
                widget.value = progress
                widget.layout.visibility = 'visible'
                widget.layout.display = 'block'
                if hasattr(widget, 'bar_style'):
                    if progress >= 100:
                        widget.bar_style = 'success'
                    elif progress > 0:
                        widget.bar_style = 'warning'
        
        # Update label
        if 'overall_label' in ui_components and ui_components['overall_label']:
            ui_components['overall_label'].value = f"<div style='color: #495057; font-weight: bold;'>🧹 {message} ({progress}%)</div>"
            
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"📊 Error updating cleanup progress: {str(e)}")

def _cleanup_progress_callback(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Callback untuk update current progress selama cleanup."""
    try:
        if 'current_progress' in ui_components and ui_components['current_progress']:
            widget = ui_components['current_progress']
            widget.value = progress
            widget.layout.visibility = 'visible'
            widget.layout.display = 'block'
            if hasattr(widget, 'bar_style'):
                widget.bar_style = 'warning'
        
        if 'current_label' in ui_components and ui_components['current_label']:
            ui_components['current_label'].value = f"<div style='color: #868e96; font-size: 12px;'>⚡ {message}</div>"
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