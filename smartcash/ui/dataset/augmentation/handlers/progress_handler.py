"""
File: smartcash/ui/dataset/augmentation/handlers/progress_handler.py
Deskripsi: Handler untuk progress tracking augmentasi dataset
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS, COLORS

logger = get_logger("augmentation_progress")

def create_progress_callback(ui_components: Dict[str, Any]) -> callable:
    """
    Buat callback untuk progress tracking augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Fungsi callback untuk progress tracking
    """
    def progress_callback(
        progress: Optional[int] = None, 
        total: Optional[int] = None, 
        message: Optional[str] = None, 
        status: str = 'info', 
        step: Optional[int] = None, 
        **kwargs
    ) -> None:
        """
        Callback untuk progress tracking augmentasi.
        
        Args:
            progress: Nilai progress saat ini
            total: Total nilai progress
            message: Pesan progress
            status: Status progress (info, success, warning, error)
            step: Langkah progress saat ini
            **kwargs: Parameter tambahan
        """
        try:
            # Update progress bar jika tersedia
            if progress is not None and total is not None and total > 0:
                if 'progress_bar' in ui_components:
                    ui_components['progress_bar'].max = total
                    ui_components['progress_bar'].value = min(progress, total)
                
                if 'overall_label' in ui_components:
                    percent = int(100 * progress / total)
                    ui_components['overall_label'].value = f"Total Progress: {percent}% ({progress}/{total})"
            
            # Update current progress jika tersedia
            current_progress = kwargs.get('current_progress')
            current_total = kwargs.get('current_total')
            
            if current_progress is not None and current_total is not None and current_total > 0:
                if 'current_progress' in ui_components:
                    ui_components['current_progress'].max = current_total
                    ui_components['current_progress'].value = min(current_progress, current_total)
            
            # Update step label jika tersedia
            if 'step_label' in ui_components and message:
                # Tambahkan emoji berdasarkan status
                emoji = ICONS.get(status, ICONS['info'])
                
                # Format pesan dengan warna berdasarkan status
                color = COLORS.get(f'alert_{status}_text', COLORS['dark'])
                
                # Tambahkan informasi step jika tersedia
                step_info = f"Step {step}: " if step is not None else ""
                
                # Tambahkan informasi kelas jika tersedia
                class_id = kwargs.get('class_id')
                class_info = f" (Kelas: {class_id})" if class_id else ""
                
                # Update label
                ui_components['step_label'].value = f"<span style='color:{color}'>{emoji} {step_info}{message}{class_info}</span>"
            
            # Log pesan jika tersedia
            if message and 'logger' in ui_components:
                ui_components['logger'].info(f"{emoji} {message}")
            
            # Update status panel jika tersedia
            split_step = kwargs.get('split_step')
            if message and status and 'status' in ui_components and split_step:
                with ui_components['status']:
                    clear_output(wait=True)
                    display(create_status_message(status, f"{split_step}: {message}"))
            
        except Exception as e:
            logger.warning(f"⚠️ Error pada progress callback: {str(e)}")
    
    return progress_callback

def register_progress_callback(ui_components: Dict[str, Any], augmentation_service: Any) -> None:
    """
    Register callback untuk progress tracking pada augmentation service.
    
    Args:
        ui_components: Dictionary komponen UI
        augmentation_service: Instance AugmentationService
    """
    if not augmentation_service:
        return
    
    try:
        # Buat callback
        callback = create_progress_callback(ui_components)
        
        # Register callback
        augmentation_service.register_progress_callback(callback)
        
        # Simpan referensi callback
        ui_components['progress_callback'] = callback
    except Exception as e:
        logger.warning(f"⚠️ Error saat register progress callback: {str(e)}")

def reset_progress_tracking(ui_components: Dict[str, Any]) -> None:
    """
    Reset progress tracking UI.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Reset progress bar
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].layout.visibility = 'hidden'
        
        # Reset current progress
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].layout.visibility = 'hidden'
        
        # Reset labels
        if 'overall_label' in ui_components:
            ui_components['overall_label'].value = "Total Progress: 0% (0/0)"
            ui_components['overall_label'].layout.visibility = 'hidden'
        
        if 'step_label' in ui_components:
            ui_components['step_label'].value = ""
            ui_components['step_label'].layout.visibility = 'hidden'
    except Exception as e:
        logger.warning(f"⚠️ Error saat reset progress tracking: {str(e)}")

def create_status_message(status: str, message: str) -> widgets.HTML:
    """
    Buat pesan status dengan format yang sesuai.
    
    Args:
        status: Status pesan (info, success, warning, error)
        message: Isi pesan
        
    Returns:
        Widget HTML dengan pesan status
    """
    from smartcash.ui.utils.alert_utils import create_status_indicator
    return create_status_indicator(status, message)
