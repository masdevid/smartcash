"""
File: smartcash/ui/dataset/preprocessing_progress_handler.py
Deskripsi: Handler untuk progress tracking preprocessing dataset
"""

from typing import Dict, Any, Optional, Union, Callable
from IPython.display import display

def setup_progress_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tracking progress preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Coba setup observer jika tersedia
    try:
        from smartcash.ui.handlers.observer_handler import create_progress_observer
        from smartcash.components.observer.event_topics_observer import EventTopics
        
        # Setup observer untuk progress bars
        create_progress_observer(
            ui_components=ui_components,
            event_type=[
                EventTopics.PREPROCESSING_PROGRESS,
                EventTopics.PREPROCESSING_START,
                EventTopics.PREPROCESSING_END,
                EventTopics.PREPROCESSING_ERROR
            ],
            total=100,
            progress_widget_key='progress_bar',
            output_widget_key='status',
            observer_group='preprocessing_observers'
        )
        
        # Setup observer untuk current progress
        create_progress_observer(
            ui_components=ui_components,
            event_type=EventTopics.PREPROCESSING_CURRENT_PROGRESS,
            total=100,
            progress_widget_key='current_progress',
            update_output=False,
            observer_group='preprocessing_observers'
        )
        
        if 'logger' in ui_components:
            ui_components['logger'].info("✅ Observers untuk progress tracking berhasil disetup")
    except (ImportError, AttributeError) as e:
        # Fallback to manual progress update
        if 'logger' in ui_components:
            ui_components['logger'].warning(f"⚠️ Tidak dapat setup observers: {str(e)}")
            ui_components['logger'].info("ℹ️ Menggunakan manual progress tracking")
        
        # Create manual progress updater
        def update_progress(progress: int, total: int, message: Optional[str] = None) -> None:
            """Update progress bar."""
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].max = total
                ui_components['progress_bar'].value = progress
            
            if message and 'status' in ui_components:
                from smartcash.ui.components.alerts import create_status_indicator
                with ui_components['status']:
                    display(create_status_indicator("info", message))
        
        def update_current_progress(progress: int, total: int) -> None:
            """Update current progress bar."""
            if 'current_progress' in ui_components:
                ui_components['current_progress'].max = total
                ui_components['current_progress'].value = progress
        
        # Add progress updaters to ui_components
        ui_components['update_progress'] = update_progress
        ui_components['update_current_progress'] = update_current_progress
    
    # Add callback registration function
    def register_progress_callback(dataset_manager: Any) -> None:
        """
        Register callback untuk progress tracking ke dataset manager.
        
        Args:
            dataset_manager: Dataset manager
        """
        if not dataset_manager:
            return
            
        # Check if dataset manager has register_progress_callback method
        if hasattr(dataset_manager, 'register_progress_callback'):
            try:
                # Create progress callback
                def progress_callback(progress: int, total: int, message: Optional[str] = None, 
                                     status: str = 'info', current_progress: Optional[int] = None, 
                                     current_total: Optional[int] = None):
                    """
                    Callback untuk progress update dari dataset manager.
                    
                    Args:
                        progress: Nilai progress overall
                        total: Nilai total overall
                        message: Pesan progress (opsional)
                        status: Status message (info, success, warning, error)
                        current_progress: Nilai progress current task (opsional)
                        current_total: Nilai total current task (opsional)
                    """
                    # Check if preprocessing is still running
                    if not ui_components.get('preprocessing_running', True):
                        return
                        
                    # Update overall progress
                    if 'progress_bar' in ui_components:
                        ui_components['progress_bar'].max = total
                        ui_components['progress_bar'].value = progress
                    
                    # Update current task progress
                    if current_progress is not None and current_total is not None and 'current_progress' in ui_components:
                        ui_components['current_progress'].max = current_total
                        ui_components['current_progress'].value = current_progress
                    
                    # Update status message
                    if message and 'status' in ui_components:
                        from smartcash.ui.components.alerts import create_status_indicator
                        with ui_components['status']:
                            display(create_status_indicator(status, message))
                
                # Register callback to dataset manager
                dataset_manager.register_progress_callback(progress_callback)
                
                if 'logger' in ui_components:
                    ui_components['logger'].info("✅ Progress callback berhasil diregistrasi")
            except Exception as e:
                if 'logger' in ui_components:
                    ui_components['logger'].warning(f"⚠️ Gagal register progress callback: {str(e)}")
    
    # Add register_progress_callback to ui_components
    ui_components['register_progress_callback'] = register_progress_callback
    
    # Try to register callback if dataset_manager already available
    dataset_manager = ui_components.get('dataset_manager')
    if dataset_manager:
        register_progress_callback(dataset_manager)
    
    return ui_components