"""
File: smartcash/ui/dataset/preprocessing_progress_handler.py
Deskripsi: Handler untuk progress tracking preprocessing dataset yang disederhanakan
"""

from typing import Dict, Any, Optional
from IPython.display import display
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.components.alerts import create_status_indicator

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
    logger = ui_components.get('logger')
    
    # Coba setup observer jika tersedia
    observer_setup_success = False
    try:
        from smartcash.components.observer.event_topics_observer import EventTopics
        from smartcash.ui.handlers.observer_handler import create_progress_observer
        
        # Progress bar utama
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
        
        # Progress current task
        create_progress_observer(
            ui_components=ui_components,
            event_type=EventTopics.PREPROCESSING_CURRENT_PROGRESS,
            total=100,
            progress_widget_key='current_progress',
            update_output=False,
            observer_group='preprocessing_observers'
        )
        
        observer_setup_success = True
        
        if logger:
            logger.info(f"{ICONS['success']} Progress tracking terintegrasi berhasil setup")
            
    except (ImportError, AttributeError):
        observer_setup_success = False
    
    # Jika observer gagal, gunakan manual progress tracking yang sederhana
    if not observer_setup_success:
        # Define simple progress functions
        def update_progress(progress, total, message=None):
            """Update progress bar utama."""
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].max = total
                ui_components['progress_bar'].value = progress
                
                # Update pesan jika ada
                if message and 'status' in ui_components:
                    with ui_components['status']:
                        display(create_status_indicator("info", message))
        
        def update_current_progress(progress, total):
            """Update progress task saat ini."""
            if 'current_progress' in ui_components:
                ui_components['current_progress'].max = total
                ui_components['current_progress'].value = progress
        
        # Register ke UI components
        ui_components['update_progress'] = update_progress
        ui_components['update_current_progress'] = update_current_progress
        
        if logger:
            logger.info(f"{ICONS['info']} Menggunakan manual progress tracking sederhana")
    
    # Registrasi callback ke dataset manager
    def register_progress_callback(dataset_manager):
        """Register callback ke dataset manager."""
        if not dataset_manager or not hasattr(dataset_manager, 'register_progress_callback'):
            return False
            
        try:
            def progress_callback(progress, total, message=None, status='info', 
                                 current_progress=None, current_total=None):
                """Progress callback dari dataset manager."""
                # Skip jika preprocessing sudah dihentikan
                if not ui_components.get('preprocessing_running', True):
                    return
                    
                # Update progress bar utama
                if 'progress_bar' in ui_components:
                    ui_components['progress_bar'].max = total
                    ui_components['progress_bar'].value = progress
                
                # Update current progress jika tersedia
                if current_progress is not None and current_total is not None and 'current_progress' in ui_components:
                    ui_components['current_progress'].max = current_total
                    ui_components['current_progress'].value = current_progress
                
                # Update pesan jika ada
                if message and 'status' in ui_components:
                    with ui_components['status']:
                        display(create_status_indicator(status, message))
            
            # Register callback
            dataset_manager.register_progress_callback(progress_callback)
            return True
            
        except Exception as e:
            if logger:
                logger.warning(f"{ICONS['warning']} Gagal register progress callback: {str(e)}")
            return False
    
    # Tambahkan register_progress_callback ke UI components
    ui_components['register_progress_callback'] = register_progress_callback
    
    # Register callback jika dataset_manager sudah tersedia
    if 'dataset_manager' in ui_components:
        register_progress_callback(ui_components['dataset_manager'])
    
    return ui_components