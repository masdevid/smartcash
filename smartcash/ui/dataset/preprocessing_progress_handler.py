"""
File: smartcash/ui/dataset/preprocessing_progress_handler.py
Deskripsi: Handler yang disederhanakan untuk progress tracking preprocessing dataset tanpa threading
"""

from typing import Dict, Any
from IPython.display import display
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.components.alerts import create_status_indicator

def setup_progress_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tracking progress preprocessing tanpa threading.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Fungsi progress callback sederhana
    def progress_callback(progress, total, message=None, status='info', 
                         current_progress=None, current_total=None):
        """Progress callback sederhana tanpa threading."""
        # Skip jika preprocessing sudah dihentikan
        if not ui_components.get('preprocessing_running', True): return
        
        # Update progress bar utama
        if progress is not None and total is not None and 'progress_bar' in ui_components:
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
    
    # Fungsi untuk registrasi callback ke dataset manager
    def register_progress_callback(dataset_manager):
        """Register callback ke dataset manager."""
        if not dataset_manager or not hasattr(dataset_manager, 'register_progress_callback'): 
            return False
        
        # Register callback ke dataset manager
        dataset_manager.register_progress_callback(progress_callback)
        return True
    
    # Coba setup observer jika tersedia
    observer_setup_success = False
    try:
        from smartcash.components.observer.event_topics_observer import EventTopics
        from smartcash.ui.handlers.observer_handler import create_progress_observer
        
        # Setup progress observers
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
        
        create_progress_observer(
            ui_components=ui_components,
            event_type=EventTopics.PREPROCESSING_CURRENT_PROGRESS,
            total=100,
            progress_widget_key='current_progress',
            update_output=False,
            observer_group='preprocessing_observers'
        )
        
        observer_setup_success = True
        if logger: logger.info(f"{ICONS['success']} Progress tracking terintegrasi berhasil setup")
    except (ImportError, AttributeError):
        observer_setup_success = False
    
    # Tambahkan fungsi progress dan register ke UI components
    ui_components.update({
        'progress_callback': progress_callback,
        'register_progress_callback': register_progress_callback
    })
    
    # Registrasi langsung jika dataset manager sudah ada
    if 'dataset_manager' in ui_components:
        register_progress_callback(ui_components['dataset_manager'])
    
    return ui_components