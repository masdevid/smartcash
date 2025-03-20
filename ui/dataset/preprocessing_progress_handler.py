"""
File: smartcash/ui/dataset/preprocessing_progress_handler.py
Deskripsi: Handler progress tracking preprocessing dataset dengan EventTopics yang diperbarui
"""

from typing import Dict, Any
from IPython.display import display
from smartcash.ui.utils.constants import ICONS

def setup_progress_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler progress tracking dengan integrasi observer standar."""
    logger = ui_components.get('logger')
    
    # Fungsi progress callback terkonsolidasi yang lebih efisien
    def progress_callback(progress=None, total=None, message=None, status='info', 
                         current_progress=None, current_total=None, **kwargs):
        """
        Progress callback dengan utilitas UI standar.
        
        Args:
            progress: Nilai progress utama
            total: Total maksimum progress utama
            message: Pesan progress 
            status: Status progress ('info', 'success', 'warning', 'error')
            current_progress: Nilai progress saat ini (substep)
            current_total: Total maksimum progress saat ini
            **kwargs: Parameter lain yang diteruskan dari caller
        """
        # Skip jika preprocessing sudah dihentikan
        if not ui_components.get('preprocessing_running', True): return
        
        # Update progress bar utama dengan validasi
        if progress is not None and total is not None and 'progress_bar' in ui_components:
            ui_components['progress_bar'].max = total
            ui_components['progress_bar'].value = progress
        
        # Update current progress jika tersedia dengan validasi
        if current_progress is not None and current_total is not None and 'current_progress' in ui_components:
            ui_components['current_progress'].max = current_total
            ui_components['current_progress'].value = current_progress
        
        # # Update pesan jika ada dengan utilitas alert standar
        # if message and 'status' in ui_components:
        #     with ui_components['status']:
        #         from smartcash.ui.utils.alert_utils import create_status_indicator
        #         display(create_status_indicator(status, message))
                
        # Notifikasi observer dengan observer standar jika progress signifikan
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            
            if progress is not None and total is not None and progress % 10 == 0:
                notify(
                    event_type=EventTopics.PREPROCESSING_PROGRESS, 
                    sender="preprocessing_handler",
                    message=message or f"Preprocessing progress: {int(progress/total*100)}%",
                    progress=progress,
                    total=total
                )
        except ImportError:
            pass
    
    # Fungsi untuk registrasi callback ke dataset manager dengan validasi
    def register_progress_callback(dataset_manager):
        """Register callback progress ke dataset manager."""
        if not dataset_manager or not hasattr(dataset_manager, 'register_progress_callback'): 
            return False
        
        # Register callback ke dataset manager
        dataset_manager.register_progress_callback(progress_callback)
        return True
    
    # Setup observer integrasi full jika tersedia
    try:
        from smartcash.components.observer.event_topics_observer import EventTopics
        from smartcash.ui.handlers.observer_handler import create_progress_observer
        
        # Setup progress observer untuk progress utama
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
        
        # Setup progress observer untuk current progress
        create_progress_observer(
            ui_components=ui_components,
            event_type=EventTopics.PREPROCESSING_CURRENT_PROGRESS,
            total=100,
            progress_widget_key='current_progress',
            update_output=False,
            observer_group='preprocessing_observers'
        )
        
        if logger: logger.info(f"{ICONS['success']} Progress tracking terintegrasi berhasil setup")
    except (ImportError, AttributeError) as e:
        if logger: logger.warning(f"{ICONS['warning']} Observer progress tidak tersedia: {str(e)}")
    
    # Helper utility untuk memudahkan update progress
    def update_progress_bar(progress, total, message=None):
        """
        Update progress bar dengan satu fungsi.
        
        Args:
            progress: Nilai progress saat ini
            total: Nilai maksimal progress
            message: Pesan opsional
        """
        # Update progress bar
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].max = total
            ui_components['progress_bar'].value = progress
            percentage = int((progress / total) * 100) if total > 0 else 0
            ui_components['progress_bar'].description = f"Progress: {percentage}%"
        
        # Update message jika ada
        if message and 'progress_label' in ui_components:
            ui_components['progress_label'].value = message
            
        # Notify observer
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(
                event_type=EventTopics.PREPROCESSING_PROGRESS, 
                sender="preprocessing_handler",
                message=message or f"Preprocessing progress: {int(progress/total*100)}%",
                progress=progress,
                total=total
            )
        except ImportError:
            pass
    
    # Helper untuk reset progress
    def reset_progress_bar():
        """Reset semua komponen progress ke nilai awal."""
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = 'Overall:'
            
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].description = 'Current:'
            
        if 'progress_label' in ui_components and hasattr(ui_components['progress_label'], 'value'):
            ui_components['progress_label'].value = "Siap untuk preprocessing"
    
    # Tambahkan fungsi progress dan register ke UI components
    ui_components.update({
        'progress_callback': progress_callback,
        'register_progress_callback': register_progress_callback,
        'update_progress_bar': update_progress_bar,
        'reset_progress_bar': reset_progress_bar
    })
    
    # Registrasi langsung jika dataset manager sudah ada
    if 'dataset_manager' in ui_components:
        register_progress_callback(ui_components['dataset_manager'])
    
    return ui_components