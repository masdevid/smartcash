"""
File: smartcash/ui/dataset/download/handlers/download_progress_observer.py
Deskripsi: Observer untuk memantau dan memperbarui progress download di UI
"""

from typing import Dict, Any, Optional
from IPython.display import display

def setup_download_progress_observer(ui_components: Dict[str, Any]) -> None:
    """
    Setup observer untuk monitoring progress download.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    try:
        # Import komponen observer
        from smartcash.components.observer.event_topics_observer import EventTopics
        
        # Observer manager
        observer_manager = ui_components.get('observer_manager')
        if not observer_manager:
            return
        
        # Register observers untuk berbagai event download
        events = [
            # Event download
            EventTopics.DOWNLOAD_START, 
            EventTopics.DOWNLOAD_PROGRESS, 
            EventTopics.DOWNLOAD_COMPLETE,
            EventTopics.DOWNLOAD_ERROR,
            
            # Event pull dataset
            EventTopics.PULL_DATASET_START,
            EventTopics.PULL_DATASET_PROGRESS,
            EventTopics.PULL_DATASET_COMPLETE,
            EventTopics.PULL_DATASET_ERROR,
            
            # Event ZIP processing
            EventTopics.ZIP_PROCESSING_START,
            EventTopics.ZIP_PROCESSING_PROGRESS,
            EventTopics.ZIP_PROCESSING_COMPLETE,
            EventTopics.ZIP_PROCESSING_ERROR
        ]
        
        # PERBAIKAN: Buat fungsi wrapper untuk menghindari rekursi maksimum
        def safe_handle_download_event(event_type, sender, **kwargs):
            # Filter sender untuk menghindari rekursi
            if hasattr(sender, '_received_from_observer') and sender._received_from_observer:
                return
                
            # PERBAIKAN: Gunakan try-except untuk menangkap error
            try:
                _handle_download_event(ui_components, event_type, **kwargs)
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Error pada observer handler: {str(e)}")
        
        observer_group = ui_components.get('observer_group', 'dataset_download_observers')
        for event in events:
            observer_manager.create_simple_observer(
                event_type=event,
                callback=safe_handle_download_event,
                name=f"DownloadProgress_{event}_Observer",
                group=observer_group
            )
        
        if logger:
            logger.debug(f"✅ Download progress observer berhasil disetup untuk {len(events)} events")
    
    except (ImportError, Exception) as e:
        if logger:
            logger.warning(f"⚠️ Error saat setup download progress observer: {str(e)}")

def _handle_download_event(ui_components: Dict[str, Any], event_type: str, **kwargs) -> None:
    """
    Handler untuk event download.
    
    Args:
        ui_components: Dictionary komponen UI
        event_type: Tipe event
        **kwargs: Parameter tambahan dari event
    """
    # PERBAIKAN: Set flag untuk menghindari rekursi
    sender = kwargs.get('sender', None)
    if hasattr(sender, '_received_from_observer'):
        sender._received_from_observer = True
    
    from smartcash.components.observer.event_topics_observer import EventTopics
    
    # Handler sesuai tipe event
    if event_type.endswith('start'):
        _handle_start_event(ui_components, event_type, **kwargs)
    elif event_type.endswith('progress'):
        _handle_progress_event(ui_components, event_type, **kwargs)
    elif event_type.endswith('complete'):
        _handle_complete_event(ui_components, event_type, **kwargs)
    elif event_type.endswith('error'):
        _handle_error_event(ui_components, event_type, **kwargs)

def _handle_start_event(ui_components: Dict[str, Any], event_type: str, **kwargs) -> None:
    """Handler untuk event .start."""
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    # Get message dan total
    message = kwargs.get('message', 'Memulai download...')
    total = kwargs.get('total', 100)
    
    # Update progress components
    if progress_bar:
        progress_bar.value = 0
        progress_bar.max = total
        progress_bar.layout.visibility = 'visible'
    
    if progress_message:
        progress_message.value = message
        progress_message.layout.visibility = 'visible'
    
    # Update UI dengan info
    from smartcash.ui.utils.ui_logger import log_to_ui
    log_to_ui(ui_components, message, "info", "🚀")

def _handle_progress_event(ui_components: Dict[str, Any], event_type: str, **kwargs) -> None:
    """Handler untuk event .progress."""
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    # Get progress, total dan message
    progress = kwargs.get('progress', 0)
    total = kwargs.get('total', 100)
    message = kwargs.get('message')
    
    # Update progress bar
    if progress_bar:
        progress_bar.max = total
        progress_bar.value = progress
    
    # Update message jika ada
    if message and progress_message:
        progress_message.value = message
    
    # Update progress tracker jika tersedia
    tracker_key = 'download_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        # Jalankan update hanya jika progress berubah signifikan (min 5%)
        if not hasattr(tracker, '_last_progress') or abs(progress - getattr(tracker, '_last_progress', 0)) >= 5:
            tracker.update(progress, message)
            setattr(tracker, '_last_progress', progress)

def _handle_complete_event(ui_components: Dict[str, Any], event_type: str, **kwargs) -> None:
    """Handler untuk event .complete."""
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    # Get message
    message = kwargs.get('message', 'Download selesai')
    
    # Update progress ke 100%
    if progress_bar:
        progress_bar.value = progress_bar.max
    
    if progress_message:
        progress_message.value = message
    
    # Update tracker
    tracker_key = 'download_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        tracker.complete(message)
    
    # Update UI dengan success
    from smartcash.ui.utils.ui_logger import log_to_ui
    log_to_ui(ui_components, message, "success", "✅")

def _handle_error_event(ui_components: Dict[str, Any], event_type: str, **kwargs) -> None:
    """Handler untuk event .error."""
    progress_message = ui_components.get('progress_message')
    
    # Get error message
    error_msg = kwargs.get('message', 'Error saat download')
    
    # Update progress message
    if progress_message:
        progress_message.value = error_msg
    
    # Update UI dengan error
    from smartcash.ui.utils.ui_logger import log_to_ui
    log_to_ui(ui_components, error_msg, "error", "❌")
    
    # Log juga ke logger
    logger = ui_components.get('logger')
    if logger:
        logger.error(f"❌ {error_msg}")