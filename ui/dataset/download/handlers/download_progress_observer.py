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
                # Set flag untuk menghindari rekursi
                if sender:
                    setattr(sender, '_received_from_observer', True)
                    
                _handle_download_event(ui_components, event_type, **kwargs)
                
                # Reset flag
                if sender:
                    setattr(sender, '_received_from_observer', False)
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
    # Hapus parameter status dari kwargs untuk mencegah duplikasi
    kwargs.pop('status', None)
    
    from smartcash.components.observer.event_topics_observer import EventTopics
    
    # Handler sesuai tipe event
    if event_type in (EventTopics.DOWNLOAD_START, EventTopics.PULL_DATASET_START, EventTopics.ZIP_PROCESSING_START):
        _handle_start_event(ui_components, event_type, **kwargs)
    elif event_type in (EventTopics.DOWNLOAD_PROGRESS, EventTopics.PULL_DATASET_PROGRESS, EventTopics.ZIP_PROCESSING_PROGRESS):
        _handle_progress_event(ui_components, event_type, **kwargs)
    elif event_type in (EventTopics.DOWNLOAD_COMPLETE, EventTopics.PULL_DATASET_COMPLETE, EventTopics.ZIP_PROCESSING_COMPLETE):
        _handle_complete_event(ui_components, event_type, **kwargs)
    elif event_type in (EventTopics.DOWNLOAD_ERROR, EventTopics.PULL_DATASET_ERROR, EventTopics.ZIP_PROCESSING_ERROR):
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
    
    # Update multi-progress trackers jika tersedia
    for tracker_key in ['download_tracker', 'download_step_tracker']:
        if tracker_key in ui_components:
            tracker = ui_components[tracker_key]
            # Reset progress tracker
            if hasattr(tracker, 'reset'):
                tracker.reset()
            tracker.current = 0
            tracker.total = total
            if hasattr(tracker, 'set_description'):
                tracker.set_description(message)

def _handle_progress_event(ui_components: Dict[str, Any], event_type: str, **kwargs) -> None:
    """Handler untuk event .progress."""
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    # Get progress, total dan message
    progress = kwargs.get('progress', 0)
    total = kwargs.get('total', 100)
    message = kwargs.get('message')
    step = kwargs.get('step', '')
    current_step = kwargs.get('current_step', 0)
    total_steps = kwargs.get('total_steps', 0)
    
    # Update progress bar
    if progress_bar:
        progress_bar.max = total
        progress_bar.value = progress
    
    # Update message jika ada
    if message and progress_message:
        progress_message.value = message
    
    # Update overall tracker
    if 'download_tracker' in ui_components:
        overall_tracker = ui_components['download_tracker']
        # Update overall progress
        if current_step > 0 and total_steps > 0:
            # Calculate overall progress based on steps
            overall_progress = (current_step - 1) * 100 / total_steps
            overall_progress += progress * (100 / total_steps) / total
            overall_tracker.update(overall_progress, message)
        else:
            # Use direct progress update
            overall_tracker.update(progress, message)
    
    # Update step tracker jika ada
    if 'download_step_tracker' in ui_components and step:
        step_tracker = ui_components['download_step_tracker']
        # Update step progress
        step_tracker.update(progress, f"Step {current_step}/{total_steps}: {message}")

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
    for tracker_key in ['download_tracker', 'download_step_tracker']:
        if tracker_key in ui_components:
            tracker = ui_components[tracker_key]
            if hasattr(tracker, 'complete'):
                tracker.complete(message)
            elif hasattr(tracker, 'update'):
                tracker.update(100, message)
    
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
        
    # Update tracker ke error state jika tersedia
    for tracker_key in ['download_tracker', 'download_step_tracker']:
        if tracker_key in ui_components:
            tracker = ui_components[tracker_key]
            if hasattr(tracker, 'complete'):
                tracker.complete(f"Error: {error_msg}")
            elif hasattr(tracker, 'update'):
                tracker.update(0, f"Error: {error_msg}")