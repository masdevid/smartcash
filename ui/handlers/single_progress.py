"""
File: smartcash/ui/handlers/single_progress.py
Deskripsi: Integrasi progress tracker dari common dengan UI komponen
"""

from typing import Dict, Any, Optional, Union, List
import ipywidgets as widgets

def setup_progress_tracking(
    ui_components: Dict[str, Any],
    tracker_name: str,
    progress_widget_key: str = 'progress_bar',
    progress_label_key: str = 'progress_message',
    total: int = 100,
    description: str = "Progress"
) -> Any:
    """
    Setup integrasi antara widget UI dan progress_tracker dari common.
    
    Args:
        ui_components: Dictionary komponen UI
        tracker_name: Nama untuk progress tracker
        progress_widget_key: Key untuk progress bar widget di ui_components
        progress_label_key: Key untuk progress label widget di ui_components
        total: Total nilai progres
        description: Deskripsi progres
        
    Returns:
        Progress tracker instance atau None jika gagal
    """
    logger = ui_components.get('logger')
    progress_bar = ui_components.get(progress_widget_key)
    progress_label = ui_components.get(progress_label_key)
    
    if not progress_bar:
        if logger:
            logger.warning(f"⚠️ Widget progress bar '{progress_widget_key}' tidak ditemukan")
        return None
    
    try:
        # Import progress tracker
        from smartcash.common.progress import get_progress_tracker, ProgressTracker
        
        # Buat progress tracker
        tracker = get_progress_tracker(tracker_name, total, description)
        
        # Simpan referensi ke ui_components
        ui_components[f'{tracker_name}_tracker'] = tracker
        
        # Buat callback untuk update progress widget
        def update_progress_ui(progress_info: Dict[str, Any]) -> None:
            """
            Update progress UI berdasarkan info dari progress tracker.
            
            Args:
                progress_info: Info progres dari progress tracker
            """
            # Update progress bar
            if progress_bar and hasattr(progress_bar, 'max') and hasattr(progress_bar, 'value'):
                progress_bar.max = progress_info['total']
                progress_bar.value = progress_info['current']
                
                # Update persentase jika punya description
                if hasattr(progress_bar, 'description'):
                    percentage = int(progress_info['progress_pct'])
                    progress_bar.description = f"Proses: {percentage}%"
                
                # Pastikan progress bar terlihat
                if hasattr(progress_bar, 'layout') and hasattr(progress_bar.layout, 'visibility'):
                    progress_bar.layout.visibility = 'visible'
            
            # Update label jika ada dan message tersedia
            if progress_label and hasattr(progress_label, 'value'):
                # Pilih pesan dari progress info
                message = progress_info.get('desc', description)
                progress_label.value = message
                
                # Pastikan label terlihat
                if hasattr(progress_label, 'layout') and hasattr(progress_label.layout, 'visibility'):
                    progress_label.layout.visibility = 'visible'
        
        # Daftarkan callback ke tracker
        tracker.add_callback(update_progress_ui)
        
        # Tampilkan progress bar jika tersedia
        if hasattr(progress_bar, 'layout') and hasattr(progress_bar.layout, 'visibility'):
            progress_bar.layout.visibility = 'visible'
        if progress_label and hasattr(progress_label, 'layout') and hasattr(progress_label.layout, 'visibility'):
            progress_label.layout.visibility = 'visible'
        
        # Daftarkan observer jika tersedia
        register_progress_observer(ui_components, tracker_name)
        
        if logger:
            logger.debug(f"✅ Progress tracking berhasil disetup: {tracker_name}")
        
        return tracker
    
    except ImportError as e:
        if logger:
            logger.warning(f"⚠️ Error saat setup progress tracking: {str(e)}")
        return None

def register_progress_observer(ui_components: Dict[str, Any], tracker_name: str) -> None:
    """
    Daftarkan progress_observer jika tersedia.
    
    Args:
        ui_components: Dictionary komponen UI
        tracker_name: Nama progress tracker
    """
    try:
        # Import progress observer
        from smartcash.common.progress import ProgressObserver
        
        # Dapatkan observer manager
        observer_manager = ui_components.get('observer_manager')
        if not observer_manager:
            return
        
        # Dapatkan tracker
        tracker_key = f'{tracker_name}_tracker'
        tracker = ui_components.get(tracker_key)
        if not tracker:
            return
        
        # Import event topics
        from smartcash.components.observer.event_topics_observer import EventTopics
        
        # Definisikan event progress berdasarkan tracker_name
        progress_events = []
        
        # Map tracker_name ke event types
        if tracker_name == 'preprocessing':
            progress_events = [
                EventTopics.PREPROCESSING_PROGRESS,
                EventTopics.PREPROCESSING_CURRENT_PROGRESS,
                EventTopics.PREPROCESSING_STEP_PROGRESS
            ]
        elif tracker_name == 'augmentation':
            progress_events = [
                EventTopics.AUGMENTATION_PROGRESS,
                EventTopics.AUGMENTATION_CURRENT_PROGRESS
            ]
        elif tracker_name == 'training':
            progress_events = [
                EventTopics.TRAINING_PROGRESS,
                EventTopics.TRAINING_EPOCH_START,
                EventTopics.TRAINING_EPOCH_END
            ]
        elif tracker_name == 'download':
            progress_events = [
                EventTopics.DOWNLOAD_PROGRESS,
                EventTopics.DOWNLOAD_START,
                EventTopics.DOWNLOAD_END
            ]
        elif tracker_name == 'dependency_installer':
            # Tambahkan event khusus untuk dependency installer
            progress_events = [
                EventTopics.PROGRESS_UPDATE,
                EventTopics.PROGRESS_START,
                EventTopics.PROGRESS_COMPLETE,
                EventTopics.DEPENDENCY_INSTALL_PROGRESS
            ]
        else:
            # Default progress events
            progress_events = [
                EventTopics.PROGRESS_UPDATE,
                EventTopics.PROGRESS_START,
                EventTopics.PROGRESS_COMPLETE
            ]
        
        # Buat observer dengan group dari ui_components
        observer_group = ui_components.get('observer_group', 'cell_observers')
        
        # Daftarkan observer untuk event-event ini
        for event_type in progress_events:
            observer_manager.create_simple_observer(
                event_type=event_type,
                callback=lambda event_type, sender, progress=None, total=None, message=None, **kwargs: 
                    _handle_progress_event(tracker, event_type, progress, total, message, **kwargs),
                name=f"{tracker_name}_{event_type}_Observer",
                group=observer_group
            )
    
    except (ImportError, Exception) as e:
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"ℹ️ Progress observer tidak disetup: {str(e)}")

def _handle_progress_event(
    tracker, 
    event_type: str, 
    progress: Optional[int] = None, 
    total: Optional[int] = None, 
    message: Optional[str] = None, 
    **kwargs
) -> None:
    """
    Handler untuk progress event.
    
    Args:
        tracker: Progress tracker
        event_type: Tipe event
        progress: Nilai progres
        total: Total progres
        message: Pesan progres
        **kwargs: Keyword arguments tambahan
    """
    if event_type.endswith('.start'):
        # Reset progress tracker
        if total:
            tracker.set_total(total)
        tracker.current = 0
        if message:
            tracker.desc = message
    
    elif event_type.endswith('.update') or 'progress' in event_type.lower():
        # Update progress
        if progress is not None:
            if total and total != tracker.total:
                tracker.set_total(total)
            
            # Handle progress sebagai persentase (0-1) atau nilai absolut
            if 0 <= progress <= 1:
                # Progress adalah persentase
                absolute_progress = int(progress * tracker.total)
                increment = absolute_progress - tracker.current
            else:
                # Progress adalah nilai absolut
                increment = progress - tracker.current
            
            # Update progress jika ada increment
            if increment > 0:
                tracker.update(increment, message)
            elif increment == 0 and message:  # Update pesan meskipun tidak ada perubahan progress
                tracker.desc = message
                tracker.notify_callbacks()
    
    elif event_type.endswith('.complete') or event_type.endswith('.end'):
        # Complete progress
        tracker.complete(message)