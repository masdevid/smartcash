"""
File: smartcash/ui/handlers/multi_progress.py
Deskripsi: Integrasi multiple progress tracker untuk UI dengan dukungan step dan overall progress
"""

from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

def setup_multi_progress_tracking(
    ui_components: Dict[str, Any],
    overall_tracker_name: str,
    step_tracker_name: str,
    overall_progress_key: str = 'overall_progress',
    step_progress_key: str = 'step_progress',
    overall_label_key: str = 'overall_message',
    step_label_key: str = 'step_message'
) -> None:
    """
    Setup multi-level progress tracking (overall dan step-by-step).
    
    Args:
        ui_components: Dictionary komponen UI
        overall_tracker_name: Nama untuk overall progress tracker
        step_tracker_name: Nama untuk step progress tracker
        overall_progress_key: Key untuk overall progress bar di ui_components
        step_progress_key: Key untuk step progress bar di ui_components
        overall_label_key: Key untuk overall label di ui_components
        step_label_key: Key untuk step label di ui_components
    """
    logger = ui_components.get('logger')
    
    try:
        # Import progress tracker
        from smartcash.common.progress_tracker import get_progress_tracker
        
        # Dapatkan widgets
        overall_progress = ui_components.get(overall_progress_key)
        step_progress = ui_components.get(step_progress_key)
        overall_label = ui_components.get(overall_label_key)
        step_label = ui_components.get(step_label_key)
        
        if not overall_progress or not step_progress:
            if logger:
                logger.warning("⚠️ Progress widgets tidak ditemukan")
            return
        
        # Buat progress trackers
        overall_tracker = get_progress_tracker(overall_tracker_name, 100, "Overall Progress")
        step_tracker = get_progress_tracker(step_tracker_name, 100, "Step Progress")
        
        # Simpan ke ui_components
        ui_components[f'{overall_tracker_name}_tracker'] = overall_tracker
        ui_components[f'{step_tracker_name}_tracker'] = step_tracker
        
        # ---------- Setup callback untuk overall progress ----------
        def update_overall_ui(progress_info: Dict[str, Any]) -> None:
            if overall_progress and hasattr(overall_progress, 'max') and hasattr(overall_progress, 'value'):
                overall_progress.max = progress_info['total']
                overall_progress.value = progress_info['current']
                
                # Update persentase
                if hasattr(overall_progress, 'description'):
                    percentage = int(progress_info['progress_pct'])
                    overall_progress.description = f"Total: {percentage}%"
            
            # Update label
            if overall_label and hasattr(overall_label, 'value'):
                message = progress_info.get('desc', "Overall Progress")
                overall_label.value = message
        
        # ---------- Setup callback untuk step progress ----------
        def update_step_ui(progress_info: Dict[str, Any]) -> None:
            if step_progress and hasattr(step_progress, 'max') and hasattr(step_progress, 'value'):
                step_progress.max = progress_info['total']
                step_progress.value = progress_info['current']
                
                # Update persentase
                if hasattr(step_progress, 'description'):
                    percentage = int(progress_info['progress_pct'])
                    step_progress.description = f"Step: {percentage}%"
            
            # Update label
            if step_label and hasattr(step_label, 'value'):
                message = progress_info.get('desc', "Step Progress")
                step_label.value = message
        
        # Register callbacks
        overall_tracker.add_callback(update_overall_ui)
        step_tracker.add_callback(update_step_ui)
        
        # Setup tampilan progress bars
        for widget in [overall_progress, step_progress]:
            if hasattr(widget, 'layout') and hasattr(widget.layout, 'visibility'):
                widget.layout.visibility = 'visible'
        
        # Setup tampilan labels
        for widget in [overall_label, step_label]:
            if widget and hasattr(widget, 'layout') and hasattr(widget.layout, 'visibility'):
                widget.layout.visibility = 'visible'
        
        # Register observer integration
        register_multi_progress_observer(ui_components, overall_tracker_name, step_tracker_name)
        
        if logger:
            logger.debug(f"✅ Multi-progress tracking berhasil disetup: {overall_tracker_name} + {step_tracker_name}")
    
    except ImportError as e:
        if logger:
            logger.warning(f"⚠️ Error saat setup multi-progress tracking: {str(e)}")

def register_multi_progress_observer(
    ui_components: Dict[str, Any], 
    overall_tracker_name: str,
    step_tracker_name: str
) -> None:
    """
    Register observer untuk multi-level progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        overall_tracker_name: Nama untuk overall progress tracker
        step_tracker_name: Nama untuk step progress tracker
    """
    try:
        # Import komponen yang diperlukan
        from smartcash.components.observer.event_topics_observer import EventTopics
        
        # Dapatkan observer manager
        observer_manager = ui_components.get('observer_manager')
        if not observer_manager:
            from smartcash.ui.handlers.observer_handler import setup_observer_handlers
            ui_components = setup_observer_handlers(ui_components)
            observer_manager = ui_components.get('observer_manager')
            if not observer_manager:
                return
        
        # Dapatkan trackers
        overall_tracker = ui_components.get(f'{overall_tracker_name}_tracker')
        step_tracker = ui_components.get(f'{step_tracker_name}_tracker')
        
        if not overall_tracker or not step_tracker:
            return
        
        # Dapatkan observer group
        observer_group = ui_components.get('observer_group', 'cell_observers')
        
        # --------- Map event types berdasarkan nama tracker ---------
        overall_events = []
        step_events = []
        
        # Tentukan event types berdasarkan specific domain
        if 'preprocess' in overall_tracker_name.lower():
            overall_events = [EventTopics.PREPROCESSING_PROGRESS]
            step_events = [
                EventTopics.PREPROCESSING_STEP_PROGRESS,
                EventTopics.PREPROCESSING_CURRENT_PROGRESS
            ]
        elif 'augment' in overall_tracker_name.lower():
            overall_events = [EventTopics.AUGMENTATION_PROGRESS]
            step_events = [
                EventTopics.AUGMENTATION_CURRENT_PROGRESS
            ]
        elif 'train' in overall_tracker_name.lower():
            overall_events = [EventTopics.TRAINING_PROGRESS]
            step_events = [
                EventTopics.TRAINING_EPOCH_START,
                EventTopics.TRAINING_EPOCH_END,
                EventTopics.TRAINING_BATCH_END
            ]
        elif 'download' in overall_tracker_name.lower():
            overall_events = [EventTopics.DOWNLOAD_PROGRESS]
            step_events = [
                EventTopics.PULL_DATASET_PROGRESS,
                EventTopics.ZIP_PROCESSING_PROGRESS
            ]
        else:
            # Default events
            overall_events = [EventTopics.PROGRESS_UPDATE]
            step_events = [EventTopics.PREPROCESSING_STEP_PROGRESS]
        
        # Add general events to both trackers
        overall_events.extend([
            EventTopics.PROGRESS_START,
            EventTopics.PROGRESS_COMPLETE
        ])
        
        # --------- Handle overall progress events ---------
        for event_type in overall_events:
            observer_manager.create_simple_observer(
                event_type=event_type,
                callback=lambda event_type, sender, progress=None, total=None, message=None, **kwargs: 
                    _handle_progress_event(overall_tracker, event_type, progress, total, message, is_step=False, **kwargs),
                name=f"Overall_{event_type}_Observer",
                group=observer_group
            )
        
        # --------- Handle step progress events ---------
        for event_type in step_events:
            observer_manager.create_simple_observer(
                event_type=event_type,
                callback=lambda event_type, sender, progress=None, total=None, message=None, **kwargs: 
                    _handle_progress_event(step_tracker, event_type, progress, total, message, is_step=True, **kwargs),
                name=f"Step_{event_type}_Observer",
                group=observer_group
            )
    
    except (ImportError, Exception) as e:
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"ℹ️ Multi-progress observer tidak disetup: {str(e)}")

def _handle_progress_event(
    tracker, 
    event_type: str, 
    progress: Optional[int] = None, 
    total: Optional[int] = None, 
    message: Optional[str] = None, 
    is_step: bool = False,
    **kwargs
) -> None:
    """
    Handler untuk progress event dengan dukungan untuk overall dan step progress.
    
    Args:
        tracker: Progress tracker
        event_type: Tipe event
        progress: Nilai progres
        total: Total progres
        message: Pesan progres
        is_step: Flag untuk step progress
        **kwargs: Keyword arguments tambahan
    """
    # Handle sub-step vs overall events sesuai dengan event type dan flag is_step
    # Untuk event yang mengandung 'step', pastikan kita update step tracker
    is_step_event = 'step' in event_type.lower() or is_step
    
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
            
            # Interpretasikan progress berdasarkan range
            if 0 <= progress <= 1:
                # Progress sebagai persentase (0-1)
                absolute_progress = int(progress * tracker.total)
                increment = max(0, absolute_progress - tracker.current)
            else:
                # Progress sebagai nilai absolut
                increment = max(0, progress - tracker.current)
            
            # Update progress jika ada increment
            if increment > 0:
                tracker.update(increment, message)
    
    elif event_type.endswith('.complete') or event_type.endswith('.end'):
        # Complete progress
        tracker.complete(message)