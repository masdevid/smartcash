"""
File: smartcash/ui/dataset/download/handlers/download_progress_observer.py
Deskripsi: Observer untuk memantau dan memperbarui progress download di UI
"""

from typing import Dict, Any, Optional
import time
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
            
            # PERBAIKAN: Filter lebih ketat untuk mencegah cross-talk dengan modul lain
            # 1. Periksa module_type jika ada
            module_type = kwargs.get('module_type', '')
            if module_type and module_type not in ['download', 'dataset']:
                if logger: logger.debug(f"🔍 Mengabaikan event dari modul: {module_type}")
                return
            
            # 2. Periksa event_type untuk memastikan hanya event download yang diproses
            if not any(event in event_type for event in ['download', 'dataset.pull', 'zip']):
                if logger: logger.debug(f"🔍 Mengabaikan event type: {event_type}")
                return
            
            # 3. Periksa jika ada kata kunci augmentasi dalam pesan
            message = kwargs.get('message', '')
            if 'augmentasi' in message.lower():
                if logger: logger.debug(f"🔍 Mengabaikan pesan augmentasi: {message}")
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
        
        # Gunakan nama grup yang spesifik untuk download
        observer_group = 'dataset_download_observers'
        ui_components['observer_group'] = observer_group
        
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
    logger = ui_components.get('logger')
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    # Get message dan total
    message = kwargs.get('message', 'Memulai download...')
    total = kwargs.get('total', 100)
    total_steps = kwargs.get('total_steps', 0)
    
    # Log start event
    if logger:
        logger.info(f"🚀 {message}")
        if total_steps > 0:
            logger.debug(f"📋 Download akan dilakukan dalam {total_steps} langkah")
    
    # Reset dan tampilkan progress bar
    if progress_bar:
        progress_bar.max = total
        progress_bar.value = 0
        if hasattr(progress_bar, 'layout'):
            progress_bar.layout.visibility = 'visible'
    
    # Set message dan tampilkan
    if progress_message:
        progress_message.value = message
        if hasattr(progress_message, 'layout'):
            progress_message.layout.visibility = 'visible'
    
    # Cek apakah tracker sudah ada, jika belum, buat baru
    if 'download_tracker' not in ui_components:
        from smartcash.common.progress.tracker import ProgressTracker
        ui_components['download_tracker'] = ProgressTracker(
            total=100,  # Selalu gunakan 0-100 untuk overall progress
            desc="Download Dataset",
            unit="%",
            display=True,
            logger=logger
        )
    
    # Reset dan inisialisasi overall tracker
    tracker = ui_components['download_tracker']
    if hasattr(tracker, 'reset'):
        tracker.reset()
    if hasattr(tracker, 'set_total'):
        tracker.set_total(100)  # Selalu gunakan 0-100 untuk overall progress
    if hasattr(tracker, 'update'):
        tracker.update(0, message)
    
    # Cek apakah step tracker sudah ada, jika belum, buat baru
    if 'download_step_tracker' not in ui_components:
        from smartcash.common.progress.tracker import ProgressTracker
        ui_components['download_step_tracker'] = ProgressTracker(
            total=total_steps if total_steps > 0 else total,
            desc="Download Step",
            unit="it",
            display=True,
            logger=logger
        )
    
    # Reset dan inisialisasi step tracker
    step_tracker = ui_components['download_step_tracker']
    if hasattr(step_tracker, 'reset'):
        step_tracker.reset()
    if hasattr(step_tracker, 'set_total'):
        if total_steps > 0:
            step_tracker.set_total(total_steps)
        else:
            step_tracker.set_total(total)
    if hasattr(step_tracker, 'update'):
        step_tracker.update(0, message)
    
    # Tambahkan metadata untuk tracking
    ui_components['download_start_time'] = time.time()
    ui_components['download_total_steps'] = total_steps
    
    # Update UI dengan info download
    from smartcash.ui.utils.ui_logger import log_to_ui
    log_to_ui(ui_components, message, "info", "⏳")
    
    # Update status panel menggunakan komponen reusable
    from smartcash.ui.components.status_panel import update_status_panel
    update_status_panel(ui_components['status_panel'], message, "info")

def _handle_progress_event(ui_components: Dict[str, Any], event_type: str, **kwargs) -> None:
    """Handler untuk event .progress."""
    logger = ui_components.get('logger')
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    # Get progress, total dan message
    progress = kwargs.get('progress', 0)
    total = kwargs.get('total', 100)
    message = kwargs.get('message', '')
    step = kwargs.get('step', '')
    current_step = kwargs.get('current_step', 0)
    total_steps = kwargs.get('total_steps', 0)
    
    # Pastikan progress dan total valid
    if progress < 0:
        progress = 0
    if total <= 0:
        total = 100
    
    # Pastikan progress tidak melebihi total
    if progress > total:
        progress = total
    
    # Hitung persentase progress untuk logging
    progress_pct = (progress / total) * 100 if total > 0 else 0
    
    # Update progress bar dengan nilai yang valid
    if progress_bar:
        # Pastikan progress bar visible
        if hasattr(progress_bar, 'layout'):
            progress_bar.layout.visibility = 'visible'
        progress_bar.max = total
        progress_bar.value = progress
    
    # Update message jika ada
    if message and progress_message:
        if hasattr(progress_message, 'layout'):
            progress_message.layout.visibility = 'visible'
        progress_message.value = message
        
        # Log progress untuk debugging
        if logger and progress_pct % 10 < 1:  # Log setiap ~10%
            logger.debug(f"📊 Progress download: {progress_pct:.1f}% - {message}")
    
    # Update overall tracker
    if 'download_tracker' in ui_components:
        overall_tracker = ui_components['download_tracker']
        
        # Update overall progress berdasarkan step dan progress
        if current_step > 0 and total_steps > 0:
            # Hitung overall progress berdasarkan step dan progress dalam step
            # Formula: (step_completed + current_progress_ratio) / total_steps * 100
            step_completed = current_step - 1  # Step yang sudah selesai
            current_progress_ratio = progress / total if total > 0 else 0  # Rasio progress di step saat ini
            
            overall_progress = (step_completed + current_progress_ratio) / total_steps * 100
            
            # Pastikan overall progress dalam rentang valid (0-100)
            overall_progress = max(0, min(100, overall_progress))
            
            # Update tracker dengan progress dan message
            if hasattr(overall_tracker, 'update'):
                overall_tracker.update(overall_progress, message)
        else:
            # Gunakan progress langsung jika tidak ada informasi step
            normalized_progress = (progress / total) * 100 if total > 0 else progress
            
            # Pastikan normalized_progress dalam rentang valid (0-100)
            normalized_progress = max(0, min(100, normalized_progress))
            
            if hasattr(overall_tracker, 'update'):
                overall_tracker.update(normalized_progress, message)
    
    # Update step tracker jika ada
    if 'download_step_tracker' in ui_components and step:
        step_tracker = ui_components['download_step_tracker']
        
        # Buat pesan step yang informatif
        step_message = f"Step {current_step}/{total_steps}: {message}" if current_step > 0 and total_steps > 0 else message
        
        # Update step progress
        if hasattr(step_tracker, 'update'):
            step_tracker.update(progress, step_message)

def _handle_complete_event(ui_components: Dict[str, Any], event_type: str, **kwargs) -> None:
    """Handler untuk event .complete."""
    logger = ui_components.get('logger')
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    # Get message dan hasil
    message = kwargs.get('message', 'Download selesai')
    result = kwargs.get('result', {})
    
    # Hitung waktu yang dibutuhkan jika tersedia start_time
    elapsed_time = ""
    if 'download_start_time' in ui_components:
        import time
        elapsed = time.time() - ui_components['download_start_time']
        if elapsed < 60:
            elapsed_time = f" dalam {elapsed:.1f} detik"
        elif elapsed < 3600:
            elapsed_time = f" dalam {elapsed/60:.1f} menit"
        else:
            elapsed_time = f" dalam {elapsed/3600:.1f} jam"
    
    # Tambahkan informasi waktu ke pesan
    if elapsed_time and not "dalam" in message:
        message = f"{message}{elapsed_time}"
    
    # Log completion
    if logger:
        logger.info(f"✅ {message}")
        
        # Log detail hasil jika ada
        if isinstance(result, dict) and result:
            for key, value in result.items():
                if key not in ['message', 'error']:
                    logger.debug(f"📊 {key}: {value}")
    
    # Update progress ke 100%
    if progress_bar:
        progress_bar.value = progress_bar.max
        # Pastikan progress bar terlihat
        if hasattr(progress_bar, 'layout'):
            progress_bar.layout.visibility = 'visible'
    
    if progress_message:
        progress_message.value = message
        # Pastikan pesan terlihat
        if hasattr(progress_message, 'layout'):
            progress_message.layout.visibility = 'visible'
    
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
    
    # Update status panel menggunakan komponen reusable
    from smartcash.ui.components.status_panel import update_status_panel
    update_status_panel(ui_components['status_panel'], message, "success")
    
    # Reset tombol dan UI setelah download selesai
    from smartcash.ui.dataset.download.handlers.download_handler import _reset_ui_after_download
    _reset_ui_after_download(ui_components)

def _handle_error_event(ui_components: Dict[str, Any], event_type: str, **kwargs) -> None:
    """Handler untuk event .error."""
    logger = ui_components.get('logger')
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    # Get error message dan detail
    error_msg = kwargs.get('message', 'Error saat download')
    error_details = kwargs.get('error_details', '')
    error_type = kwargs.get('error_type', 'download_error')
    
    # Tambahkan detail ke pesan error jika ada
    full_error_msg = error_msg
    if error_details and not error_details in error_msg:
        full_error_msg = f"{error_msg}: {error_details}"
    
    # Update progress message dengan error
    if progress_message:
        progress_message.value = full_error_msg
        # Pastikan pesan terlihat
        if hasattr(progress_message, 'layout'):
            progress_message.layout.visibility = 'visible'
    
    # Update progress bar untuk menunjukkan error
    if progress_bar:
        # Set progress bar ke merah untuk menunjukkan error
        if hasattr(progress_bar, 'style'):
            progress_bar.style.bar_color = 'red'
        # Pastikan progress bar terlihat
        if hasattr(progress_bar, 'layout'):
            progress_bar.layout.visibility = 'visible'
    
    # Update UI dengan error
    from smartcash.ui.utils.ui_logger import log_to_ui
    log_to_ui(ui_components, full_error_msg, "error", "❌")
    
    # Update status panel menggunakan komponen reusable
    from smartcash.ui.components.status_panel import update_status_panel
    update_status_panel(ui_components['status_panel'], full_error_msg, "error")
    
    # Log error ke logger dengan detail
    if logger:
        logger.error(f"❌ {full_error_msg}")
        if error_details and error_details != full_error_msg:
            logger.debug(f"Detail error: {error_details}")
    
    # Update tracker ke error state jika tersedia
    for tracker_key in ['download_tracker', 'download_step_tracker']:
        if tracker_key in ui_components:
            tracker = ui_components[tracker_key]
            # Tambahkan metadata error ke tracker
            if hasattr(tracker, 'set_metrics'):
                tracker.set_metrics({
                    'error': full_error_msg,
                    'error_type': error_type,
                    'error_time': time.time()
                })
            # Tandai sebagai complete dengan error
            if hasattr(tracker, 'complete'):
                tracker.complete(f"Error: {full_error_msg}")
            elif hasattr(tracker, 'update'):
                tracker.update(0, f"Error: {full_error_msg}")
    
    # Reset tombol dan UI setelah error
    from smartcash.ui.dataset.download.handlers.download_handler import _reset_ui_after_download
    _reset_ui_after_download(ui_components)