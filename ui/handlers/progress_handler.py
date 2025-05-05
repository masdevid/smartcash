"""
File: smartcash/ui/handlers/progress_handler.py
Deskripsi: Handler progress yang teroptimasi untuk UI processing dengan throttling untuk menghindari flooding
"""

from typing import Dict, Any, Tuple, Callable, Optional, Union
import time
from IPython.display import display
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_throttled_progress_callback(ui_components: Dict[str, Any], logger=None) -> Tuple[Callable, Callable, Callable]:
    """
    Setup callback progress dengan throttling optimal dan normalisasi progress.
    
    Args:
        ui_components: Dictionary komponen UI
        logger: Logger instance (opsional)
        
    Returns:
        Tuple (progress_callback, register_progress_callback, reset_progress_bar)
    """
    # State minimal untuk tracking
    state = {
        'last_update': 0,           # Timestamp UI update terakhir
        'last_message_update': 0,   # Timestamp log message terakhir
        'last_notify_update': 0,    # Timestamp notifikasi observer terakhir
        'total_files': 0,           # Total file
        'processed_files': 0,       # Jumlah file yang sudah diproses
        'current_split': '',        # Split yang sedang diproses
        'splits': {},               # Progress per split
        'important_messages': set() # Pesan penting (no throttle)
    }
    
    # Throttling config
    update_interval = 0.25    # UI update interval (detik)
    message_throttle = 2.0    # Log message interval (detik)
    
    # Progress callback dengan optimasi
    def progress_callback(progress: Optional[int] = None, total: Optional[int] = None, 
                         message: Optional[str] = None, status: str = 'info', 
                         current_progress: Optional[int] = None, current_total: Optional[int] = None, 
                         **kwargs) -> None:
        """Progress callback dengan throttling dan normalisasi."""
        # Skip jika proses sudah dihentikan (early return)
        process_key = next((k for k in ['preprocessing_running', 'augmentation_running'] if k in ui_components), None)
        if process_key and not ui_components.get(process_key, True): return
            
        # Get current time untuk throttling decisions
        current_time = time.time()
        
        # Extract informasi tambahan dari kwargs
        split = kwargs.get('split', '')
        split_step = kwargs.get('split_step', '')
        step = kwargs.get('step', 0)
        total_files_all = kwargs.get('total_files_all', 0)
        
        # Tracking split change untuk log
        if split and split != state['current_split']:
            state['current_split'] = split
            # Log split change (selalu tampilkan)
            if logger: logger.info(f"{ICONS['processing']} Memulai proses split {split}")
            
            with ui_components['status']:
                display(create_status_indicator('info', f"{ICONS['processing']} Memulai proses split {split}"))
        
        # Update split info untuk progress agregasi
        if split and total is not None and split not in state['splits']:
            state['splits'][split] = {'total': total, 'progress': 0, 'weight': 1.0}
            # Update total dengan estimasi yang lebih akurat
            state['total_files'] = total_files_all if total_files_all > 0 else state.get('total_files', 0) + total
        
        # Update split progress dengan normalisasi
        if split and progress is not None and total is not None and split in state['splits']:
            # Get delta progress dengan normalisasi untuk mencegah overflow
            old_progress = state['splits'][split].get('progress', 0)
            norm_progress = min(progress, total)  # Normalisasi agar tidak melebihi total
            delta_progress = max(0, norm_progress - old_progress)  # Hindari delta negatif
            
            # Update progress state jika ada perubahan
            if delta_progress > 0 and total > 0:
                # Update split progress
                state['splits'][split]['progress'] = norm_progress
                
                # Hitung kontribusi ke total progress berdasarkan proporsi
                split_contribution = min(delta_progress / total, 1.0) * state['splits'][split]['weight']
                split_proportion = split_contribution * state['total_files'] / max(1, len(state['splits']))
                state['processed_files'] += split_proportion
        
        # Normalisasi overall progress (one-liner)
        overall_progress, overall_total = min(state['processed_files'], total_files_all or state['total_files']), max(total_files_all or state['total_files'], 1)
        
        # Update UI dengan throttling (one-liner condition)
        if current_time - state['last_update'] >= update_interval:
            # Update overall progress bar
            if 'progress_bar' in ui_components:
                # One-line assignment untuk progress properties
                ui_components['progress_bar'].max, ui_components['progress_bar'].value = overall_total, overall_progress
                ui_components['progress_bar'].description = f"Total: {min(int(overall_progress / overall_total * 100), 100)}%"
                ui_components['progress_bar'].layout.visibility = 'visible'
            
            # Update current progress (one-liner style)
            if progress is not None and total is not None and 'current_progress' in ui_components:
                # Update progress bar dengan normalisasi
                ui_components['current_progress'].max, ui_components['current_progress'].value = total, min(progress, total)
                
                # Smart description berdasarkan context (one-liner)
                description = (f"{split_step}: {min(int(progress / total * 100), 100)}%" if split_step else 
                              f"Split {split}: {min(int(progress / total * 100), 100)}%" if split else 
                              f"Progress: {min(int(progress / total * 100), 100)}%")
                
                ui_components['current_progress'].description = description
                ui_components['current_progress'].layout.visibility = 'visible'
            
            # Update message labels
            if message:
                if 'overall_label' in ui_components and hasattr(ui_components['overall_label'], 'value'):
                    ui_components['overall_label'].value = message
                    ui_components['overall_label'].layout.visibility = 'visible'
                
                if split_step and 'step_label' in ui_components and hasattr(ui_components['step_label'], 'value'):
                    ui_components['step_label'].value = f"Proses {split_step}"
                    ui_components['step_label'].layout.visibility = 'visible'
            
            # Update timestamp terakhir
            state['last_update'] = current_time
        
        # Message logging dengan smart throttling
        if message:
            # Determine apakah pesan penting (one-liner condition)
            is_important = (status != 'info' or step == 0 or step == 2 or
                           any(keyword in message.lower() for keyword in ["selesai", "memulai", "error", "balancing"]) or
                           message not in state['important_messages'])
            
            # Cache pesan penting
            if is_important: state['important_messages'].add(message)
            
            # Log dengan throttling (one-liner decision)
            time_to_log = is_important or current_time - state['last_message_update'] >= message_throttle
            
            if time_to_log:
                # Log ke logger jika tersedia (dynamic attribute access)
                if logger:
                    log_method = getattr(logger, status) if hasattr(logger, status) else logger.info
                    log_method(message)
                
                # Update UI jika penting atau sesuai throttle
                if status != 'info' or is_important or time_to_log:
                    with ui_components['status']:
                        display(create_status_indicator(status, message))
                
                # Update timestamp
                state['last_message_update'] = current_time
        
        # Notifikasi observer dengan throttling
        _notify_progress_update(ui_components, current_time, state, overall_progress, overall_total, progress, total, message, kwargs)
    
    # Fungsi untuk register callback ke manager dengan metode yang tepat
    def register_progress_callback(manager) -> bool:
        """
        Register callback ke manager instance dengan metode yang tersedia.
        
        Args:
            manager: Manager instance (DatasetManager, AugmentationService, dll)
            
        Returns:
            Boolean menunjukkan keberhasilan registrasi
        """
        if not manager: return False
            
        try:
            # One-liner untuk metode register yang mungkin tersedia
            if hasattr(manager, 'register_progress_callback'):
                manager.register_progress_callback(progress_callback)
                if logger: logger.debug(f"{ICONS['success']} Progress callback berhasil didaftarkan ke manager")
                return True
            elif hasattr(manager, '_progress_callback'):
                manager._progress_callback = progress_callback
                if logger: logger.debug(f"{ICONS['success']} Progress callback berhasil didaftarkan ke manager") 
                return True
            return False
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal mendaftarkan callback: {str(e)}")
            
        return False
    
    # Fungsi reset progress bar yang mengembalikan ke kondisi awal
    def reset_progress_bar() -> None:
        """Reset semua komponen progress ke kondisi awal."""
        # Reset UI components (one-liner style)
        [setattr(ui_components[comp], 'value', 0) for comp in ['progress_bar', 'current_progress'] 
         if comp in ui_components]
        
        [setattr(ui_components[comp].layout, 'visibility', 'hidden') for comp in ['progress_bar', 'current_progress', 'overall_label', 'step_label'] 
         if comp in ui_components and hasattr(ui_components[comp], 'layout')]
        
        # Reset running flag
        process_key = next((k for k in ['preprocessing_running', 'augmentation_running'] if k in ui_components), None)
        if process_key: ui_components[process_key] = False
        
        # Reset tracking state untuk progress baru dengan dict method
        state.clear()
        state.update({
            'last_update': 0,
            'last_message_update': 0,
            'last_notify_update': 0,
            'total_files': 0,
            'processed_files': 0,
            'current_split': '',
            'splits': {},
            'important_messages': set()
        })
        
        # Log reset jika debug
        if logger: logger.debug(f"{ICONS['refresh']} Progress bar direset")
    
    return progress_callback, register_progress_callback, reset_progress_bar

def _notify_progress_update(
    ui_components: Dict[str, Any], 
    current_time: float, 
    state: Dict[str, Any],
    overall_progress: int, 
    overall_total: int, 
    progress: Optional[int], 
    total: Optional[int], 
    message: Optional[str],
    kwargs: Dict[str, Any]
) -> None:
    """Notifikasi observer tentang update progress dengan throttling."""
    # Daftar event types yang dapat dikirim
    message_throttle = 2.0  # Throttle interval
    
    try:
        if current_time - state['last_notify_update'] >= message_throttle:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            
            # Determine event type dan sender berdasarkan ui_components
            event_type_prefix = 'PREPROCESSING' if 'preprocessing_running' in ui_components else 'AUGMENTATION'
            sender = f"{event_type_prefix.lower()}_handler"
            
            # Create message dengan informasi progress
            notify_message = message or f"Progress {min(int(overall_progress/overall_total*100), 100)}%"
            
            # Tambahkan info split jika tersedia (one-liner)
            if state['current_split'] and progress is not None and total is not None:
                notify_message = f"Split {state['current_split']}: {min(int(progress/total*100), 100)}% - {notify_message}"
            
            # Bersihkan kwargs dari parameter yang sudah digunakan (one-liner)
            clean_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['progress', 'total', 'message', 'current_progress', 'current_total']}
            
            # Kirim notifikasi dengan data ternormalisasi
            notify(
                event_type=getattr(EventTopics, f"{event_type_prefix}_PROGRESS"),
                sender=sender,
                message=notify_message,
                progress=overall_progress,
                total=overall_total,
                current_progress=progress if progress is not None else 0,
                current_total=total if total is not None else 1,
                **clean_kwargs
            )
            
            # Update timestamp
            state['last_notify_update'] = current_time
    except ImportError:
        # Observer tidak tersedia, skip silently
        pass