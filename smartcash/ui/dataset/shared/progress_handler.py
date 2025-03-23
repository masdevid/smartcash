"""
File: smartcash/ui/dataset/shared/progress_handler.py
Deskripsi: Utilitas bersama untuk progress tracking dengan throttling yang digunakan oleh preprocessing dan augmentasi
"""

from typing import Dict, Any, Tuple, Callable
import time
from IPython.display import display
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_throttled_progress_callback(ui_components: Dict[str, Any], logger=None):
    """
    Setup callback progress dengan batasan frekuensi update dan throttling.
    
    Args:
        ui_components: Dictionary komponen UI
        logger: Logger instance
        
    Returns:
        Tuple (progress_callback, register_progress_callback, reset_progress_bar)
    """
    # Simpan state tracking untuk batasi frekuensi update
    last_update = {'time': 0, 'message_time': 0, 'notify_time': 0}
    update_interval = 0.5  # Hanya update UI setiap 0.5 detik
    message_throttle_interval = 2.0  # Hanya log pesan setiap 2 detik
    
    # Informasi jumlah file untuk penghitungan progress
    total_info = {'total_files': 0, 'processed_files': 0, 'current_split': '', 'splits': {}}
    
    # Fungsi progress callback dengan throttling log
    def progress_callback(progress=None, total=None, message=None, status='info', 
                         current_progress=None, current_total=None, **kwargs):
        """
        Progress callback dengan throttling log untuk mencegah flooding UI.
        
        Args:
            progress: Nilai progress utama
            total: Total maksimum progress utama
            message: Pesan progress 
            status: Status progress ('info', 'success', 'warning', 'error')
            current_progress: Nilai progress saat ini (substep)
            current_total: Total maksimum progress saat ini
            **kwargs: Parameter lain yang diteruskan dari caller
        """
        # Skip jika proses sudah dihentikan
        process_running_key = 'preprocessing_running' if 'preprocessing_running' in ui_components else 'augmentation_running'
        if not ui_components.get(process_running_key, True): 
            return
            
        current_time = time.time()
        
        # Ekstrak metadata untuk perhitungan progress
        step = kwargs.get('step', 0)  # 0=persiapan, 1=proses split, 2=finalisasi
        split = kwargs.get('split', '')
        
        # Update informasi split saat ini jika berbeda
        if split and split != total_info['current_split']:
            total_info['current_split'] = split
            # Log pergantian split secara eksplisit
            if logger:
                logger.info(f"{ICONS['processing']} Memulai proses split {split}")
            
            with ui_components['status']:
                display(create_status_indicator('info', f"{ICONS['processing']} Memulai proses split {split}"))
        
        # ===== PERHITUNGAN PROGRESS KESELURUHAN =====
        # Jika ini pertama kali melihat split ini, tambahkan ke statistik
        if split and total is not None and split not in total_info['splits']:
            total_info['splits'][split] = {'total': total, 'progress': 0}
            total_info['total_files'] += total
            
        # Update progress untuk split ini
        if split and progress is not None and total is not None:
            if split in total_info['splits']:
                # Hitung delta dari progress sebelumnya
                old_progress = total_info['splits'][split].get('progress', 0)
                if progress > old_progress:
                    # Tambahkan delta ke total progress
                    total_info['processed_files'] += (progress - old_progress)
                    # Update progress untuk split ini
                    total_info['splits'][split]['progress'] = progress
        
        # Hitung total progress berdasarkan total file di semua split
        overall_progress = total_info['processed_files']
        overall_total = max(total_info['total_files'], 1)  # Hindari division by zero
        
        # ===== UPDATE PROGRESS BAR TOTAL =====
        if time.time() - last_update.get('time', 0) >= update_interval:
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].max = overall_total
                ui_components['progress_bar'].value = min(overall_progress, overall_total)
                
                # Hitung persentase untuk overall progress bar
                overall_percentage = int(overall_progress / overall_total * 100)
                ui_components['progress_bar'].description = f"Total: {overall_percentage}%"
                ui_components['progress_bar'].layout.visibility = 'visible'
                
                last_update['time'] = time.time()
                
        # ===== UPDATE CURRENT PROGRESS BAR (UNTUK SPLIT SAAT INI) =====
        if progress is not None and total is not None and total > 0 and 'current_progress' in ui_components:
            if time.time() - last_update.get('time', 0) >= update_interval:
                ui_components['current_progress'].max = total
                ui_components['current_progress'].value = min(progress, total)
                
                # Format deskripsi split
                split_percentage = int(progress / total * 100) if total > 0 else 0
                
                if split:
                    ui_components['current_progress'].description = f"Split {split}: {split_percentage}%"
                else:
                    ui_components['current_progress'].description = f"Progress: {split_percentage}%"
                    
                ui_components['current_progress'].layout.visibility = 'visible'
                
                last_update['time'] = time.time()
        
        # ===== UPDATE LOG UI (DENGAN THROTTLING) =====
        if message:
            time_to_log = current_time - last_update.get('message_time', 0) >= message_throttle_interval
            
            # Log selesainya split secara eksplisit
            split_completion = "selesai" in message.lower() and split in message.lower()
            
            # Log hanya pesan penting atau dengan interval waktu yang cukup
            if time_to_log or step == 0 or step == 2 or split_completion or status != 'info':
                # Log dengan logger jika tersedia
                if logger:
                    log_method = getattr(logger, status) if hasattr(logger, status) else logger.info
                    log_method(message)
                
                # Update timestamp terakhir untuk message
                last_update['message_time'] = current_time
                
                # Display di output widget jika status penting atau pergantian step
                if status != 'info' or step == 0 or step == 2 or split_completion:
                    with ui_components['status']:
                        display(create_status_indicator(status, message))
                        
        # ===== NOTIFY OBSERVER UNTUK INTEGRASI DENGAN SISTEM LAIN =====
        if time.time() - last_update.get('notify_time', 0) >= message_throttle_interval:
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                
                # Tentukan event type berdasarkan context (preprocessing atau augmentation)
                event_type_prefix = (
                    'PREPROCESSING' if 'preprocessing_running' in ui_components 
                    else 'AUGMENTATION'
                )
                
                # Update overall progress untuk observer
                notify(
                    event_type=getattr(EventTopics, f"{event_type_prefix}_PROGRESS"),
                    sender=f"{'preprocessing' if 'preprocessing_running' in ui_components else 'augmentation'}_handler",
                    message=message or f"Progress total: {int(overall_progress/overall_total*100)}%",
                    progress=overall_progress,
                    total=overall_total,
                    current_split=split,
                    **kwargs
                )
                
                # Update progress untuk split saat ini jika ada
                if split and progress is not None and total is not None:
                    notify(
                        event_type=getattr(EventTopics, f"{event_type_prefix}_CURRENT_PROGRESS"),
                        sender=f"{'preprocessing' if 'preprocessing_running' in ui_components else 'augmentation'}_handler",
                        message=f"Progress split {split}: {int(progress/total*100)}%",
                        progress=progress,
                        total=total,
                        **kwargs
                    )
                
                last_update['notify_time'] = time.time()
            except ImportError:
                pass
    
    # Fungsi untuk registrasi callback ke manager
    def register_progress_callback(manager):
        """Register callback progress ke manager."""
        if not manager: 
            return False
            
        # Mencoba register dengan berbagai metode yang mungkin tersedia
        try:
            # Pendekatan 1: register_progress_callback 
            if hasattr(manager, 'register_progress_callback'):
                manager.register_progress_callback(progress_callback)
                if logger: logger.info(f"{ICONS['success']} Progress callback berhasil didaftarkan ke manager")
                return True
                
            # Pendekatan 2: _progress_callback
            elif hasattr(manager, '_progress_callback'):
                manager._progress_callback = progress_callback
                if logger: logger.info(f"{ICONS['success']} Progress callback berhasil didaftarkan via atribut langsung")
                return True
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal mendaftarkan callback: {str(e)}")
            
        return False
    
    # Helper untuk reset progress bar
    def reset_progress_bar():
        """Reset semua komponen progress ke nilai awal."""
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = 'Total:'
            ui_components['progress_bar'].layout.visibility = 'hidden'
            
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].description = 'Split:'
            ui_components['current_progress'].layout.visibility = 'hidden'
        
        # Reset status flags dan informasi progress
        process_running_key = 'preprocessing_running' if 'preprocessing_running' in ui_components else 'augmentation_running'
        ui_components[process_running_key] = False
        
        total_info.clear()
        total_info.update({'total_files': 0, 'processed_files': 0, 'current_split': '', 'splits': {}})
        
        # Reset timestamp throttling
        last_update.clear()
        last_update.update({'time': 0, 'message_time': 0, 'notify_time': 0})
        
        # Log reset
        if logger: logger.debug(f"{ICONS['refresh']} Progress bar direset")
    
    return progress_callback, register_progress_callback, reset_progress_bar