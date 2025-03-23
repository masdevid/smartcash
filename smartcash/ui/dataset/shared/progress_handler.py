"""
File: smartcash/ui/dataset/shared/progress_handler.py
Deskripsi: Utilitas bersama untuk progress tracking dengan throttling yang ditingkatkan
"""

from typing import Dict, Any, Tuple, Callable
import time
from IPython.display import display
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_throttled_progress_callback(ui_components: Dict[str, Any], logger=None):
    """
    Setup callback progress dengan batasan frekuensi update dan throttling yang ditingkatkan.
    
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
    total_info = {
        'total_files': 0, 
        'processed_files': 0, 
        'current_split': '', 
        'splits': {},
        'important_messages': set()  # Untuk menyimpan pesan penting
    }
    
    # Fungsi progress callback dengan throttling log yang ditingkatkan
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
        
        # Ekstrak metadata untuk perhitungan progress yang lebih akurat
        step = kwargs.get('step', 0)      # 0=persiapan, 1=proses split, 2=finalisasi
        split = kwargs.get('split', '')
        split_step = kwargs.get('split_step', '')
        
        # PERBAIKAN: Tambahkan informasi total file di semua split
        total_files_all = kwargs.get('total_files_all', 0)
        
        # Update informasi split saat ini jika berbeda
        if split and split != total_info['current_split']:
            total_info['current_split'] = split
            # Log pergantian split secara eksplisit sebagai pesan penting
            if logger:
                logger.info(f"{ICONS['processing']} Memulai proses split {split}")
            
            with ui_components['status']:
                display(create_status_indicator('info', f"{ICONS['processing']} Memulai proses split {split}"))
        
        # ===== PERHITUNGAN PROGRESS KESELURUHAN DENGAN PERBAIKAN =====
        # Jika ini pertama kali melihat split ini, tambahkan ke statistik
        if split and total is not None and split not in total_info['splits']:
            total_info['splits'][split] = {'total': total, 'progress': 0}
            total_info['total_files'] = total_files_all or total_info['total_files'] + total
            
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
        
        # PERBAIKAN: Gunakan total_files_all jika tersedia
        total_files_all = total_files_all or total_info['total_files']
        
        # Hitung total progress berdasarkan total file di semua split
        overall_progress = total_info['processed_files']
        overall_total = max(total_files_all, 1)  # Hindari division by zero
        
        # ===== UPDATE PROGRESS BAR TOTAL =====
        if current_time - last_update.get('time', 0) >= update_interval:
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].max = overall_total
                ui_components['progress_bar'].value = min(overall_progress, overall_total)
                
                # Hitung persentase untuk overall progress bar
                overall_percentage = int(overall_progress / overall_total * 100)
                ui_components['progress_bar'].description = f"Total: {overall_percentage}%"
                ui_components['progress_bar'].layout.visibility = 'visible'
                
                last_update['time'] = current_time
                
        # ===== UPDATE CURRENT PROGRESS BAR (UNTUK SPLIT SAAT INI) =====
        if progress is not None and total is not None and total > 0 and 'current_progress' in ui_components:
            if current_time - last_update.get('time', 0) >= update_interval:
                ui_components['current_progress'].max = total
                ui_components['current_progress'].value = min(progress, total)
                
                # Format deskripsi split dengan lebih informatif
                split_percentage = int(progress / total * 100) if total > 0 else 0
                
                # PERBAIKAN: Tampilkan informasi split step jika tersedia
                if split_step:
                    ui_components['current_progress'].description = f"Split {split_step}: {split_percentage}%"
                elif split:
                    ui_components['current_progress'].description = f"Split {split}: {split_percentage}%"
                else:
                    ui_components['current_progress'].description = f"Progress: {split_percentage}%"
                    
                ui_components['current_progress'].layout.visibility = 'visible'
                
                last_update['time'] = current_time
        
        # ===== UPDATE LOG UI (DENGAN THROTTLING YANG DITINGKATKAN) =====
        if message:
            # PERBAIKAN: Menentukan apakah pesan saat ini adalah pesan penting
            is_important = (
                status != 'info' or                # Bukan info biasa
                step == 0 or step == 2 or          # Tahap persiapan atau finalisasi
                "selesai" in message.lower() or    # Pesan selesai
                "memulai" in message.lower() or    # Pesan mulai
                "analyzing" in message.lower() or  # Pesan analisis
                "balancing" in message.lower() or  # Pesan balancing
                "memindahkan" in message.lower() or # Pesan memindahkan
                message not in total_info['important_messages'] # Pesan belum pernah ditampilkan
            )
            
            # Simpan pesan penting untuk menghindari duplikasi
            if is_important:
                total_info['important_messages'].add(message)
            
            time_to_log = current_time - last_update.get('message_time', 0) >= message_throttle_interval
            
            # PERBAIKAN: Log hanya untuk pesan penting atau sesuai interval
            if (is_important and time_to_log) or status != 'info':
                # Log dengan logger jika tersedia
                if logger:
                    log_method = getattr(logger, status) if hasattr(logger, status) else logger.info
                    log_method(message)
                
                # Update timestamp terakhir untuk message
                last_update['message_time'] = current_time
                
                # Display di output widget jika status penting atau pergantian step
                if status != 'info' or step == 0 or step == 2 or is_important:
                    with ui_components['status']:
                        display(create_status_indicator(status, message))
                        
        # ===== NOTIFY OBSERVER UNTUK INTEGRASI DENGAN SISTEM LAIN =====
        if current_time - last_update.get('notify_time', 0) >= message_throttle_interval:
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                
                # Tentukan event type berdasarkan context (preprocessing atau augmentation)
                event_type_prefix = (
                    'PREPROCESSING' if 'preprocessing_running' in ui_components 
                    else 'AUGMENTATION'
                )
                
                # PERBAIKAN: Sertakan informasi split dan persentase yang lebih akurat
                notify_message = message or f"Progress {int(overall_progress/overall_total*100)}%"
                if split and current_progress is not None and current_total is not None:
                    notify_message = f"Split {split}: {int(current_progress/current_total*100)}% - {notify_message}"
                
                # Update overall progress untuk observer
                notify(
                    event_type=getattr(EventTopics, f"{event_type_prefix}_PROGRESS"),
                    sender=f"{'preprocessing' if 'preprocessing_running' in ui_components else 'augmentation'}_handler",
                    message=notify_message,
                    progress=overall_progress,
                    total=overall_total,
                    current_split=split,
                    current_progress=progress,
                    current_total=total,
                    total_files_all=total_files_all,
                    **kwargs
                )
                
                last_update['notify_time'] = current_time
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
        
        # Reset informasi progress tracking
        total_info.clear()
        total_info.update({
            'total_files': 0, 
            'processed_files': 0, 
            'current_split': '', 
            'splits': {},
            'important_messages': set()
        })
        
        # Reset timestamp throttling
        last_update.clear()
        last_update.update({'time': 0, 'message_time': 0, 'notify_time': 0})
        
        # Log reset
        if logger: logger.debug(f"{ICONS['refresh']} Progress bar direset")
    
    return progress_callback, register_progress_callback, reset_progress_bar