"""
File: smartcash/ui/dataset/shared/progress_handler.py
Deskripsi: Utilitas bersama untuk progress tracking dengan throttling yang ditingkatkan, 
normalisasi progress dan pencegahan UI flooding
"""

from typing import Dict, Any, Tuple, Callable, Optional, Union
import time
from IPython.display import display
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_throttled_progress_callback(ui_components: Dict[str, Any], logger=None) -> Tuple[Callable, Callable, Callable]:
    """
    Setup callback progress dengan throttling optimal untuk mencegah flooding UI,
    normalisasi progress dan dukungan multi-level tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        logger: Logger instance (opsional)
        
    Returns:
        Tuple (progress_callback, register_progress_callback, reset_progress_bar)
    """
    # State untuk tracking dengan minimal memory footprint
    state = {
        'last_update': 0,           # Timestamp update UI terakhir
        'last_message_update': 0,   # Timestamp log message terakhir
        'last_notify_update': 0,    # Timestamp notifikasi observer terakhir
        'total_files': 0,           # Total file yang diproses
        'processed_files': 0,       # Jumlah file yang sudah diproses
        'current_split': '',        # Split yang sedang diproses
        'splits': {},               # Info progress per split
        'important_messages': set() # Set pesan penting (untuk mencegah duplikasi)
    }
    
    # Konfigurasi throttling
    update_interval = 0.25          # Interval minimum antar update progress bar (detik)
    message_throttle = 2.0          # Interval minimum antar log pesan (detik)
    
    # Progress callback yang optimal dengan throttling dan normalisasi
    def progress_callback(progress: Optional[int] = None, total: Optional[int] = None, 
                         message: Optional[str] = None, status: str = 'info', 
                         current_progress: Optional[int] = None, current_total: Optional[int] = None, 
                         **kwargs) -> None:
        """
        Progress callback dengan throttling untuk mencegah UI flooding dan normalisasi progress.
        
        Args:
            progress: Nilai progress utama (0-total)
            total: Total maksimum progress
            message: Pesan progress
            status: Status progress ('info', 'success', 'warning', 'error')
            current_progress: Progress saat ini (untuk substep)
            current_total: Total untuk substep
            **kwargs: Parameter tambahan dari caller
        """
        # Skip jika proses sudah dihentikan
        process_key = 'preprocessing_running' if 'preprocessing_running' in ui_components else 'augmentation_running'
        if not ui_components.get(process_key, True):
            return
            
        # Waktu saat ini untuk throttling
        current_time = time.time()
        
        # Ekstrak informasi tambahan untuk tracking lebih akurat
        split = kwargs.get('split', '')         # Split dataset saat ini
        split_step = kwargs.get('split_step', '')  # Step dalam split saat ini
        step = kwargs.get('step', 0)            # 0=persiapan, 1=proses, 2=finalisasi
        total_files_all = kwargs.get('total_files_all', 0)  # Total file di semua split
        
        # Tracking split yang sedang diproses untuk progress yang akurat
        if split and split != state['current_split']:
            state['current_split'] = split
            # Log pergantian split (selalu ditampilkan)
            if logger:
                logger.info(f"{ICONS['processing']} Memulai proses split {split}")
            
            with ui_components['status']:
                display(create_status_indicator('info', f"{ICONS['processing']} Memulai proses split {split}"))
        
        # Update informasi split untuk perhitungan progress agregat
        if split and total is not None and split not in state['splits']:
            state['splits'][split] = {'total': total, 'progress': 0, 'weight': 1.0}
            # Update total file dengan estimasi yang lebih akurat
            state['total_files'] = total_files_all if total_files_all > 0 else state.get('total_files', 0) + total
        
        # Update progress untuk split tertentu dengan normalisasi
        if split and progress is not None and total is not None and split in state['splits']:
            # Dapatkan delta progress yang ternormalisasi
            old_progress = state['splits'][split].get('progress', 0)
            # Normalisasi progress untuk mencegah melebihi total
            normalized_progress = min(progress, total)
            delta_progress = max(0, normalized_progress - old_progress)  # Cegah delta negatif
            
            # Update state dengan progress ternormalisasi
            if delta_progress > 0 and total > 0:
                state['splits'][split]['progress'] = normalized_progress
                # Hitung kontribusi split ini ke total berdasarkan proporsi
                split_contribution = min(delta_progress / total, 1.0) * state['splits'][split]['weight']
                split_proportion = split_contribution * state['total_files'] / len(state['splits'])
                state['processed_files'] += split_proportion
        
        # Normalisasi overall progress untuk tidak melebihi 100%
        total_files = total_files_all or state['total_files']
        overall_progress = min(state['processed_files'], total_files)
        overall_total = max(total_files, 1)  # Hindari division by zero
        
        # Update progress bar dengan throttling
        if current_time - state['last_update'] >= update_interval:
            # Update overall progress bar
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].max = overall_total
                ui_components['progress_bar'].value = overall_progress
                
                # Persentase dengan normalisasi (max 100%)
                overall_percentage = min(int(overall_progress / overall_total * 100), 100)
                ui_components['progress_bar'].description = f"Total: {overall_percentage}%"
                ui_components['progress_bar'].layout.visibility = 'visible'
            
            # Update current progress bar (untuk split/step saat ini)
            if progress is not None and total is not None and 'current_progress' in ui_components:
                # Normalisasi progress untuk tidak melebihi total
                normalized_curr_progress = min(progress, total)
                ui_components['current_progress'].max = total
                ui_components['current_progress'].value = normalized_curr_progress
                
                # Format deskripsi yang lebih informatif
                current_percentage = min(int(normalized_curr_progress / total * 100), 100)
                
                # Tambahkan info step jika tersedia
                if split_step:
                    description = f"{split_step}: {current_percentage}%"
                elif split:
                    description = f"Split {split}: {current_percentage}%"
                else:
                    description = f"Progress: {current_percentage}%"
                    
                ui_components['current_progress'].description = description
                ui_components['current_progress'].layout.visibility = 'visible'
            
            # Update timestamp terakhir
            state['last_update'] = current_time
        
        # Logging pesan dengan throttling cerdas
        if message:
            # Tentukan apakah pesan ini penting (tidak di-throttle)
            is_important = (
                status != 'info' or                # Bukan log info biasa
                step == 0 or step == 2 or          # Tahap persiapan/finalisasi
                "selesai" in message.lower() or    # Pesan selesai proses
                "memulai" in message.lower() or    # Pesan memulai proses
                "error" in message.lower() or      # Pesan error
                "balancing" in message.lower() or  # Pesan penting lainnya
                message not in state['important_messages']  # Belum pernah ditampilkan
            )
            
            # Simpan pesan penting untuk hindari duplikasi
            if is_important:
                state['important_messages'].add(message)
            
            # Gunakan throttling hanya untuk pesan yang tidak penting
            time_to_log = current_time - state['last_message_update'] >= message_throttle
            
            # Log pesan penting atau sesuai interval throttling
            if is_important or time_to_log:
                # Log dengan logger jika tersedia
                if logger:
                    log_method = getattr(logger, status) if hasattr(logger, status) else logger.info
                    log_method(message)
                
                # Update UI jika status penting atau sesuai throttling
                if status != 'info' or is_important or time_to_log:
                    with ui_components['status']:
                        display(create_status_indicator(status, message))
                
                # Update timestamp terakhir
                state['last_message_update'] = current_time
        
        # Notifikasi observer untuk integrasi sistem lain (dengan throttling)
        if current_time - state['last_notify_update'] >= message_throttle:
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                
                # Tentukan tipe event berdasarkan konteks modul
                event_type_prefix = 'PREPROCESSING' if 'preprocessing_running' in ui_components else 'AUGMENTATION'
                sender = 'preprocessing_handler' if 'preprocessing_running' in ui_components else 'augmentation_handler'
                
                # Format pesan dengan informasi lengkap
                overall_percentage = min(int(overall_progress/overall_total*100), 100)
                notify_message = message or f"Progress {overall_percentage}%"
                
                # Tambahkan info split jika tersedia
                if split:
                    split_percentage = min(int(progress/total*100), 100) if progress is not None and total is not None else 0
                    notify_message = f"Split {split}: {split_percentage}% - {notify_message}"
                
                # Bersihkan kwargs dari parameter yang sudah kita gunakan
                clean_kwargs = {k: v for k, v in kwargs.items() 
                               if k not in ['progress', 'total', 'message', 'current_progress', 'current_total']}
                
                # Kirim notifikasi dengan data yang ternormalisasi
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
                
                # Update timestamp terakhir
                state['last_notify_update'] = current_time
            except ImportError:
                # Observer tidak tersedia, skip
                pass
    
    # Fungsi untuk register callback ke manager dengan metode yang tepat
    def register_progress_callback(manager) -> bool:
        """
        Register callback ke manager instance dengan metode yang tersedia.
        
        Args:
            manager: Manager instance (DatasetManager, AugmentationService, dll)
            
        Returns:
            Boolean menunjukkan keberhasilan registrasi
        """
        if not manager:
            return False
            
        try:
            # Coba register dengan metode standar
            if hasattr(manager, 'register_progress_callback'):
                manager.register_progress_callback(progress_callback)
                if logger: logger.debug(f"{ICONS['success']} Progress callback berhasil didaftarkan ke manager")
                return True
                
            # Fallback: Set callback langsung ke atribut
            elif hasattr(manager, '_progress_callback'):
                manager._progress_callback = progress_callback
                if logger: logger.debug(f"{ICONS['success']} Progress callback berhasil didaftarkan via atribut langsung")
                return True
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal mendaftarkan callback: {str(e)}")
            
        return False
    
    # Fungsi reset progress bar yang mengembalikan ke kondisi awal
    def reset_progress_bar() -> None:
        """Reset semua komponen progress ke kondisi awal."""
        # Reset UI components
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = 'Total:'
            ui_components['progress_bar'].layout.visibility = 'hidden'
            
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].description = 'Progress:'
            ui_components['current_progress'].layout.visibility = 'hidden'
        
        # Reset running flag
        process_key = 'preprocessing_running' if 'preprocessing_running' in ui_components else 'augmentation_running'
        ui_components[process_key] = False
        
        # Reset tracking state untuk progress baru
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