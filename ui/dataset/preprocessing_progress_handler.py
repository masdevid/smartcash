"""
File: smartcash/ui/dataset/preprocessing_progress_handler.py
Deskripsi: Handler progress tracking untuk preprocessing dataset dengan throttling log yang ditingkatkan
"""

from typing import Dict, Any
import time  # Pastikan import time ada di sini
from IPython.display import display
from smartcash.ui.utils.constants import ICONS

def setup_progress_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler progress tracking untuk preprocessing dengan throttling log yang ditingkatkan."""
    logger = ui_components.get('logger')
    
    # Simpan state tracking untuk batasi frekuensi update
    last_update = {'overall': 0, 'current': 0, 'message': None, 'message_time': 0, 'notify_time': 0}
    update_interval = 0.5  # Hanya update UI setiap 0.5 detik
    message_throttle_interval = 2.0  # Hanya log pesan setiap 2 detik
    key_progress_points = {0, 25, 50, 75, 100}  # Titik progress penting untuk selalu dilog
    
    # Reset flag status untuk progress bar
    ui_components['preprocessing_running'] = False
    
    # Fungsi progress callback yang dioptimalkan dengan throttling yang lebih cerdas
    def progress_callback(progress=None, total=None, message=None, status='info', 
                         current_progress=None, current_total=None, **kwargs):
        """Progress callback dengan throttling log untuk mencegah flooding UI."""
        # Skip jika preprocessing sudah dihentikan
        if not ui_components.get('preprocessing_running', True): 
            return
            
        current_time = time.time()
        
        # Ekstrak metadata untuk step tracking
        step = kwargs.get('step', 0)  # 0=persiapan, 1=proses split, 2=finalisasi
        current_step = kwargs.get('current_step', 0)
        current_total_steps = kwargs.get('current_total', 1)
        
        # ===== UPDATE PROGRESS BAR UTAMA =====
        if progress is not None and total is not None and total > 0 and 'progress_bar' in ui_components:
            if current_time - last_update['overall'] >= update_interval:
                ui_components['progress_bar'].max = total
                ui_components['progress_bar'].value = min(progress, total)
                
                # Hitung persentase untuk progress bar utama
                percentage = int(progress / total * 100) if total > 0 else 0
                ui_components['progress_bar'].description = f"Overall: {percentage}%"
                ui_components['progress_bar'].layout.visibility = 'visible'
                
                # Update timestamp terakhir update overall
                last_update['overall'] = current_time
        
        # ===== UPDATE PROGRESS BAR LANGKAH SAAT INI =====
        if current_progress is not None and current_total is not None and current_total > 0 and 'current_progress' in ui_components:
            if current_time - last_update['current'] >= update_interval:
                ui_components['current_progress'].max = current_total
                ui_components['current_progress'].value = min(current_progress, current_total)
                
                # Format deskripsi step saat ini untuk better understanding
                step_descriptions = ["Persiapan", "Proses Split", "Finalisasi"]
                step_desc = step_descriptions[step] if step < len(step_descriptions) else f"Step {step}"
                step_progress = f"{current_step}/{current_total_steps}" if current_total_steps > 1 else ""
                
                # Update deskripsi progress current
                if current_total_steps > 1:
                    ui_components['current_progress'].description = f"{step_desc} {step_progress}: {int(current_progress/current_total*100)}%"
                else:
                    ui_components['current_progress'].description = f"{step_desc}: {int(current_progress/current_total*100)}%"
                    
                ui_components['current_progress'].layout.visibility = 'visible'
                
                # Update timestamp terakhir update current
                last_update['current'] = current_time
        
        # ===== UPDATE LOG UI (DENGAN THROTTLING) =====
        if message and 'status' in ui_components:
            # Cek apakah pesan baru atau penting (untuk menghindari duplikasi log)
            is_new_message = message != last_update.get('message')
            percent_complete = int(progress / total * 100) if progress is not None and total is not None and total > 0 else None
            is_key_progress = percent_complete in key_progress_points if percent_complete is not None else False
            time_to_log = current_time - last_update.get('message_time', 0) >= message_throttle_interval
            
            # Log pesan jika baru, penting, atau interval waktu cukup
            if is_new_message and (is_key_progress or time_to_log):
                from smartcash.ui.utils.alert_utils import create_status_indicator
                
                with ui_components['status']:
                    # Format pesan dengan emoji status yang sesuai
                    display(create_status_indicator(status, message))
                
                # Juga log ke logger jika tersedia
                if logger:
                    log_func = getattr(logger, status) if hasattr(logger, status) else logger.info
                    log_func(message)
                
                # Simpan pesan terakhir dan waktu log
                last_update['message'] = message
                last_update['message_time'] = current_time
        
        # ===== NOTIFY OBSERVER (DENGAN THROTTLING) =====
        try:
            # Hanya notifikasi pada poin progress penting atau interval minimum
            time_since_last_notify = current_time - last_update.get('notify_time', 0)
            should_notify = (
                is_key_progress or 
                time_since_last_notify >= message_throttle_interval or
                kwargs.get('force_notify', False)
            )
            
            if should_notify:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                
                notify(
                    event_type=EventTopics.PREPROCESSING_PROGRESS,
                    sender="preprocessing_handler",
                    message=message,
                    progress=progress,
                    total=total,
                    status=status,
                    **kwargs
                )
                
                last_update['notify_time'] = current_time
        except ImportError:
            pass
    
    # Fungsi untuk registrasi callback ke preprocessing_manager
    def register_progress_callback(preprocessing_manager):
        """Register callback progress ke preprocessing_manager."""
        if not preprocessing_manager: 
            return False
            
        # Coba register dengan berbagai metode yang mungkin tersedia
        try:
            # Pendekatan 1: register_progress_callback
            if hasattr(preprocessing_manager, 'register_progress_callback'):
                preprocessing_manager.register_progress_callback(progress_callback)
                if logger: logger.info(f"{ICONS['success']} Progress callback berhasil didaftarkan ke preprocessing manager")
                return True
                
            # Pendekatan 2: _progress_callback
            elif hasattr(preprocessing_manager, '_progress_callback'):
                preprocessing_manager._progress_callback = progress_callback
                if logger: logger.info(f"{ICONS['success']} Progress callback berhasil didaftarkan via atribut langsung")
                return True
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal mendaftarkan callback: {str(e)}")
            
        return False
    
    # Setup progress observer dengan batasan update
    try:
        from smartcash.components.observer.event_topics_observer import EventTopics
        from smartcash.ui.handlers.observer_handler import create_progress_observer
        
        # Buat observer untuk overall progress
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
        
        # Buat observer untuk current progress
        create_progress_observer(
            ui_components=ui_components,
            event_type=EventTopics.PREPROCESSING_CURRENT_PROGRESS,
            total=100,
            progress_widget_key='current_progress',
            update_output=False,  # Tidak perlu duplikasi output
            output_widget_key='status',
            observer_group='preprocessing_observers'
        )
        
        if logger: logger.info(f"{ICONS['success']} Progress tracking observer berhasil diinisialisasi")
    except (ImportError, AttributeError) as e:
        if logger: logger.debug(f"{ICONS['info']} Observer progress tidak tersedia: {str(e)}")
    
    # Helper untuk update progress bar dengan satu fungsi
    def update_progress_bar(progress, total, message=None):
        """Update progress bar dengan satu fungsi untuk kemudahan integrasi."""
        progress_callback(
            progress=progress, 
            total=total, 
            message=message or f"Progress: {int(progress/total*100) if total > 0 else 0}%",
            force_notify=True  # Selalu notifikasi untuk update eksplisit
        )
    
    # Helper untuk reset progress bar
    def reset_progress_bar():
        """Reset semua komponen progress ke nilai awal."""
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = 'Overall:'
            ui_components['progress_bar'].layout.visibility = 'hidden'
            
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].description = 'Current:'
            ui_components['current_progress'].layout.visibility = 'hidden'
        
        # Reset status flags
        ui_components['preprocessing_running'] = False
        
        # Reset timestamp-timestamp throttling
        last_update.clear()
        last_update.update({'overall': 0, 'current': 0, 'message': None, 'message_time': 0, 'notify_time': 0})
        
        # Log reset
        if logger: logger.debug(f"{ICONS['refresh']} Progress bar direset")
    
    # Tambahkan semua fungsi dan state ke UI components
    ui_components.update({
        'progress_callback': progress_callback,
        'register_progress_callback': register_progress_callback,
        'update_progress_bar': update_progress_bar,
        'reset_progress_bar': reset_progress_bar,
        'preprocessing_running': False
    })
    
    # Coba register callback jika preprocessing_manager sudah ada
    if 'preprocessing_manager' in ui_components or 'dataset_manager' in ui_components:
        manager = ui_components.get('preprocessing_manager') or ui_components.get('dataset_manager')
        register_progress_callback(manager)
    
    return ui_components