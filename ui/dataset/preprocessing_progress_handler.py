"""
File: smartcash/ui/dataset/preprocessing_progress_handler.py
Deskripsi: Handler progress tracking untuk preprocessing dataset dengan penghitungan progress yang ditingkatkan
"""

from typing import Dict, Any
import time
from IPython.display import display
from smartcash.ui.utils.constants import ICONS

def setup_progress_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler progress tracking untuk preprocessing dengan penghitungan yang ditingkatkan."""
    logger = ui_components.get('logger')
    
    # Simpan state tracking untuk batasi frekuensi update dan untuk perhitungan total progress
    last_update = {'time': 0, 'message_time': 0, 'notify_time': 0}
    update_interval = 0.5  # Hanya update UI setiap 0.5 detik
    message_throttle_interval = 2.0  # Hanya log pesan setiap 2 detik
    
    # Informasi jumlah file untuk penghitungan progress total
    total_info = {'total_files': 0, 'processed_files': 0, 'current_split': '', 'splits': {}}
    
    # Reset flag status untuk progress bar
    ui_components['preprocessing_running'] = False
    
    # Fungsi progress callback yang dioptimalkan dengan throttling log dan manajemen perhitungan
    def progress_callback(progress=None, total=None, message=None, status='info', 
                         current_progress=None, current_total=None, **kwargs):
        """
        Progress callback dengan perhitungan yang ditingkatkan dan throttling log untuk mencegah flooding UI.
        
        Args:
            progress: Nilai progress saat ini untuk split tertentu
            total: Total maksimum progress untuk split tertentu
            message: Pesan progress 
            status: Status progress ('info', 'success', 'warning', 'error')
            current_progress: Nilai progress saat ini (substep)
            current_total: Total maksimum progress saat ini (substep)
            **kwargs: Parameter lain yang diteruskan dari caller
        """
        # Skip jika preprocessing sudah dihentikan
        if not ui_components.get('preprocessing_running', True): 
            return
            
        current_time = time.time()
        
        # Ekstrak metadata untuk step tracking dan perhitungan total
        step = kwargs.get('step', 0)  # 0=persiapan, 1=proses split, 2=finalisasi
        current_step = kwargs.get('current_step', 0)
        current_total_steps = kwargs.get('current_total', 1)
        split = kwargs.get('split', '')
        
        # Update informasi split saat ini jika berbeda
        if split and split != total_info['current_split']:
            total_info['current_split'] = split
            # Log pergantian split secara eksplisit
            if logger:
                logger.info(f"{ICONS['processing']} Memulai preprocessing split {split}")
            
            with ui_components['status']:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                display(create_status_indicator('info', f"{ICONS['processing']} Memulai preprocessing split {split}"))
        
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
                
                # Format deskripsi split saat ini untuk better understanding
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
                from smartcash.ui.utils.alert_utils import create_status_indicator
                
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
                
                # Update overall progress untuk observer
                notify(
                    event_type=EventTopics.PREPROCESSING_PROGRESS,
                    sender="preprocessing_handler",
                    message=message or f"Preprocessing progress total: {int(overall_progress/overall_total*100)}%",
                    progress=overall_progress,
                    total=overall_total,
                    current_split=split,
                    **kwargs
                )
                
                # Update progress untuk split saat ini jika ada
                if split and progress is not None and total is not None:
                    notify(
                        event_type=EventTopics.PREPROCESSING_CURRENT_PROGRESS,
                        sender="preprocessing_handler",
                        message=f"Preprocessing split {split}: {int(progress/total*100)}%",
                        progress=progress,
                        total=total,
                        **kwargs
                    )
                
                last_update['notify_time'] = time.time()
            except ImportError:
                pass
    
    # Fungsi untuk registrasi callback ke preprocessing_manager
    def register_progress_callback(preprocessing_manager):
        """Register callback progress ke preprocessing_manager."""
        if not preprocessing_manager: 
            return False
            
        # Mencoba register dengan berbagai metode yang mungkin tersedia
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
        ui_components['preprocessing_running'] = False
        total_info.clear()
        total_info.update({'total_files': 0, 'processed_files': 0, 'current_split': '', 'splits': {}})
        
        # Reset timestamp throttling
        last_update.clear()
        last_update.update({'time': 0, 'message_time': 0, 'notify_time': 0})
        
        # Log reset
        if logger: logger.debug(f"{ICONS['refresh']} Progress bar direset")
    
    # Tambahkan semua fungsi dan state ke UI components
    ui_components.update({
        'progress_callback': progress_callback,
        'register_progress_callback': register_progress_callback,
        'reset_progress_bar': reset_progress_bar,
        'preprocessing_running': False
    })
    
    # Coba register callback jika preprocessing_manager sudah ada
    if 'preprocessing_manager' in ui_components or 'dataset_manager' in ui_components:
        manager = ui_components.get('preprocessing_manager') or ui_components.get('dataset_manager')
        register_progress_callback(manager)
    
    return ui_components