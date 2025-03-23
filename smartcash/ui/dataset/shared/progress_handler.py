"""
File: smartcash/ui/dataset/shared/progress_handler.py
Deskripsi: Utilitas shared untuk mengelola progress tracking pada modul dataset
"""

from typing import Dict, Any, Optional, Callable
import time
from IPython.display import display

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_progress_handler(ui_components: Dict[str, Any], module_name: str = "process") -> Dict[str, Any]:
    """
    Setup handler progress tracking yang dapat digunakan bersama.
    
    Args:
        ui_components: Dictionary komponen UI
        module_name: Nama modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Untuk batasi jumlah update log
    last_update_time = {'value': 0}
    log_interval = 1.0  # Interval minimum antar update log (detik)
    
    # State untuk tracking progress
    total_info = {'total_files': 0, 'processed_files': 0, 'current_item': '', 'items': {}}
    
    # Reset flag status
    ui_components[f'{module_name}_running'] = False
    
    def progress_callback(progress=None, total=None, message=None, status='info', 
                         current_progress=None, current_total=None, **kwargs):
        """
        Progress callback dengan throttling untuk mencegah flooding log dan UI.
        
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
        if not ui_components.get(f'{module_name}_running', True): return
        
        # Ekstrak metadata untuk progress tracking
        step = kwargs.get('step', 0)
        current_item = kwargs.get('current_item', kwargs.get('split', ''))
        
        # Update informasi untuk current_item jika berbeda
        if current_item and current_item != total_info['current_item']:
            total_info['current_item'] = current_item
            
            # Log pergantian item secara eksplisit
            if logger:
                logger.info(f"{ICONS['processing']} Memulai {module_name} item {current_item}")
            
            with ui_components['status']:
                display(create_status_indicator('info', f"{ICONS['processing']} Memulai {module_name} item {current_item}"))
        
        # ===== UPDATE PROGRESS STATISTICS =====
        if current_item and total is not None and current_item not in total_info['items']:
            total_info['items'][current_item] = {'total': total, 'progress': 0}
            total_info['total_files'] += total
            
        # Update progress untuk item ini
        if current_item and progress is not None and total is not None:
            if current_item in total_info['items']:
                old_progress = total_info['items'][current_item].get('progress', 0)
                if progress > old_progress:
                    total_info['processed_files'] += (progress - old_progress)
                    total_info['items'][current_item]['progress'] = progress
        
        # Hitung total progress
        overall_progress = total_info['processed_files']
        overall_total = max(total_info['total_files'], 1)  # Hindari division by zero
        
        # Batasi frekuensi update UI
        current_time = time.time()
        time_since_last_update = current_time - last_update_time['value']
        
        # Hanya update UI jika:
        # 1. Interval minimum terpenuhi, atau
        # 2. Status penting (error/warning), atau
        # 3. Perubahan signifikan (0%, 25%, 50%, 75%, 100%)
        is_status_event = status in ('error', 'warning') or kwargs.get('event_type', '').endswith(('_start', '_complete', '_error'))
        is_significant_progress = progress is not None and total is not None and (
            progress == 0 or progress == total or 
            (total >= 4 and progress % (total // 4) == 0)  # 0%, 25%, 50%, 75%, 100%
        )
        
        if is_status_event or is_significant_progress or time_since_last_update >= log_interval:
            # Update timestamp untuk log berikutnya
            last_update_time['value'] = current_time
            
            # ===== UPDATE PROGRESS BAR TOTAL =====
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].max = overall_total
                ui_components['progress_bar'].value = min(overall_progress, overall_total)
                
                # Hitung persentase
                overall_percentage = int(overall_progress / overall_total * 100) if overall_total > 0 else 0
                ui_components['progress_bar'].description = f"Total: {overall_percentage}%"
                ui_components['progress_bar'].layout.visibility = 'visible'
            
            # ===== UPDATE CURRENT PROGRESS BAR =====
            if progress is not None and total is not None and 'current_progress' in ui_components:
                ui_components['current_progress'].max = total
                ui_components['current_progress'].value = min(progress, total)
                
                # Format deskripsi
                current_percentage = int(progress / total * 100) if total > 0 else 0
                
                if current_item:
                    ui_components['current_progress'].description = f"Item {current_item}: {current_percentage}%"
                else:
                    ui_components['current_progress'].description = f"Progress: {current_percentage}%"
                    
                ui_components['current_progress'].layout.visibility = 'visible'
            
            # Log jika ada message
            if message and logger:
                log_method = getattr(logger, status) if hasattr(logger, status) else logger.info
                log_method(message)
                
                # Display message di status jika status penting atau pergantian step
                if status != 'info' or step in (0, 2) or 'selesai' in message.lower():
                    with ui_components['status']:
                        display(create_status_indicator(status, message))
            
            # Notifikasi observer
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            
            # Map modul ke event type
            event_map = {
                'preprocessing': 'PREPROCESSING',
                'augmentation': 'AUGMENTATION'
            }
            event_base = event_map.get(module_name, 'PROCESS').upper()
            
            # Notify overall progress
            progress_event = getattr(EventTopics, f"{event_base}_PROGRESS")
            notify(
                event_type=progress_event,
                sender=f"{module_name}_handler",
                message=message or f"{module_name} progress: {int(overall_progress/overall_total*100) if overall_total > 0 else 0}%",
                progress=overall_progress,
                total=overall_total,
                **kwargs
            )
            
            # Notify current progress jika ada
            if current_item and progress is not None and total is not None:
                current_event = getattr(EventTopics, f"{event_base}_CURRENT_PROGRESS")
                notify(
                    event_type=current_event,
                    sender=f"{module_name}_handler",
                    message=f"{module_name} item {current_item}: {int(progress/total*100) if total > 0 else 0}%",
                    progress=progress,
                    total=total,
                    **kwargs
                )
    
    # Fungsi untuk registrasi callback ke manager
    def register_progress_callback(manager):
        """Register progress callback ke manager dengan deteksi method yang tersedia."""
        if not manager: return False
            
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
            
        return False
    
    # Setup observer
    from smartcash.components.observer.event_topics_observer import EventTopics
    from smartcash.ui.handlers.observer_handler import create_progress_observer
    
    # Map modul ke event type
    event_map = {
        'preprocessing': 'PREPROCESSING',
        'augmentation': 'AUGMENTATION'
    }
    event_base = event_map.get(module_name, 'PROCESS').upper()
    
    # Setup observer untuk overall progress
    progress_events = [
        getattr(EventTopics, f"{event_base}_PROGRESS"),
        getattr(EventTopics, f"{event_base}_START"),
        getattr(EventTopics, f"{event_base}_END"),
        getattr(EventTopics, f"{event_base}_ERROR")
    ]
    
    create_progress_observer(
        ui_components=ui_components,
        event_type=progress_events,
        total=100,
        progress_widget_key='progress_bar',
        output_widget_key='status',
        observer_group=f'{module_name}_observers'
    )
    
    # Setup observer untuk current progress
    create_progress_observer(
        ui_components=ui_components,
        event_type=getattr(EventTopics, f"{event_base}_CURRENT_PROGRESS"),
        total=100,
        progress_widget_key='current_progress',
        update_output=False,
        observer_group=f'{module_name}_observers'
    )
    
    if logger: logger.info(f"{ICONS['success']} Progress observer berhasil diinisialisasi")
    
    # Helper untuk reset progress
    def reset_progress_bar():
        """Reset semua komponen progress ke nilai awal."""
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = 'Total:'
            ui_components['progress_bar'].layout.visibility = 'hidden'
            
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].description = 'Current:'
            ui_components['current_progress'].layout.visibility = 'hidden'
        
        # Reset internal state
        ui_components[f'{module_name}_running'] = False
        total_info.clear()
        total_info.update({'total_files': 0, 'processed_files': 0, 'current_item': '', 'items': {}})
        last_update_time['value'] = 0
    
    # Tambahkan semua fungsi dan state ke UI components
    ui_components.update({
        'progress_callback': progress_callback,
        'register_progress_callback': register_progress_callback,
        'reset_progress_bar': reset_progress_bar,
        f'{module_name}_running': False
    })
    
    # Coba register callback jika manager sudah ada
    manager_keys = [f'{module_name}_manager', 'dataset_manager', 'augmentation_manager']
    for key in manager_keys:
        if key in ui_components:
            register_progress_callback(ui_components[key])
            break
    
    return ui_components