"""
File: smartcash/ui/dataset/preprocessing_progress_handler.py
Deskripsi: Handler progress tracking untuk preprocessing dataset dengan integrasi observer standard
"""

from typing import Dict, Any
import time
from IPython.display import display
from smartcash.ui.utils.constants import ICONS

def setup_progress_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler progress tracking untuk preprocessing dengan integrasi observer standar."""
    logger = ui_components.get('logger')
    
    # Filter untuk batasi jumlah update log
    last_log_time = {'value': 0}
    log_interval = 1.0  # Minimal interval antar log dalam detik
    
    # Fungsi progress callback terkonsolidasi yang lebih efisien
    def progress_callback(progress=None, total=None, message=None, status='info', 
                        current_progress=None, current_total=None, **kwargs):
        """
        Progress callback dengan pembatasan update log untuk mencegah masalah performa.
        
        Args:
            progress: Nilai progress utama
            total: Total maksimum progress utama
            message: Pesan progress 
            status: Status progress ('info', 'success', 'warning', 'error')
            current_progress: Nilai progress saat ini (substep)
            current_total: Total maksimum progress saat ini
            **kwargs: Parameter lain yang diteruskan dari caller
        """
        # Skip jika preprocessing sudah dihentikan
        if not ui_components.get('preprocessing_running', True): return
        
        # Update progress bar utama dengan validasi
        if progress is not None and total is not None and total > 0 and 'progress_bar' in ui_components:
            ui_components['progress_bar'].max = total
            ui_components['progress_bar'].value = min(progress, total)  # Ensure progress <= total
            ui_components['progress_bar'].description = f"{int(progress/total*100) if total > 0 else 0}%"
            ui_components['progress_bar'].layout.visibility = 'visible'
        
        # Update current progress jika tersedia dengan validasi
        if current_progress is not None and current_total is not None and current_total > 0 and 'current_progress' in ui_components:
            ui_components['current_progress'].max = current_total
            ui_components['current_progress'].value = min(current_progress, current_total)
            ui_components['current_progress'].description = f"Step: {current_progress}/{current_total}"
            ui_components['current_progress'].layout.visibility = 'visible'
        
        # Batasi notifikasi observer untuk mencegah flooding log
        current_time = time.time()
        time_since_last_log = current_time - last_log_time['value']
        
        # Proses notifikasi hanya jika: 
        # 1. Telah melewati interval minimum, atau
        # 2. Perubahan status signifikan (start/complete/error), atau
        # 3. Perubahan progress signifikan (0%, 25%, 50%, 75%, 100%)
        is_status_event = status in ('error', 'warning') or kwargs.get('event_type', '').endswith(('_start', '_complete', '_error'))
        is_significant_progress = progress is not None and total is not None and (
            progress == 0 or progress == total or 
            (total >= 4 and progress % (total // 4) == 0)  # 0%, 25%, 50%, 75%, 100%
        )
        
        if is_status_event or is_significant_progress or time_since_last_log >= log_interval:
            # Update timestamp untuk log berikutnya
            last_log_time['value'] = current_time
            
            # Notifikasi observer dengan observer standar
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                
                notify(
                    event_type=EventTopics.PREPROCESSING_PROGRESS, 
                    sender="preprocessing_handler",
                    message=message or f"Preprocessing progress: {int(progress/total*100) if progress is not None and total is not None and total > 0 else 0}%",
                    progress=progress,
                    total=total,
                    status=status,
                    **kwargs
                )
            except ImportError:
                pass
    
    # Fungsi untuk registrasi callback ke preprocessing_manager dengan validasi
    def register_progress_callback(preprocessing_manager):
        """Register callback progress ke preprocessing_manager."""
        if not preprocessing_manager or not hasattr(preprocessing_manager, 'register_progress_callback'): 
            return False
        
        # Register callback ke preprocessing_manager
        preprocessing_manager.register_progress_callback(progress_callback)
        return True
    
    # Setup observer integrasi full jika tersedia (dengan batasan update)
    try:
        from smartcash.components.observer.event_topics_observer import EventTopics
        from smartcash.ui.handlers.observer_handler import create_progress_observer
        
        # Setup progress observer untuk progress utama
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
        
        # Setup progress observer untuk current progress
        create_progress_observer(
            ui_components=ui_components,
            event_type=EventTopics.PREPROCESSING_CURRENT_PROGRESS,
            total=100,
            progress_widget_key='current_progress',
            update_output=False,
            observer_group='preprocessing_observers'
        )
        
        if logger: logger.info(f"{ICONS['success']} Progress tracking terintegrasi berhasil setup")
    except (ImportError, AttributeError) as e:
        if logger: logger.debug(f"{ICONS['info']} Observer progress tidak tersedia: {str(e)}")
    
    # Helper utility untuk memudahkan update progress
    def update_progress_bar(progress, total, message=None):
        """
        Update progress bar dengan satu fungsi.
        
        Args:
            progress: Nilai progress saat ini
            total: Nilai maksimal progress
            message: Pesan opsional
        """
        # Skip jika preprocessing sudah dihentikan
        if not ui_components.get('preprocessing_running', True): return
        
        # Update progress bar
        if 'progress_bar' in ui_components and total > 0:
            ui_components['progress_bar'].max = total
            ui_components['progress_bar'].value = min(progress, total)
            percentage = int((progress / total) * 100) if total > 0 else 0
            ui_components['progress_bar'].description = f"Progress: {percentage}%"
            ui_components['progress_bar'].layout.visibility = 'visible'
        
        # Update message jika ada
        if message and 'current_progress' in ui_components:
            ui_components['current_progress'].description = message
            ui_components['current_progress'].layout.visibility = 'visible'
            
        # Notify observer dengan batasan update untuk mencegah flooding log
        current_time = time.time()
        time_since_last_log = current_time - last_log_time['value']
        
        # Hanya update pada interval tertentu atau perubahan signifikan
        if time_since_last_log >= log_interval or progress == 0 or progress == total or (total >= 4 and progress % (total // 4) == 0):
            last_log_time['value'] = current_time
            
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.PREPROCESSING_PROGRESS, 
                    sender="preprocessing_handler",
                    message=message or f"Preprocessing progress: {percentage}%",
                    progress=progress,
                    total=total
                )
            except ImportError:
                pass
    
    # Helper untuk reset progress
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
        
        # Reset timestamp log
        last_log_time['value'] = 0
    
    # Tambahkan fungsi progress dan register ke UI components
    ui_components.update({
        'progress_callback': progress_callback,
        'register_progress_callback': register_progress_callback,
        'update_progress_bar': update_progress_bar,
        'reset_progress_bar': reset_progress_bar,
        'preprocessing_running': False  # Flag untuk menandai status preprocessing
    })
    
    # Registrasi langsung jika preprocessing_manager sudah ada
    if 'preprocessing_manager' in ui_components:
        register_progress_callback(ui_components['preprocessing_manager'])
    
    return ui_components