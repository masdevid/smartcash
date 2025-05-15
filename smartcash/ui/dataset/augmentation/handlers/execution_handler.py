"""
File: smartcash/ui/dataset/augmentation/handlers/execution_handler.py
Deskripsi: Handler untuk eksekusi proses augmentasi
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output
import time
import traceback
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger

# Import handler terpisah untuk SRP
from smartcash.ui.dataset.augmentation.handlers.observer_handler import (
    notify_process_start, notify_process_complete, notify_process_error
)
from smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler import (
    get_augmentation_service, get_augmentation_config
)
from smartcash.ui.dataset.augmentation.handlers.initialization_handler import (
    validate_augmentation_prerequisites
)

logger = get_logger("execution_handler")

def run_augmentation(ui_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Jalankan proses augmentasi dataset.

    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi tambahan (opsional)

    Returns:
        Dictionary hasil augmentasi
    """
    # Dapatkan logger jika tersedia
    logger = ui_components.get('logger', get_logger('augmentation_execution'))

    # Tandai waktu mulai untuk perhitungan durasi
    start_time = time.time()

    try:
        # Dapatkan service
        service = get_augmentation_service(ui_components)

        # Dapatkan konfigurasi augmentasi
        aug_config = service.config

        # Notifikasi observer bahwa proses telah dimulai
        display_info = f"dengan {aug_config['num_per_image']} gambar per input"
        notify_process_start(ui_components, "augmentasi dataset", display_info)

        # Update status di UI
        status = ui_components.get('status')
        if status:
            with status:
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS['processing']} Menjalankan augmentasi dataset..."))

        # Jalankan augmentasi
        result = service.run_augmentation()

        # Tambahkan konfigurasi ke hasil
        result['config'] = aug_config

        # Tambahkan waktu eksekusi ke hasil
        execution_time = time.time() - start_time
        result['execution_time'] = execution_time

        # Notifikasi observer bahwa proses telah selesai
        notify_process_complete(ui_components, result, display_info)

        # Update status di UI
        if status:
            with status:
                clear_output(wait=True)
                display(create_status_indicator("success", f"{ICONS['success']} Augmentasi selesai dalam {execution_time:.2f} detik"))

        return result
        
        # Cek hasil augmentasi
        if result['status'] == 'success':
            logger.info(f"{ICONS['success']} Augmentasi berhasil dijalankan")
            
            # Tambahkan path output jika tidak ada
            if 'output_dir' not in result:
                result['output_dir'] = directories['output_dir']
            
            # Update status panel
            update_status_panel(ui_components, "success", 
                f"{ICONS['success']} Augmentasi dataset berhasil diselesaikan")
            
            # Notifikasi observer tentang selesai
            notify_process_complete(ui_components, result, split_info)
            
            return {
                'status': 'success',
                'message': f"Augmentasi {split_info} berhasil diselesaikan",
                'result': result
            }
        else:
            error_message = result.get('message', 'Terjadi kesalahan saat augmentasi')
            logger.error(f"{ICONS['error']} {error_message}")
            
            # Update status panel
            update_status_panel(ui_components, "error", 
                f"{ICONS['error']} Augmentasi gagal: {error_message}")
            
            # Notifikasi observer tentang error
            notify_process_error(ui_components, error_message)
            
            return {
                'status': 'error',
                'message': error_message
            }
    
    except Exception as e:
        error_message = str(e)
        logger.error(f"{ICONS['error']} Error saat augmentasi dataset: {error_message}")
        
        # Update status panel
        update_status_panel(ui_components, "error", 
            f"{ICONS['error']} Augmentasi gagal: {error_message}")
        
        # Notifikasi observer tentang error
        notify_process_error(ui_components, error_message)
        
        return {
            'status': 'error',
            'message': error_message
        }
