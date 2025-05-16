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
    logger.info(f"{ICONS['info']} Memulai proses augmentasi dataset")

    # Tandai waktu mulai untuk perhitungan durasi
    start_time = time.time()

    try:
        # Validasi prasyarat augmentasi
        validation_result = validate_augmentation_prerequisites(ui_components)
        if not validation_result['valid']:
            error_message = validation_result.get('message', 'Prasyarat augmentasi tidak terpenuhi')
            logger.error(f"{ICONS['error']} {error_message}")
            
            # Update status di UI
            status = ui_components.get('status')
            if status:
                with status:
                    clear_output(wait=True)
                    display(create_status_indicator("error", f"{ICONS['error']} {error_message}"))
            
            return {
                'status': 'error',
                'message': error_message
            }
        
        # Dapatkan service
        logger.info(f"{ICONS['info']} Menginisialisasi augmentation service")
        service = get_augmentation_service(ui_components)
        if not service:
            error_message = "Gagal menginisialisasi augmentation service"
            logger.error(f"{ICONS['error']} {error_message}")
            return {
                'status': 'error',
                'message': error_message
            }

        # Dapatkan konfigurasi augmentasi
        aug_config = get_augmentation_config(ui_components)
        if not aug_config or 'augmentation' not in aug_config:
            error_message = "Konfigurasi augmentasi tidak valid"
            logger.error(f"{ICONS['error']} {error_message}")
            return {
                'status': 'error',
                'message': error_message
            }
        
        # Ekstrak parameter augmentasi
        aug_params = aug_config.get('augmentation', {})
        
        # Register progress callback
        from smartcash.ui.dataset.augmentation.handlers.progress_handler import register_progress_callback
        register_progress_callback(ui_components, service)
        logger.info(f"{ICONS['info']} Progress callback berhasil terdaftar")

        # Notifikasi observer bahwa proses telah dimulai
        num_variations = aug_params.get('num_variations', 2)
        display_info = f"dengan {num_variations} variasi per gambar"
        notify_process_start(ui_components, "augmentasi dataset", display_info)

        # Update status di UI
        status = ui_components.get('status')
        if status:
            with status:
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS['processing']} Menjalankan augmentasi dataset..."))
        
        # Tampilkan progress bar dan label
        for element in ['progress_bar', 'current_progress', 'overall_label', 'step_label']:
            if element in ui_components and hasattr(ui_components[element], 'layout'):
                ui_components[element].layout.visibility = 'visible'

        # Jalankan augmentasi dengan parameter dari konfigurasi
        logger.info(f"{ICONS['info']} Menjalankan augmentasi dengan parameter: {aug_params}")
        from smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler import execute_augmentation
        
        # Ekstrak parameter untuk augmentasi
        params = {
            'split': aug_params.get('split', 'train'),
            'augmentation_types': aug_params.get('types', ['combined']),
            'num_variations': aug_params.get('num_variations', 2),
            'output_prefix': aug_params.get('output_prefix', 'aug_'),
            'validate_results': aug_params.get('validate_results', True),
            'process_bboxes': aug_params.get('process_bboxes', True),
            'target_balance': aug_params.get('target_balance', True),
            'num_workers': aug_params.get('num_workers', 4),
            'move_to_preprocessed': aug_params.get('move_to_preprocessed', True),
            'target_count': aug_params.get('target_count', 1000)
        }
        
        # Jalankan augmentasi
        result = execute_augmentation(service, params)

        # Tambahkan konfigurasi ke hasil
        result['config'] = aug_config

        # Tambahkan waktu eksekusi ke hasil
        execution_time = time.time() - start_time
        result['execution_time'] = execution_time

        # Cek hasil augmentasi
        if result.get('status') == 'success':
            logger.info(f"{ICONS['success']} Augmentasi berhasil dijalankan dalam {execution_time:.2f} detik")
            
            # Notifikasi observer bahwa proses telah selesai
            notify_process_complete(ui_components, result, display_info)
            
            # Update status di UI
            if status:
                with status:
                    clear_output(wait=True)
                    display(create_status_indicator("success", f"{ICONS['success']} Augmentasi selesai dalam {execution_time:.2f} detik"))
            
            return result
        else:
            error_message = result.get('message', 'Terjadi kesalahan saat augmentasi')
            logger.error(f"{ICONS['error']} {error_message}")
            
            # Notifikasi observer tentang error
            notify_process_error(ui_components, error_message)
            
            # Update status di UI
            if status:
                with status:
                    clear_output(wait=True)
                    display(create_status_indicator("error", f"{ICONS['error']} Augmentasi gagal: {error_message}"))
            
            return {
                'status': 'error',
                'message': error_message
            }
    
    except Exception as e:
        error_message = str(e)
        logger.error(f"{ICONS['error']} Error saat augmentasi dataset: {error_message}")
        logger.error(traceback.format_exc())
        
        # Notifikasi observer tentang error
        notify_process_error(ui_components, error_message)
        
        # Update status di UI
        status = ui_components.get('status')
        if status:
            with status:
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS['error']} Augmentasi gagal: {error_message}"))
        
        return {
            'status': 'error',
            'message': error_message
        }
