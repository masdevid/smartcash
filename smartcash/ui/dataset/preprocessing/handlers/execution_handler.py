"""
File: smartcash/ui/dataset/preprocessing/handlers/execution_handler.py
Deskripsi: Handler untuk eksekusi proses preprocessing
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output
import time
import traceback
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger

# Import handler terpisah untuk SRP
from smartcash.ui.dataset.preprocessing.handlers.status_handler import update_status_panel

logger = get_logger()

def run_preprocessing(ui_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Jalankan proses preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi tambahan (opsional)
        
    Returns:
        Dictionary hasil preprocessing
    """
    # Dapatkan logger jika tersedia
    logger = ui_components.get('logger', get_logger())
    
    # Tandai waktu mulai untuk perhitungan durasi
    start_time = time.time()
    
    try:
        # Validasi prasyarat preprocessing
        from smartcash.ui.dataset.preprocessing.handlers.initialization_handler import validate_preprocessing_prerequisites
        prerequisites = validate_preprocessing_prerequisites(ui_components)
        
        if not prerequisites.get('success', False):
            raise Exception(prerequisites.get('message', 'Validasi prasyarat gagal'))
        
        # Dapatkan dataset manager
        from smartcash.ui.dataset.preprocessing.handlers.button_handlers import get_dataset_manager
        dataset_manager = get_dataset_manager(ui_components, config, logger)
        
        if not dataset_manager:
            raise Exception("Gagal membuat dataset manager")
        
        # Dapatkan split dari prerequisites
        split = prerequisites.get('split')
        split_info = f"Split {split}" if split else "Semua split"
        
        # Update status di UI
        status = ui_components.get('status')
        if status:
            with status:
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS['processing']} Menjalankan preprocessing {split_info}..."))
        
        # Update status panel
        update_status_panel(ui_components, "info", f"{ICONS['processing']} Menjalankan preprocessing {split_info}...")
        
        # Notifikasi observer tentang mulai preprocessing
        try:
            from smartcash.components.observer import notify
            notify('preprocessing_start', {
                'split': split,
                'input_dir': prerequisites.get('input_dir'),
                'output_dir': prerequisites.get('output_dir'),
                'image_count': prerequisites.get('image_count', 0)
            })
        except ImportError:
            pass  # Observer tidak tersedia, lanjutkan
        
        # Dapatkan konfigurasi preprocessing
        preprocess_config = prerequisites.get('preprocess_config', {})
        
        # Log parameter yang akan digunakan
        logger.info(f"{ICONS['info']} Parameter preprocessing: ")
        for key, value in preprocess_config.items():
            logger.info(f"  - {key}: {value}")
        
        # Jalankan preprocessing
        result = dataset_manager.preprocess_dataset(
            split=split,
            **preprocess_config
        )
        
        # Tambahkan waktu eksekusi ke hasil
        execution_time = time.time() - start_time
        result['execution_time'] = execution_time
        
        # Tambahkan konfigurasi ke hasil
        result['config'] = preprocess_config
        
        # Notifikasi observer tentang selesai preprocessing
        try:
            from smartcash.components.observer import notify
            notify('preprocessing_complete', {
                'split': split,
                'result': result
            })
        except ImportError:
            pass  # Observer tidak tersedia, lanjutkan
        
        # Update status di UI
        if status:
            with status:
                clear_output(wait=True)
                display(create_status_indicator("success", f"{ICONS['success']} Preprocessing {split_info} selesai dalam {execution_time:.2f} detik"))
        
        # Update status panel
        update_status_panel(ui_components, "success", f"{ICONS['success']} Preprocessing {split_info} selesai dalam {execution_time:.2f} detik")
        
        return result
    except Exception as e:
        # Notifikasi observer tentang error preprocessing
        try:
            from smartcash.components.observer import notify
            notify('preprocessing_error', {
                'error': str(e)
            })
        except ImportError:
            pass  # Observer tidak tersedia, lanjutkan
        
        # Update status di UI
        if status:
            with status:
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS['error']} Error saat preprocessing: {str(e)}"))
                display(traceback.format_exc())
        
        # Update status panel
        update_status_panel(ui_components, "error", f"{ICONS['error']} Error saat preprocessing: {str(e)}")
        
        # Log error
        logger.error(f"{ICONS['error']} Error saat preprocessing: {str(e)}")
        
        # Re-raise exception untuk penanganan di level yang lebih tinggi
        raise e
