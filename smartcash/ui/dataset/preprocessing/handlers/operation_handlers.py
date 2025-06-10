"""
File: smartcash/ui/dataset/preprocessing/handlers/operation_handlers.py
Deskripsi: Handlers untuk operasi preprocessing (preprocess, check, cleanup)
"""
from typing import Dict, Any, Tuple, Optional, Callable
from smartcash.common.logger import get_logger

logger = get_logger('operation_handlers')

def get_operation_config(operation_type: str) -> Tuple[Optional[Callable], Optional[str], Optional[str]]:
    """Dapatkan konfigurasi operasi berdasarkan tipe"""
    operations = {
        'preprocess': (
            execute_preprocessing,
            'preprocess_button',
            'Preprocessing dataset...'
        ),
        'check': (
            check_dataset,
            'check_button',
            'Memeriksa dataset...'
        ),
        'cleanup': (
            cleanup_dataset,
            'cleanup_button',
            'Membersihkan dataset...'
        )
    }
    return operations.get(operation_type, (None, None, None))

def execute_operation(ui_components: Dict[str, Any], operation_type: str, config: Dict[str, Any]) -> Tuple[bool, str]:
    """Base operation handler dengan error handling"""
    try:
        # Dapatkan konfigurasi operasi
        operation_func, button_key, processing_msg = get_operation_config(operation_type)
        if not operation_func:
            raise ValueError(f"Operasi tidak valid: {operation_type}")
        
        # Setup progress tracker
        if 'setup_dual_progress_tracker' in ui_components:
            ui_components['setup_dual_progress_tracker'](ui_components, processing_msg)
        
        # Eksekusi operasi
        success, message = operation_func(ui_components, config)
        
        # Update UI berdasarkan hasil
        if success:
            if 'complete_progress_tracker' in ui_components:
                ui_components['complete_progress_tracker'](ui_components, message)
            if 'show_ui_success' in ui_components:
                ui_components['show_ui_success'](ui_components, message)
        else:
            if 'error_progress_tracker' in ui_components:
                ui_components['error_progress_tracker'](ui_components, message)
            if 'handle_ui_error' in ui_components:
                ui_components['handle_ui_error'](ui_components, message)
                
        return success, message
        
    except Exception as e:
        error_msg = f"Gagal menjalankan operasi {operation_type}: {str(e)}"
        # Log error without exc_info since SmartCashLogger doesn't support it
        logger.error(f"{error_msg}\n{str(e)}")
        if 'handle_ui_error' in ui_components:
            ui_components['handle_ui_error'](ui_components, error_msg)
        return False, error_msg

def execute_preprocessing(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """Execute dataset preprocessing"""
    try:
        # Validasi dataset
        if 'validate_dataset_ready' not in ui_components:
            return False, "Fungsi validasi dataset tidak tersedia"
            
        is_valid, message = ui_components['validate_dataset_ready'](config)
        if not is_valid:
            return False, f"Validasi dataset gagal: {message}"
        
        # Cek apakah sudah ada hasil preprocessing
        if 'check_preprocessed_exists' in ui_components:
            exists, message = ui_components['check_preprocessed_exists'](config)
            if exists:
                return True, f"Dataset sudah diproses sebelumnya: {message}"
        
        # Buat preprocessor
        if 'create_backend_preprocessor' not in ui_components:
            return False, "Fungsi pembuatan preprocessor tidak tersedia"
            
        preprocessor = ui_components['create_backend_preprocessor'](ui_components)
        if not preprocessor:
            return False, "Gagal membuat preprocessor"
        
        # Konversi konfigurasi UI ke format backend
        if '_convert_ui_to_backend_config' not in ui_components:
            return False, "Fungsi konversi konfigurasi tidak tersedia"
            
        backend_config = ui_components['_convert_ui_to_backend_config'](ui_components)
        
        # Jalankan preprocessing
        result = preprocessor.preprocess(
            config=backend_config,
            progress_callback=ui_components.get('progress_callback')
        )
        
        if result.get('success', False):
            return True, f"Preprocessing berhasil: {result.get('message', 'Selesai')}"
        else:
            error_msg = result.get('message', 'Unknown error')
            return False, f"Preprocessing gagal: {error_msg}"
                
    except Exception as e:
        error_msg = str(e)
        # Log error without exc_info since SmartCashLogger doesn't support it
        logger.error(f"Error saat preprocessing: {error_msg}\n{str(e)}")
        return False, f"Terjadi kesalahan saat preprocessing: {error_msg}"

def check_dataset(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """Check dataset validity"""
    try:
        if 'create_backend_checker' not in ui_components:
            return False, "Fungsi pembuat checker tidak tersedia"
            
        checker = ui_components['create_backend_checker'](config)
        if not checker:
            return False, "Gagal membuat dataset checker"
        
        # Validasi dataset
        is_valid, message = checker.validate()
        return is_valid, message
        
    except Exception as e:
        error_msg = f"Error saat memeriksa dataset: {str(e)}"
        # Log error without exc_info since SmartCashLogger doesn't support it
        logger.error(f"{error_msg}\n{str(e)}")
        return False, error_msg

def cleanup_dataset(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """Cleanup preprocessed dataset"""
    try:
        if 'create_backend_cleanup_service' not in ui_components:
            return False, "Fungsi pembuat layanan cleanup tidak tersedia"
            
        cleanup_service = ui_components['create_backend_cleanup_service'](config)
        if not cleanup_service:
            return False, "Gagal membuat cleanup service"
        
        # Jalankan cleanup dengan config
        result = cleanup_service.cleanup(config)
        success = result.get('success', False)
        message = result.get('message', 'Tidak ada pesan')
        return success, message
        
    except Exception as e:
        error_msg = f"Error saat membersihkan dataset: {str(e)}"
        # Log error without exc_info since SmartCashLogger doesn't support it
        logger.error(f"{error_msg}\n{str(e)}")
        return False, error_msg
