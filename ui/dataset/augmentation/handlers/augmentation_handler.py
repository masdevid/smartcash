"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Handler utama untuk proses augmentasi dataset dengan validasi dan koordinasi
"""

import threading
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message
from smartcash.ui.dataset.augmentation.utils.validation_utils import validate_augmentation_parameters
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import (
    update_ui_before_augmentation, reset_ui_after_augmentation, 
    is_augmentation_running, set_augmentation_state
)
from smartcash.ui.dataset.augmentation.utils.progress_manager import (
    start_progress, update_progress, complete_progress
)

# Konstanta namespace logger
AUGMENTATION_LOGGER_NAMESPACE = "smartcash.dataset.augmentation"

def handle_augmentation_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol mulai augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget yang diklik
    """
    logger = ui_components.get('logger', get_logger(AUGMENTATION_LOGGER_NAMESPACE))
    
    # Cek apakah sudah berjalan
    if is_augmentation_running(ui_components):
        log_message(ui_components, "⚠️ Augmentasi sedang berjalan", "warning")
        return
    
    # Set state running
    set_augmentation_state(ui_components, True)
    
    # Update UI sebelum augmentasi
    update_ui_before_augmentation(ui_components)
    
    # Validasi parameter
    validation_result = validate_augmentation_parameters(ui_components)
    if not validation_result['valid']:
        log_message(ui_components, f"❌ {validation_result['message']}", "error")
        reset_ui_after_augmentation(ui_components)
        return
    
    # Tampilkan konfirmasi
    from smartcash.ui.dataset.augmentation.handlers.confirmation_handler import show_augmentation_confirmation
    show_augmentation_confirmation(ui_components, validation_result['params'])

def execute_augmentation_process(ui_components: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Eksekusi proses augmentasi dengan progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter augmentasi yang sudah divalidasi
    """
    logger = ui_components.get('logger', get_logger(AUGMENTATION_LOGGER_NAMESPACE))
    
    try:
        # Start progress tracking
        start_progress(ui_components, "🔄 Memulai augmentasi dataset...")
        
        # Jalankan di thread terpisah
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_augmentation_with_progress, ui_components, params)
            
            # Monitor progress
            while not future.done():
                if ui_components.get('stop_requested', False):
                    log_message(ui_components, "⏹️ Augmentasi dihentikan oleh pengguna", "warning")
                    future.cancel()
                    break
                threading.Event().wait(0.1)  # Check setiap 100ms
            
            # Dapatkan hasil
            if not future.cancelled():
                result = future.result()
                _handle_augmentation_result(ui_components, result)
    
    except Exception as e:
        log_message(ui_components, f"❌ Error augmentasi: {str(e)}", "error")
        logger.error(f"🔥 Error augmentasi: {str(e)}")
    finally:
        reset_ui_after_augmentation(ui_components)

def _run_augmentation_with_progress(ui_components: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Jalankan augmentasi dengan progress tracking detail.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter augmentasi
        
    Returns:
        Dictionary hasil augmentasi
    """
    from smartcash.ui.dataset.augmentation.handlers.augmentation_executor import AugmentationExecutor
    
    # Buat executor dengan progress callback
    executor = AugmentationExecutor(ui_components)
    
    # Register progress callback
    def progress_callback(current: int, total: int, message: str = ""):
        if ui_components.get('stop_requested', False):
            return False  # Signal untuk stop
        update_progress(ui_components, current, total, message)
        return True
    
    executor.set_progress_callback(progress_callback)
    
    # Jalankan augmentasi
    return executor.execute(params)

def _handle_augmentation_result(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Handle hasil augmentasi dan update UI.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil augmentasi
    """
    if result.get('status') == 'success':
        generated_count = result.get('generated_images', 0)
        complete_progress(ui_components, f"✅ Augmentasi selesai: {generated_count} gambar dihasilkan")
        log_message(ui_components, f"🎉 Augmentasi berhasil! {generated_count} gambar baru dibuat", "success")
        
        # Tampilkan opsi cleanup
        ui_components.get('cleanup_button', {}).layout.display = 'block'
        
    elif result.get('status') == 'cancelled':
        log_message(ui_components, "⏹️ Augmentasi dibatalkan", "warning")
        
    else:
        error_msg = result.get('message', 'Augmentasi gagal')
        log_message(ui_components, f"❌ {error_msg}", "error")

def get_augmentation_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ekstrak konfigurasi augmentasi dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi augmentasi
    """
    config = {}
    
    # Extract dari form fields
    for field_name in ['num_variations', 'target_count', 'output_prefix', 'split_target']:
        if field_name in ui_components and hasattr(ui_components[field_name], 'value'):
            config[field_name] = ui_components[field_name].value
    
    # Extract augmentation types
    if 'augmentation_types' in ui_components and hasattr(ui_components['augmentation_types'], 'value'):
        config['types'] = list(ui_components['augmentation_types'].value)
    
    # Extract boolean options
    for bool_field in ['balance_classes', 'move_to_preprocessed', 'validate_results']:
        if bool_field in ui_components and hasattr(ui_components[bool_field], 'value'):
            config[bool_field] = ui_components[bool_field].value
    
    return config

def validate_dataset_availability(ui_components: Dict[str, Any], split: str) -> Dict[str, Any]:
    """
    Validasi ketersediaan dataset untuk augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        split: Split dataset yang akan diaugmentasi
        
    Returns:
        Dictionary hasil validasi
    """
    import os
    
    data_dir = ui_components.get('data_dir', 'data')
    dataset_path = os.path.join(data_dir, 'preprocessed', split)
    
    if not os.path.exists(dataset_path):
        return {
            'valid': False,
            'message': f'Dataset {split} tidak ditemukan di {dataset_path}'
        }
    
    # Cek jumlah gambar dan label
    images_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')
    
    if not os.path.exists(images_path) or not os.listdir(images_path):
        return {
            'valid': False,
            'message': f'Tidak ada gambar di dataset {split}'
        }
    
    if not os.path.exists(labels_path) or not os.listdir(labels_path):
        return {
            'valid': False,
            'message': f'Tidak ada label di dataset {split}'
        }
    
    image_count = len([f for f in os.listdir(images_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    label_count = len([f for f in os.listdir(labels_path) 
                      if f.lower().endswith('.txt')])
    
    return {
        'valid': True,
        'message': f'Dataset {split} siap: {image_count} gambar, {label_count} label',
        'image_count': image_count,
        'label_count': label_count
    }