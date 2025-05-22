"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Handler utama untuk proses augmentasi dataset tanpa threading (Colab compatible)
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message
from smartcash.ui.dataset.augmentation.utils.validation_utils import validate_augmentation_parameters
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import (
    update_ui_before_augmentation, reset_ui_after_augmentation, 
    is_augmentation_running, set_augmentation_state, show_confirmation
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
    
    # Validasi parameter
    validation_result = validate_augmentation_parameters(ui_components)
    if not validation_result['valid']:
        log_message(ui_components, f"❌ {validation_result['message']}", "error")
        return
    
    # Tampilkan konfirmasi
    show_augmentation_confirmation(ui_components, validation_result['params'])

def show_augmentation_confirmation(ui_components: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Tampilkan konfirmasi augmentasi."""
    aug_types = params.get('types', ['combined'])
    split = params.get('split_target', 'train')
    
    message = f"Anda akan menjalankan augmentasi {', '.join(aug_types)} pada dataset {split}. "
    message += f"Akan menghasilkan {params.get('num_variations', 2)} variasi per gambar. Lanjutkan?"
    
    def on_confirm():
        log_message(ui_components, "✅ Konfirmasi augmentasi diterima", "info")
        execute_augmentation_process(ui_components, params)
    
    def on_cancel():
        log_message(ui_components, "❌ Augmentasi dibatalkan", "info")
    
    show_confirmation(ui_components, "Konfirmasi Augmentasi Dataset", message, on_confirm, on_cancel)

def execute_augmentation_process(ui_components: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Eksekusi proses augmentasi secara synchronous (tanpa threading).
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter augmentasi yang sudah divalidasi
    """
    logger = ui_components.get('logger', get_logger(AUGMENTATION_LOGGER_NAMESPACE))
    
    # Set state running
    set_augmentation_state(ui_components, True)
    update_ui_before_augmentation(ui_components)
    
    try:
        # Start progress tracking
        start_progress(ui_components, "🔄 Memulai augmentasi dataset...")
        
        # Jalankan augmentasi secara synchronous
        result = _run_augmentation_sync(ui_components, params)
        
        # Handle hasil
        _handle_augmentation_result(ui_components, result)
        
    except Exception as e:
        log_message(ui_components, f"❌ Error augmentasi: {str(e)}", "error")
        logger.error(f"🔥 Error augmentasi: {str(e)}")
        
        # Reset UI pada error
        reset_ui_after_augmentation(ui_components)
    finally:
        # Pastikan state direset
        set_augmentation_state(ui_components, False)

def _run_augmentation_sync(ui_components: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Jalankan augmentasi secara synchronous dengan progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter augmentasi
        
    Returns:
        Dictionary hasil augmentasi
    """
    # Import augmentation service
    from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
    
    # Buat service instance
    service = AugmentationService()
    
    # Setup progress callback
    def progress_callback(current: int, total: int, message: str = ""):
        if ui_components.get('stop_requested', False):
            return False  # Signal untuk stop
        
        # Update progress UI
        progress_percentage = int((current / total) * 100) if total > 0 else 0
        update_progress(ui_components, progress_percentage, message)
        
        return True
    
    # Jalankan augmentasi dengan progress callback
    try:
        log_message(ui_components, f"🚀 Memulai augmentasi dengan parameter: {params}", "info")
        
        # Panggil service augmentasi
        result = service.augment_dataset(
            data_dir=params.get('data_dir', 'data'),
            split=params.get('split_target', 'train'),
            types=params.get('types', ['combined']),
            num_variations=params.get('num_variations', 2),
            target_count=params.get('target_count', 1000),
            output_prefix=params.get('output_prefix', 'aug'),
            balance_classes=params.get('balance_classes', False),
            validate_results=params.get('validate_results', True),
            progress_callback=progress_callback,
            create_symlinks=True  # Selalu buat symlinks
        )
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error saat augmentasi: {str(e)}',
            'generated_images': 0
        }

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
        
        # Tampilkan tombol cleanup
        if 'cleanup_button' in ui_components and hasattr(ui_components['cleanup_button'], 'layout'):
            ui_components['cleanup_button'].layout.display = 'block'
        
    elif result.get('status') == 'cancelled':
        log_message(ui_components, "⏹️ Augmentasi dibatalkan", "warning")
        
    else:
        error_msg = result.get('message', 'Augmentasi gagal')
        log_message(ui_components, f"❌ {error_msg}", "error")
    
    # Reset UI setelah selesai
    reset_ui_after_augmentation(ui_components)

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
    for bool_field in ['balance_classes', 'validate_results']:
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