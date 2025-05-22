"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Handler utama untuk proses augmentasi dataset dengan integrasi yang diperbaiki
"""

import time
from typing import Dict, Any
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message
from smartcash.ui.dataset.augmentation.utils.parameter_extractor import get_parameter_extractor
from smartcash.ui.dataset.augmentation.utils.progress_coordinator import get_progress_coordinator
from smartcash.ui.dataset.augmentation.utils.symlink_setup_manager import get_symlink_setup_manager
from smartcash.ui.dataset.augmentation.utils.stop_signal_manager import get_stop_signal_manager
from smartcash.ui.dataset.augmentation.utils.result_formatter import get_result_formatter, AugmentationStatus
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import (
    update_ui_before_augmentation, reset_ui_after_augmentation, 
    show_confirmation, set_augmentation_state
)

# Konstanta namespace logger
AUGMENTATION_LOGGER_NAMESPACE = "smartcash.dataset.augmentation"

def handle_augmentation_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol mulai augmentasi dengan integrasi yang diperbaiki.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget yang diklik
    """
    logger = ui_components.get('logger', get_logger(AUGMENTATION_LOGGER_NAMESPACE))
    
    # Cek apakah sudah berjalan
    if ui_components.get('augmentation_running', False):
        log_message(ui_components, "⚠️ Augmentasi sedang berjalan", "warning")
        return
    
    # Reset stop signal sebelum memulai
    stop_manager = get_stop_signal_manager(ui_components)
    stop_manager.reset_stop_signal()
    
    # Ekstrak dan validasi parameter menggunakan consolidated extractor
    param_extractor = get_parameter_extractor(ui_components)
    is_valid, error_message, validated_params = param_extractor.validate_parameters()
    
    if not is_valid:
        log_message(ui_components, f"❌ Validasi parameter gagal: {error_message}", "error")
        return
    
    # Tampilkan konfirmasi
    show_augmentation_confirmation(ui_components, validated_params)

def show_augmentation_confirmation(ui_components: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Tampilkan konfirmasi augmentasi dengan parameter yang sudah divalidasi."""
    aug_types = params.get('types', ['combined'])
    split = params.get('split', 'train')
    target_count = params.get('target_count', 500)
    
    message = f"Augmentasi {', '.join(aug_types)} pada dataset {split}.\n"
    message += f"Target: {target_count} instance per kelas, {params.get('num_variations', 2)} variasi per gambar.\n"
    message += "Hasil akan otomatis tersimpan ke Google Drive jika symlink aktif. Lanjutkan?"
    
    def on_confirm():
        log_message(ui_components, "✅ Konfirmasi augmentasi diterima", "info")
        execute_augmentation_process(ui_components, params)
    
    def on_cancel():
        log_message(ui_components, "❌ Augmentasi dibatalkan", "info")
    
    show_confirmation(ui_components, "Konfirmasi Augmentasi Dataset", message, on_confirm, on_cancel)

def execute_augmentation_process(ui_components: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Eksekusi proses augmentasi secara synchronous dengan integrasi lengkap.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter augmentasi yang sudah divalidasi
    """
    logger = ui_components.get('logger', get_logger(AUGMENTATION_LOGGER_NAMESPACE))
    start_time = time.time()
    
    # Set state running
    set_augmentation_state(ui_components, True)
    update_ui_before_augmentation(ui_components)
    
    # Initialize managers
    progress_coordinator = get_progress_coordinator(ui_components)
    stop_manager = get_stop_signal_manager(ui_components)
    symlink_manager = get_symlink_setup_manager(ui_components)
    result_formatter = get_result_formatter()
    
    try:
        # Step 1: Setup symlink (5%)
        progress_coordinator.start_progress(100, "🔗 Setup symlink untuk Google Drive...")
        symlink_success, symlink_message, symlink_info = symlink_manager.setup_symlinks_for_augmentation()
        
        if not symlink_success:
            result = result_formatter.create_error_result(f"Symlink setup gagal: {symlink_message}")
            _handle_augmentation_result(ui_components, result, time.time() - start_time)
            return
        
        progress_coordinator.update_step("Symlink Setup", 10, "✅ Symlink setup berhasil")
        
        # Step 2: Verify paths (10%)
        split = params.get('split', 'train')
        path_success, path_message, path_info = symlink_manager.verify_augmentation_paths(split)
        
        if not path_success:
            result = result_formatter.create_error_result(f"Path verification gagal: {path_message}")
            _handle_augmentation_result(ui_components, result, time.time() - start_time)
            return
        
        progress_coordinator.update_step("Path Verification", 20, "✅ Path verification berhasil")
        
        # Step 3: Initialize service dan jalankan augmentasi (20-90%)
        progress_coordinator.update_step("Augmentasi", 25, "🚀 Memulai proses augmentasi...")
        
        # Update params dengan symlink info
        params.update({
            'uses_symlink': symlink_info.get('uses_symlink', False),
            'storage_type': symlink_info.get('storage_type', 'Local'),
            'data_path': symlink_info.get('data_path', 'data')
        })
        
        # Jalankan augmentasi dengan progress callback
        service_callback = progress_coordinator.create_service_callback()
        stop_callback = stop_manager.create_progress_callback_with_stop_check(service_callback)
        
        result = _run_augmentation_sync(ui_components, params, stop_callback)
        
        # Step 4: Process result (90-100%)
        progress_coordinator.update_step("Finalisasi", 95, "🔄 Memproses hasil augmentasi...")
        
        # Format result dengan info lengkap
        duration = time.time() - start_time
        formatted_result = result_formatter.format_service_result(
            result, duration, split, 
            params.get('storage_type', 'Local'),
            params.get('uses_symlink', False)
        )
        
        progress_coordinator.finish_progress("🎉 Augmentasi selesai!", formatted_result.status == AugmentationStatus.SUCCESS)
        
        # Handle hasil akhir
        _handle_augmentation_result(ui_components, formatted_result, duration)
        
    except Exception as e:
        duration = time.time() - start_time
        error_result = result_formatter.create_error_result(f"Error augmentasi: {str(e)}")
        
        progress_coordinator.finish_progress(f"❌ Error: {str(e)}", False)
        _handle_augmentation_result(ui_components, error_result, duration)
        
        logger.error(f"🔥 Error augmentasi: {str(e)}")
        
    finally:
        # Pastikan state direset
        set_augmentation_state(ui_components, False)

def _run_augmentation_sync(ui_components: Dict[str, Any], params: Dict[str, Any], progress_callback) -> Dict[str, Any]:
    """
    Jalankan augmentasi secara synchronous dengan balanced class manager.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter augmentasi
        progress_callback: Callback untuk progress tracking
        
    Returns:
        Dictionary hasil augmentasi
    """
    try:
        # Import service dan balanced class manager
        from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
        from smartcash.dataset.services.augmentor.balanced_class_manager import get_balanced_class_manager
        
        # Initialize service dengan config yang tepat
        service_config = {
            'data_dir': params.get('data_path', 'data'),
            'augmented_dir': f"{params.get('data_path', 'data')}/augmented",
            'num_workers': 1  # Synchronous processing untuk Colab
        }
        
        service = AugmentationService(
            config=service_config,
            data_dir=params.get('data_path', 'data'),
            logger=ui_components.get('logger'),
            num_workers=1,
            ui_components=ui_components
        )
        
        # Jika balance_classes enabled, gunakan balanced class manager
        if params.get('balance_classes', False):
            log_message(ui_components, "⚖️ Menggunakan balanced class manager (Layer 1 & 2)", "info")
            
            balanced_manager = get_balanced_class_manager(ui_components, ui_components.get('logger'))
            
            # Dapatkan list file untuk balancing
            import os
            split = params.get('split', 'train')
            images_dir = os.path.join(params.get('data_path', 'data'), 'preprocessed', split, 'images')
            labels_dir = os.path.join(params.get('data_path', 'data'), 'preprocessed', split, 'labels')
            
            image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Prepare balanced dataset
            balanced_data = balanced_manager.prepare_balanced_augmentation(
                image_files, labels_dir, params.get('target_count', 500)
            )
            
            # Override selected files dengan hasil balancing
            if balanced_data.get('selected_files'):
                log_message(ui_components, f"🎯 Balancing: {len(balanced_data['selected_files'])} file dipilih", "info")
            else:
                log_message(ui_components, "ℹ️ Tidak ada file yang perlu dibalance, menggunakan augmentasi umum", "info")
        
        # Jalankan augmentasi service
        result = service.augment_dataset(
            data_dir=params.get('data_path', 'data'),
            split=params.get('split', 'train'),
            types=params.get('types', ['combined']),
            num_variations=params.get('num_variations', 2),
            target_count=params.get('target_count', 500),
            output_prefix=params.get('output_prefix', 'aug_'),
            balance_classes=params.get('balance_classes', False),
            validate_results=params.get('validate_results', True),
            progress_callback=progress_callback,
            create_symlinks=True
        )
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error saat augmentasi: {str(e)}',
            'generated_images': 0,
            'processed': 0
        }

def _handle_augmentation_result(ui_components: Dict[str, Any], result, duration: float) -> None:
    """
    Handle hasil augmentasi dan update UI dengan format yang konsisten.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil augmentasi (AugmentationResult atau dict)
        duration: Durasi proses
    """
    # Convert ke format UI jika perlu
    if hasattr(result, 'status'):
        # AugmentationResult object
        ui_result = get_result_formatter().format_ui_result(result)
    else:
        # Legacy dict format
        from smartcash.ui.dataset.augmentation.utils.result_formatter import get_result_formatter
        formatter = get_result_formatter()
        formatted_result = formatter.format_service_result(result, duration)
        ui_result = formatter.format_ui_result(formatted_result)
    
    # Update UI berdasarkan status
    if ui_result['status'] == 'success':
        log_message(ui_components, ui_result['message'], "success")
        
        # Tampilkan tombol cleanup jika berhasil
        if 'cleanup_button' in ui_components and hasattr(ui_components['cleanup_button'], 'layout'):
            ui_components['cleanup_button'].layout.display = 'block'
        
    elif ui_result['status'] == 'cancelled':
        log_message(ui_components, ui_result['message'], "warning")
        
    else:
        log_message(ui_components, ui_result['message'], "error")
    
    # Reset UI setelah selesai
    reset_ui_after_augmentation(ui_components)
    
    # Log details untuk debugging
    details = ui_result.get('details', {})
    if details.get('generated_images', 0) > 0:
        log_message(ui_components, 
                   f"📊 Detail: {details['generated_images']} gambar, {details['success_rate']:.1f}% berhasil, {details['duration']:.1f}s",
                   "debug")

def get_augmentation_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ekstrak konfigurasi augmentasi dari UI components menggunakan parameter extractor.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi augmentasi
    """
    # Gunakan parameter extractor yang sudah dikonsolidasi
    extractor = get_parameter_extractor(ui_components)
    return extractor.extract_service_parameters()

def validate_dataset_availability(ui_components: Dict[str, Any], split: str) -> Dict[str, Any]:
    """
    Validasi ketersediaan dataset untuk augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        split: Split dataset yang akan diaugmentasi
        
    Returns:
        Dictionary hasil validasi
    """
    # Gunakan symlink manager untuk validasi path
    symlink_manager = get_symlink_setup_manager(ui_components)
    success, message, path_info = symlink_manager.verify_augmentation_paths(split)
    
    if success:
        return {
            'valid': True,
            'message': message,
            'paths': path_info
        }
    else:
        return {
            'valid': False,
            'message': message
        }