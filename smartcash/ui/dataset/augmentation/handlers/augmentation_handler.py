"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Handler utama untuk proses augmentasi dataset dengan logger bridge yang diperbaiki
"""

import time
from typing import Dict, Any
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.dataset.augmentation.handlers.parameter_handler import extract_and_validate_parameters
from smartcash.ui.dataset.augmentation.handlers.symlink_handler import setup_augmentation_symlinks
from smartcash.ui.dataset.augmentation.handlers.progress_handler import ProgressHandler
from smartcash.ui.dataset.augmentation.handlers.confirmation_handler import show_augmentation_confirmation
from smartcash.ui.dataset.augmentation.handlers.result_handler import handle_augmentation_result
from smartcash.ui.dataset.augmentation.handlers.state_handler import StateHandler

def handle_augmentation_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol mulai augmentasi dengan logger bridge dan SRP handlers.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget yang diklik
    """
    # Setup logger bridge untuk komunikasi UI-Service
    ui_logger = create_ui_logger_bridge(ui_components, "augmentation")
    state_handler = StateHandler(ui_components, ui_logger)
    
    # Cek apakah sudah berjalan
    if state_handler.is_running():
        ui_logger.warning("⚠️ Augmentasi sedang berjalan")
        return
    
    # Reset state sebelum memulai
    state_handler.reset_signals()
    
    # Ekstrak dan validasi parameter
    is_valid, error_message, validated_params = extract_and_validate_parameters(ui_components, ui_logger)
    if not is_valid:
        ui_logger.error(f"❌ Validasi parameter gagal: {error_message}")
        return
    
    # Tampilkan konfirmasi
    show_augmentation_confirmation(ui_components, validated_params, ui_logger, 
                                 lambda: execute_augmentation_process(ui_components, validated_params, ui_logger))

def execute_augmentation_process(ui_components: Dict[str, Any], params: Dict[str, Any], ui_logger) -> None:
    """
    Eksekusi proses augmentasi dengan handlers terpisah.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter augmentasi yang sudah divalidasi
        ui_logger: UI Logger bridge
    """
    start_time = time.time()
    state_handler = StateHandler(ui_components, ui_logger)
    progress_handler = ProgressHandler(ui_components, ui_logger)
    
    try:
        # Set state running
        state_handler.set_running(True)
        
        # Setup symlink (5-10%)
        progress_handler.start_progress("🔗 Setup symlink untuk Google Drive...")
        symlink_success, symlink_message, symlink_info = setup_augmentation_symlinks(ui_components, params, ui_logger)
        
        if not symlink_success:
            result = {'status': 'error', 'message': f"Symlink setup gagal: {symlink_message}"}
            handle_augmentation_result(ui_components, result, time.time() - start_time, ui_logger)
            return
        
        progress_handler.update_progress(15, "✅ Symlink setup berhasil")
        
        # Update params dengan symlink info
        params.update(symlink_info)
        
        # Jalankan augmentasi dengan progress callback (15-95%)
        progress_handler.update_progress(20, "🚀 Memulai proses augmentasi...")
        
        # Setup progress callback untuk service
        service_callback = progress_handler.create_service_callback()
        
        # Jalankan augmentasi
        result = _run_augmentation_with_service(ui_components, params, service_callback, ui_logger)
        
        # Finalisasi (95-100%)
        progress_handler.update_progress(98, "🔄 Memproses hasil augmentasi...")
        
        # Handle hasil akhir
        duration = time.time() - start_time
        handle_augmentation_result(ui_components, result, duration, ui_logger)
        progress_handler.complete_progress("🎉 Augmentasi selesai!")
        
    except Exception as e:
        duration = time.time() - start_time
        error_result = {'status': 'error', 'message': f"Error augmentasi: {str(e)}"}
        
        progress_handler.complete_progress(f"❌ Error: {str(e)}", False)
        handle_augmentation_result(ui_components, error_result, duration, ui_logger)
        ui_logger.error(f"🔥 Error augmentasi: {str(e)}")
        
    finally:
        state_handler.set_running(False)

def _run_augmentation_with_service(ui_components: Dict[str, Any], params: Dict[str, Any], 
                                  progress_callback, ui_logger) -> Dict[str, Any]:
    """
    Jalankan augmentasi menggunakan service dengan progress callback.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter augmentasi
        progress_callback: Callback untuk progress tracking
        ui_logger: UI Logger bridge
        
    Returns:
        Dictionary hasil augmentasi
    """
    try:
        from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
        
        # Initialize service dengan UI callback
        service_config = {
            'data_dir': params.get('data_path', 'data'),
            'augmented_dir': f"{params.get('data_path', 'data')}/augmented",
            'num_workers': 1
        }
        
        # Pass UI components untuk progress reporting
        ui_components['progress_callback'] = progress_callback
        
        service = AugmentationService(
            config=service_config,
            data_dir=params.get('data_path', 'data'),
            num_workers=1,
            ui_components=ui_components
        )
        
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