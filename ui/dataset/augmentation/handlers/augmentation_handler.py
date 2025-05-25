"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Handler utama untuk proses augmentasi dataset dengan integrasi button state manager terbaru
"""

import time
from typing import Dict, Any
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.utils.button_state_manager import get_button_state_manager
from smartcash.ui.dataset.augmentation.handlers.parameter_handler import extract_and_validate_parameters
from smartcash.ui.dataset.augmentation.handlers.symlink_handler import setup_augmentation_symlinks
from smartcash.ui.dataset.augmentation.handlers.confirmation_handler import show_augmentation_confirmation
from smartcash.ui.dataset.augmentation.handlers.result_handler import handle_augmentation_result
from smartcash.ui.dataset.augmentation.handlers.state_handler import StateHandler

def handle_augmentation_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol mulai augmentasi dengan button state manager terbaru.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget yang diklik
    """
    # Setup logger dan state managers
    ui_logger = create_ui_logger_bridge(ui_components, "augmentation")
    button_state_manager = get_button_state_manager(ui_components)
    state_handler = StateHandler(ui_components, ui_logger)
    
    # Cek apakah operation bisa dimulai (shared button state manager)
    can_start, reason = button_state_manager.can_start_operation("augmentation")
    if not can_start:
        ui_logger.warning(f"⚠️ {reason}")
        return
    
    # Reset state sebelum memulai
    state_handler.reset_signals()
    
    # Ekstrak dan validasi parameter
    is_valid, error_message, validated_params = extract_and_validate_parameters(ui_components, ui_logger)
    if not is_valid:
        ui_logger.error(f"❌ Validasi parameter gagal: {error_message}")
        return
    
    # Tampilkan konfirmasi dengan shared confirmation dialog
    show_augmentation_confirmation(ui_components, validated_params, ui_logger, 
                                 lambda: execute_augmentation_process(ui_components, validated_params, ui_logger))

def execute_augmentation_process(ui_components: Dict[str, Any], params: Dict[str, Any], ui_logger) -> None:
    """
    Eksekusi proses augmentasi dengan shared button state manager.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter augmentasi yang sudah divalidasi
        ui_logger: UI Logger bridge
    """
    start_time = time.time()
    button_state_manager = get_button_state_manager(ui_components)
    
    # Gunakan context manager untuk disable semua buttons
    with button_state_manager.operation_context("augmentation"):
        try:
            ui_logger.info("🚀 Memulai proses augmentasi dengan shared components...")
            
            # Setup symlink (5-10%)
            _update_progress(ui_components, 5, "🔗 Setup symlink untuk Google Drive...")
            symlink_success, symlink_message, symlink_info = setup_augmentation_symlinks(ui_components, params, ui_logger)
            
            if not symlink_success:
                result = {'status': 'error', 'message': f"Symlink setup gagal: {symlink_message}"}
                handle_augmentation_result(ui_components, result, time.time() - start_time, ui_logger)
                return
            
            _update_progress(ui_components, 15, "✅ Symlink setup berhasil")
            
            # Update params dengan symlink info
            params.update(symlink_info)
            
            # Jalankan augmentasi dengan progress callback (15-95%)
            _update_progress(ui_components, 20, "🚀 Memulai proses augmentasi...")
            
            # Setup progress callback untuk service
            service_callback = _create_service_progress_callback(ui_components)
            
            # Jalankan augmentasi
            result = _run_augmentation_with_service(ui_components, params, service_callback, ui_logger)
            
            # Finalisasi (95-100%)
            _update_progress(ui_components, 98, "🔄 Memproses hasil augmentasi...")
            
            # Handle hasil akhir
            duration = time.time() - start_time
            handle_augmentation_result(ui_components, result, duration, ui_logger)
            _complete_progress(ui_components, "🎉 Augmentasi selesai!")
            
        except Exception as e:
            duration = time.time() - start_time
            error_result = {'status': 'error', 'message': f"Error augmentasi: {str(e)}"}
            
            _error_progress(ui_components, f"❌ Error: {str(e)}")
            handle_augmentation_result(ui_components, error_result, duration, ui_logger)
            ui_logger.error(f"🔥 Error augmentasi: {str(e)}")

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

def _create_service_progress_callback(ui_components: Dict[str, Any]):
    """Buat progress callback untuk service yang compatible dengan shared progress tracking."""
    def service_callback(current: int, total: int, message: str = "", **kwargs) -> bool:
        # Cek stop signal
        if ui_components.get('stop_requested', False):
            return False
        
        # Hitung progress percentage (15-95% range untuk augmentasi)
        if total > 0:
            # Map service progress (0-100) ke UI progress range (15-95)
            service_progress = min(100, (current / total) * 100)
            ui_progress = 15 + int(service_progress * 0.8)  # 15 + (0-100 * 0.8) = 15-95
            
            # Format message
            if message:
                display_message = f"{message} ({current}/{total})"
            else:
                display_message = f"Memproses: {current}/{total}"
            
            _update_progress(ui_components, ui_progress, display_message)
        
        return True  # Continue processing
    
    return service_callback

def _update_progress(ui_components: Dict[str, Any], value: int, message: str):
    """Update progress menggunakan shared progress tracking."""
    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
        ui_components['update_progress']('overall', value, message)
    elif 'tracker' in ui_components:
        ui_components['tracker'].update('overall', value, message)

def _complete_progress(ui_components: Dict[str, Any], message: str):
    """Complete progress menggunakan shared progress tracking."""
    if 'complete_operation' in ui_components and callable(ui_components['complete_operation']):
        ui_components['complete_operation'](message)
    elif 'tracker' in ui_components:
        ui_components['tracker'].complete(message)

def _error_progress(ui_components: Dict[str, Any], message: str):
    """Error progress menggunakan shared progress tracking."""
    if 'error_operation' in ui_components and callable(ui_components['error_operation']):
        ui_components['error_operation'](message)
    elif 'tracker' in ui_components:
        ui_components['tracker'].error(message)