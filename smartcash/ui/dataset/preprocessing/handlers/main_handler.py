"""
File: smartcash/ui/dataset/preprocessing/handlers/main_handler.py
Deskripsi: Main handler untuk tombol preprocessing dengan integrasi service baru
"""

from typing import Dict, Any
from concurrent.futures import Future

from smartcash.common.logger import get_logger
from smartcash.ui.dataset.preprocessing.components.config_manager import get_config_from_ui
from smartcash.ui.dataset.preprocessing.utils.dialog_utils import create_preprocessing_confirmation_dialog
from smartcash.ui.dataset.preprocessing.services.service_runner import create_service_runner

logger = get_logger(__name__)

def handle_preprocessing_button_click(button: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol preprocessing utama.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Disable button untuk mencegah double click
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        # Reset stop flag
        ui_components['stop_requested'] = False
        
        # Get konfigurasi dari UI
        config = get_config_from_ui(ui_components)
        preprocessing_config = config.get('preprocessing', {})
        
        # Log konfigurasi
        logger.info("🔧 Mempersiapkan preprocessing dengan konfigurasi:")
        logger.info(f"  • Resolusi: {preprocessing_config.get('img_size', 'default')}")
        logger.info(f"  • Normalisasi: {preprocessing_config.get('normalization', 'minmax')}")
        logger.info(f"  • Split: {preprocessing_config.get('split', 'all')}")
        
        # Update status panel
        from smartcash.ui.utils.alert_utils import update_status_panel
        update_status_panel(ui_components['status_panel'], "⚙️ Mempersiapkan preprocessing...", "info")
        
        # Tampilkan dialog konfirmasi dengan existing data check
        create_preprocessing_confirmation_dialog(
            ui_components,
            preprocessing_config,
            lambda: _execute_preprocessing(ui_components, config),
            lambda: _cancel_preprocessing(ui_components, button)
        )
        
    except Exception as e:
        logger.error(f"❌ Error persiapan preprocessing: {str(e)}")
        _handle_preprocessing_error(ui_components, button, str(e))

def _execute_preprocessing(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Eksekusi preprocessing setelah konfirmasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi preprocessing
    """
    try:
        # Create atau get service runner
        if 'service_runner' not in ui_components:
            service_runner = create_service_runner(ui_components)
        else:
            service_runner = ui_components['service_runner']
        
        # Update UI state untuk running
        _update_ui_for_processing_start(ui_components)
        
        # Setup storage (Drive atau lokal)
        storage_success, storage_message = service_runner.setup_storage()
        logger.info(f"💾 Storage setup: {storage_message}")
        
        # Jalankan preprocessing async
        future: Future = service_runner.run_preprocessing(config)
        
        # Store future untuk tracking
        ui_components['processing_future'] = future
        
        # Setup callback untuk completion
        def on_complete(fut: Future):
            try:
                result = fut.result()  # Akan raise exception jika ada error
                _handle_preprocessing_success(ui_components, result)
            except Exception as e:
                _handle_preprocessing_error(ui_components, None, str(e))
        
        # Add callback (non-blocking)
        future.add_done_callback(on_complete)
        
        logger.info("🚀 Preprocessing dimulai secara asynchronous")
        
    except Exception as e:
        logger.error(f"❌ Error eksekusi preprocessing: {str(e)}")
        _handle_preprocessing_error(ui_components, None, str(e))

def _cancel_preprocessing(ui_components: Dict[str, Any], button: Any) -> None:
    """
    Cancel preprocessing dan reset UI.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Tombol yang perlu di-enable kembali
    """
    logger.info("ℹ️ Preprocessing dibatalkan oleh pengguna")
    
    # Update status panel
    from smartcash.ui.utils.alert_utils import update_status_panel
    update_status_panel(ui_components['status_panel'], "Preprocessing dibatalkan", "info")
    
    # Re-enable button
    if button and hasattr(button, 'disabled'):
        button.disabled = False

def _update_ui_for_processing_start(ui_components: Dict[str, Any]) -> None:
    """
    Update UI saat preprocessing dimulai.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Disable main button
    if 'preprocess_button' in ui_components:
        ui_components['preprocess_button'].disabled = True
    
    # Enable dan tampilkan stop button
    if 'stop_button' in ui_components:
        ui_components['stop_button'].disabled = False
        ui_components['stop_button'].layout.display = 'inline-block'
    
    # Disable other buttons
    buttons_to_disable = ['save_button', 'reset_button', 'cleanup_button']
    for button_name in buttons_to_disable:
        if button_name in ui_components and hasattr(ui_components[button_name], 'disabled'):
            ui_components[button_name].disabled = True
    
    # Update status
    from smartcash.ui.utils.alert_utils import update_status_panel
    update_status_panel(ui_components['status_panel'], "🚀 Preprocessing sedang berjalan...", "info")
    
    # Set flags
    ui_components['preprocessing_running'] = True

def _handle_preprocessing_success(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Handle preprocessing yang berhasil.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil preprocessing
    """
    # Log hasil
    total_processed = result.get('processed', 0)
    total_skipped = result.get('skipped', 0)
    total_failed = result.get('failed', 0)
    
    logger.info(f"✅ Preprocessing berhasil diselesaikan:")
    logger.info(f"  • Diproses: {total_processed}")
    logger.info(f"  • Dilewati: {total_skipped}")
    logger.info(f"  • Gagal: {total_failed}")
    
    # Update status panel
    from smartcash.ui.utils.alert_utils import update_status_panel
    update_status_panel(
        ui_components['status_panel'], 
        f"✅ Preprocessing selesai - {total_processed} file diproses", 
        "success"
    )
    
    # Reset UI
    _reset_ui_after_processing(ui_components)

def _handle_preprocessing_error(ui_components: Dict[str, Any], button: Any, error_message: str) -> None:
    """
    Handle error saat preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Tombol yang perlu di-enable kembali
        error_message: Pesan error
    """
    logger.error(f"❌ Error preprocessing: {error_message}")
    
    # Update status panel
    from smartcash.ui.utils.alert_utils import update_status_panel
    update_status_panel(ui_components['status_panel'], f"❌ Error: {error_message}", "error")
    
    # Re-enable buttons
    if button and hasattr(button, 'disabled'):
        button.disabled = False
    
    # Reset UI
    _reset_ui_after_processing(ui_components)

def _reset_ui_after_processing(ui_components: Dict[str, Any]) -> None:
    """
    Reset UI setelah preprocessing selesai atau error.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Enable main button
    if 'preprocess_button' in ui_components:
        ui_components['preprocess_button'].disabled = False
    
    # Hide dan disable stop button
    if 'stop_button' in ui_components:
        ui_components['stop_button'].disabled = True
        ui_components['stop_button'].layout.display = 'none'
    
    # Enable other buttons
    buttons_to_enable = ['save_button', 'reset_button', 'cleanup_button']
    for button_name in buttons_to_enable:
        if button_name in ui_components and hasattr(ui_components[button_name], 'disabled'):
            ui_components[button_name].disabled = False
    
    # Clear progress
    if 'progress_tracker' in ui_components:
        ui_components['progress_tracker'].reset()
    
    # Reset flags
    ui_components['preprocessing_running'] = False
    ui_components['stop_requested'] = False
    
    # Clear confirmation area
    if 'confirmation_area' in ui_components:
        ui_components['confirmation_area'].clear_output(wait=True)
        ui_components['confirmation_area'].layout.display = 'none'
    
    logger.debug("🔄 UI berhasil direset setelah preprocessing")