"""
File: smartcash/ui/dataset/augmentation/handlers/result_handler.py
Deskripsi: Handler untuk pemrosesan hasil augmentasi (SRP)
"""

from typing import Dict, Any
from smartcash.ui.dataset.augmentation.handlers.state_handler import StateHandler

def handle_augmentation_result(ui_components: Dict[str, Any], result: Dict[str, Any], 
                              duration: float, ui_logger) -> None:
    """
    Handle hasil augmentasi dan update UI.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil augmentasi dari service
        duration: Durasi proses dalam detik
        ui_logger: UI Logger bridge
    """
    state_handler = StateHandler(ui_components, ui_logger)
    
    try:
        # Format hasil untuk UI
        ui_result = _format_result_for_ui(result, duration)
        
        # Update UI berdasarkan status
        if ui_result['status'] == 'success':
            _handle_success_result(ui_components, ui_result, ui_logger)
        elif ui_result['status'] == 'cancelled':
            _handle_cancelled_result(ui_components, ui_result, ui_logger)
        else:
            _handle_error_result(ui_components, ui_result, ui_logger)
        
        # Log details untuk debugging
        _log_result_details(ui_result, ui_logger)
        
    finally:
        # Reset UI state
        state_handler.set_running(False)

def _format_result_for_ui(result: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """Format hasil service untuk UI."""
    status = result.get('status', 'error')
    generated = result.get('generated_images', 0)
    processed = result.get('processed', 0)
    
    # Hitung success rate
    success_rate = result.get('success_rate', 0)
    if success_rate == 0 and processed > 0:
        success_rate = (generated / processed) * 100
    
    # Format message
    message = result.get('message', 'Proses selesai')
    if status == 'success' and generated > 0:
        message = f"✅ Augmentasi berhasil: {generated} gambar dibuat dalam {duration:.1f}s"
    elif status == 'cancelled':
        message = f"⏹️ Augmentasi dibatalkan: {generated} gambar telah dibuat"
    elif status == 'error':
        message = f"❌ Error: {message}"
    
    return {
        'status': status,
        'message': message,
        'generated_images': generated,
        'processed': processed,
        'success_rate': success_rate,
        'duration': duration,
        'storage_type': result.get('storage_type', 'Local'),
        'class_stats': result.get('class_stats', {}),
        'details': {
            'split': result.get('split', 'train'),
            'output_dir': result.get('output_dir', 'data/augmented')
        }
    }

def _handle_success_result(ui_components: Dict[str, Any], ui_result: Dict[str, Any], ui_logger) -> None:
    """Handle hasil sukses."""
    ui_logger.success(ui_result['message'])
    
    # Tampilkan tombol cleanup jika berhasil
    if 'cleanup_button' in ui_components:
        cleanup_button = ui_components['cleanup_button']
        if hasattr(cleanup_button, 'layout'):
            cleanup_button.layout.display = 'block'
    
    # Update status panel jika ada
    _update_status_panel(ui_components, ui_result['message'], 'success')

def _handle_cancelled_result(ui_components: Dict[str, Any], ui_result: Dict[str, Any], ui_logger) -> None:
    """Handle hasil cancelled."""
    ui_logger.warning(ui_result['message'])
    _update_status_panel(ui_components, ui_result['message'], 'warning')

def _handle_error_result(ui_components: Dict[str, Any], ui_result: Dict[str, Any], ui_logger) -> None:
    """Handle hasil error."""
    ui_logger.error(ui_result['message'])
    _update_status_panel(ui_components, ui_result['message'], 'error')

def _update_status_panel(ui_components: Dict[str, Any], message: str, status: str) -> None:
    """Update status panel jika tersedia."""
    if 'status_panel' in ui_components:
        try:
            from smartcash.ui.components.status_panel import update_status_panel
            update_status_panel(ui_components['status_panel'], message, status)
        except ImportError:
            pass  # Status panel tidak tersedia

def _log_result_details(ui_result: Dict[str, Any], ui_logger) -> None:
    """Log detail hasil untuk debugging."""
    details = ui_result.get('details', {})
    generated = ui_result.get('generated_images', 0)
    success_rate = ui_result.get('success_rate', 0)
    duration = ui_result.get('duration', 0)
    storage_type = ui_result.get('storage_type', 'Local')
    
    if generated > 0:
        ui_logger.debug(f"📊 Detail: {generated} gambar, {success_rate:.1f}% berhasil, {duration:.1f}s, Storage: {storage_type}")
    
    # Log class stats jika ada
    class_stats = ui_result.get('class_stats', {})
    if class_stats:
        fulfilled_classes = [cls for cls, stats in class_stats.items() 
                           if stats.get('generated', 0) > 0]
        if fulfilled_classes:
            ui_logger.debug(f"🎯 Kelas yang diaugmentasi: {', '.join(map(str, fulfilled_classes))}")