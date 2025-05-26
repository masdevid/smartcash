"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Main augmentation handler - orchestration dengan SRP handlers
"""

from typing import Dict, Any
from IPython.display import display
from smartcash.dataset.augmentor.service import create_service_from_ui
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from .button_state_handler import create_button_state_handler
from .config_handler import create_config_handler  
from .progress_handler import create_progress_handler

def handle_augmentation_button_click(ui_components: Dict[str, Any], button: Any):
    """
    Handler utama untuk tombol augmentasi - orchestration dengan SRP handlers.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Widget button yang diklik
    """
    # Initialize SRP handlers
    button_handler = create_button_state_handler(ui_components)
    progress_handler = create_progress_handler(ui_components)
    config_handler = create_config_handler(ui_components)
    
    try:
        # 1. Set processing state
        button_handler.set_processing_state('augment_button', "🚀 Augmenting...")
        
        # 2. Start progress tracking
        progress_handler.start_operation('augmentation')
        
        # 3. Extract dan validate config
        config = config_handler.extract_config_from_ui()
        validation = config_handler.validate_config(config)
        
        if not validation['valid']:
            error_msg = "Konfigurasi tidak valid: " + "; ".join(validation['errors'])
            _handle_error(ui_components, button_handler, progress_handler, error_msg)
            return
        
        # 4. Create service dan setup progress callback
        service = create_service_from_ui(ui_components)
        progress_callback = progress_handler.create_progress_callback()
        
        # 5. Execute full augmentation pipeline
        result = service.run_full_augmentation_pipeline(
            target_split='train',
            progress_callback=progress_callback
        )
        
        # 6. Handle hasil
        if result['status'] == 'success':
            success_msg = f"Pipeline berhasil: {result['total_files']} file → {result['final_output']}"
            button_handler.set_success_state('augment_button', "✅ Selesai!")
            progress_handler.complete_operation(success_msg)
            _update_status_log(ui_components, success_msg, 'success')
        else:
            error_msg = result.get('message', 'Error tidak diketahui')
            _handle_error(ui_components, button_handler, progress_handler, f"Augmentasi gagal: {error_msg}")
        
    except Exception as e:
        _handle_error(ui_components, button_handler, progress_handler, f"Error handler: {str(e)}")

def handle_check_dataset_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler untuk check dataset status."""
    button_handler = create_button_state_handler(ui_components)
    
    try:
        button_handler.set_processing_state('check_button', "🔍 Checking...")
        
        service = create_service_from_ui(ui_components)
        status = service.get_augmentation_status()
        
        status_lines = [
            f"📁 Raw Data: {'✅ Ada' if status['raw_exists'] else '❌ Tidak Ada'}",
            f"🔄 Augmented: {'✅ Ada' if status['augmented_exists'] else '❌ Tidak Ada'} ({status['augmented_files']} files)",
            f"📊 Preprocessed: {'✅ Ada' if status['preprocessed_exists'] else '❌ Tidak Ada'} ({status['preprocessed_files']} files)"
        ]
        
        status_msg = "📊 Status Dataset:\n" + "\n".join(status_lines)
        _update_status_log(ui_components, status_msg, 'info')
        button_handler.set_success_state('check_button', "✅ Checked!")
        
    except Exception as e:
        button_handler.set_error_state('check_button', "❌ Error!")
        _update_status_log(ui_components, f"Error check dataset: {str(e)}", 'error')

def handle_cleanup_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler untuk cleanup dengan confirmation dialog."""
    def confirm_cleanup(confirm_button):
        button_handler = create_button_state_handler(ui_components)
        progress_handler = create_progress_handler(ui_components)
        
        try:
            button_handler.set_processing_state('cleanup_button', "🧹 Cleaning...")
            progress_handler.start_operation('cleanup')
            
            service = create_service_from_ui(ui_components)
            progress_callback = progress_handler.create_progress_callback()
            
            result = service.cleanup_augmented_data(
                include_preprocessed=True,
                progress_callback=progress_callback
            )
            
            if result['status'] == 'success':
                success_msg = f"Cleanup berhasil: {result.get('total_deleted', 0)} file dihapus"
                button_handler.set_success_state('cleanup_button', "✅ Clean!")
                progress_handler.complete_operation(success_msg)
                _update_status_log(ui_components, success_msg, 'success')
            else:
                error_msg = result.get('message', 'Error cleanup')
                _handle_error(ui_components, button_handler, progress_handler, f"Cleanup gagal: {error_msg}")
                
        except Exception as e:
            _handle_error(ui_components, button_handler, progress_handler, f"Error cleanup: {str(e)}")
    
    def cancel_cleanup(cancel_button):
        _update_status_log(ui_components, "Cleanup dibatalkan", 'info')
    
    confirmation_dialog = create_confirmation_dialog(
        title="Konfirmasi Cleanup Dataset",
        message="Apakah Anda yakin ingin menghapus semua file augmented?\n\n" +
                "File yang akan dihapus:\n" +
                "• Semua file dengan prefix 'aug_' di folder augmented\n" +
                "• Semua file dengan prefix 'aug_' di folder preprocessed\n\n" +
                "⚠️ Tindakan ini tidak dapat dibatalkan!",
        on_confirm=confirm_cleanup,
        on_cancel=cancel_cleanup,
        confirm_text="Ya, Hapus Semua",
        cancel_text="Batal",
        danger_mode=True
    )
    
    if 'confirmation_area' in ui_components:
        ui_components['confirmation_area'].clear_output()
        with ui_components['confirmation_area']:
            display(confirmation_dialog)

def handle_save_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler untuk save config."""
    button_handler = create_button_state_handler(ui_components)
    config_handler = create_config_handler(ui_components)
    
    try:
        button_handler.set_processing_state('save_button', "💾 Saving...")
        result = config_handler.save_config()
        
        if result['status'] == 'success':
            button_handler.set_success_state('save_button', "✅ Saved!")
            _update_status_log(ui_components, result['message'], 'success')
        else:
            button_handler.set_error_state('save_button', "❌ Error!")
            _update_status_log(ui_components, result['message'], 'error')
            
    except Exception as e:
        button_handler.set_error_state('save_button', "❌ Error!")
        _update_status_log(ui_components, f"Error save config: {str(e)}", 'error')

def handle_reset_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler untuk reset config."""
    button_handler = create_button_state_handler(ui_components)
    config_handler = create_config_handler(ui_components)
    
    try:
        button_handler.set_processing_state('reset_button', "🔄 Resetting...")
        result = config_handler.reset_to_default()
        
        if result['status'] == 'success':
            button_handler.set_success_state('reset_button', "✅ Reset!")
            _update_status_log(ui_components, result['message'], 'success')
        else:
            button_handler.set_error_state('reset_button', "❌ Error!")
            _update_status_log(ui_components, result['message'], 'error')
            
    except Exception as e:
        button_handler.set_error_state('reset_button', "❌ Error!")
        _update_status_log(ui_components, f"Error reset config: {str(e)}", 'error')

# Helper functions
def _handle_error(ui_components: Dict[str, Any], button_handler, progress_handler, error_msg: str):
    """Helper untuk handle error dengan consistent logging."""
    button_handler.restore_button_state('augment_button')
    progress_handler.error_operation(error_msg)
    _update_status_log(ui_components, f"❌ {error_msg}", 'error')
    
    try:
        from smartcash.common.logger import get_logger
        get_logger("augmentation_handler").error(error_msg)
    except:
        print(f"DEBUG: {error_msg}")

def _update_status_log(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Helper untuk update status log UI."""
    try:
        if 'logger' in ui_components:
            log_method = getattr(ui_components['logger'], level, ui_components['logger'].info)
            log_method(message)
        
        if 'status_panel' in ui_components:
            from smartcash.ui.components.status_panel import update_status_panel
            update_status_panel(ui_components['status_panel'], message, level)
            
    except Exception:
        print(f"[{level.upper()}] {message}")

def register_augmentation_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Register semua augmentation handlers ke UI components."""
    button_handlers_map = {
        'augment_button': handle_augmentation_button_click,
        'check_button': handle_check_dataset_button_click,
        'cleanup_button': handle_cleanup_button_click,
        'save_button': handle_save_config_button_click,
        'reset_button': handle_reset_config_button_click
    }
    
    registered_count = 0
    for button_key, handler_func in button_handlers_map.items():
        button = ui_components.get(button_key)
        if button and hasattr(button, 'on_click'):
            button.on_click(lambda b, h=handler_func, ui=ui_components: h(ui, b))
            registered_count += 1
    
    ui_components['registered_handlers'] = {
        'total': registered_count,
        'handlers': list(button_handlers_map.keys())
    }
    
    return ui_components