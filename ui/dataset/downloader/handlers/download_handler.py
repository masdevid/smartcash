"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Fixed download handler dengan proper integration dan mengatasi error 'bool' object is not callable
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable
from smartcash.ui.utils.fallback_utils import try_operation_safe, show_status_safe
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.dataset.downloader.handlers.check_handler import setup_check_handler
from smartcash.ui.dataset.downloader.handlers.cleanup_handler import setup_cleanup_handler
from smartcash.ui.dataset.downloader.handlers.validation_handler import setup_validation_handler
from smartcash.ui.dataset.downloader.utils.operation_utils import get_streamlined_download_operations
from smartcash.dataset.downloader import get_downloader_instance  # Fixed import
from smartcash.common.logger import get_logger
from smartcash.common.utils.one_liner_fixes import safe_operation_or_none, safe_widget_operation

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handlers dengan fixed integration dan proper error handling"""
    
    logger = ui_components.get('logger') or get_logger('downloader.handlers')
    
    # Setup handlers dengan fixed wrapper dan safe operations
    check_handler = _wrap_with_safe_state_management(setup_check_handler(ui_components, config, logger), ui_components, logger)
    cleanup_handler = _wrap_with_safe_state_management(setup_cleanup_handler(ui_components, config, logger), ui_components, logger)
    validation_handler = setup_validation_handler(ui_components, config, logger)
    download_handler = _wrap_with_safe_state_management(_create_fixed_download_handler(ui_components, config, logger), ui_components, logger)
    
    # Bind handlers dengan safe button management
    _bind_button_handlers_safe(ui_components, {
        'check_handler': check_handler, 'download_handler': download_handler, 'cleanup_handler': cleanup_handler
    })
    
    ui_components.update({
        'check_handler': check_handler, 'download_handler': download_handler,
        'cleanup_handler': cleanup_handler, 'validation_handler': validation_handler
    })
    
    return ui_components

def _wrap_with_safe_state_management(handler: Callable, ui_components: Dict[str, Any], logger) -> Callable:
    """Fixed wrapper dengan safe button state management dan error handling"""
    
    def safe_state_managed_handler(button):
        """Fixed wrapper untuk mutual exclusion buttons dengan safe operations"""
        all_buttons = []
        try:
            # Safe button collection dengan fallback
            all_buttons = (_get_all_buttons_safe(button, ui_components) or 
                          [ui_components.get(btn) for btn in ['check_button', 'download_button', 'cleanup_button']])
            all_buttons = [btn for btn in all_buttons if btn and hasattr(btn, 'disabled')]
            
            # Safe disable all buttons
            safe_operation_or_none(lambda: [setattr(btn, 'disabled', True) for btn in all_buttons])
            
            # Execute handler dengan safe operation
            safe_operation_or_none(lambda: handler(button))
            
        except Exception as e:
            error_msg = f"❌ Error in handler: {str(e)}"
            logger and logger.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
        finally:
            # Always re-enable buttons
            safe_operation_or_none(lambda: [setattr(btn, 'disabled', False) for btn in all_buttons])
    
    return safe_state_managed_handler

def _get_all_buttons_safe(button, ui_components: Dict[str, Any]) -> list:
    """Safe get all buttons dengan fallback"""
    # Try to get from button attribute first
    if button and hasattr(button, '_all_buttons'):
        return getattr(button, '_all_buttons', [])
    
    # Fallback to UI components
    button_names = ['check_button', 'download_button', 'cleanup_button']
    return [ui_components.get(btn) for btn in button_names if btn in ui_components]

def _create_fixed_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Create fixed download handler dengan proper service integration dan error handling"""
    
    def handle_download(button):
        """Handle download dengan fixed validation dan service call"""
        try:
            # Safe config extraction
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                show_status_safe("❌ Config handler tidak ditemukan", "error", ui_components)
                return
            
            current_config = safe_operation_or_none(lambda: config_handler.extract_config(ui_components))
            if not current_config:
                show_status_safe("❌ Gagal mengambil konfigurasi", "error", ui_components)
                return
            
            # Safe validation
            validation = safe_operation_or_none(lambda: config_handler.validate_config(current_config))
            if not validation or not validation.get('valid', False):
                error_msg = f"❌ Config tidak valid: {'; '.join(validation.get('errors', ['Unknown error']) if validation else ['Validation failed'])}"
                show_status_safe(error_msg, "error", ui_components)
                return
            
            # Pastikan konfigurasi memiliki format yang benar untuk download service
            download_config = {
                'endpoint': 'roboflow',  # Default endpoint
                'workspace': current_config.get('workspace', 'smartcash-wo2us'),
                'project': current_config.get('project', 'rupiah-emisi-2022'),
                'version': current_config.get('version', '3'),
                'api_key': current_config.get('api_key', ''),
                'output_format': current_config.get('output_format', 'yolov5pytorch'),
                'validate_download': current_config.get('validate_download', True),
                'organize_dataset': True,
                'backup_existing': current_config.get('backup_existing', False)
            }
            
            # Safe existing dataset check
            operations = safe_operation_or_none(lambda: get_streamlined_download_operations())
            has_existing = safe_operation_or_none(lambda: operations.check_existing_dataset() if operations else False) or False
            
            if has_existing:
                _show_download_confirmation_safe(ui_components, download_config, logger)
            else:
                _execute_download_sync_safe(ui_components, download_config, logger)
                
        except Exception as e:
            error_msg = f"❌ Error download handler: {str(e)}"
            logger and logger.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
    
    return handle_download

def _show_download_confirmation_safe(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Show download confirmation dengan safe dialog creation dan proper error handling"""
    # Extract config values untuk dialog
    workspace = config.get('workspace', '')
    project = config.get('project', '')
    version = config.get('version', '')
    dataset_id = f"{workspace}/{project}:v{version}"
    
    def on_confirm_handler(button):
        safe_operation_or_none(lambda: _execute_download_sync_safe(ui_components, config, logger))
    
    def on_cancel_handler(button):
        safe_operation_or_none(lambda: _clear_confirmation_area_safe(ui_components))
    
    try:
        # Import confirmation dialog creator
        from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
        
        confirmation_dialog = create_confirmation_dialog(
            title="Konfirmasi Download Dataset",
            message=f"""📥 **Dataset Download Confirmation**

🎯 **Target Dataset:** {dataset_id}

⚠️ **Perhatian:**
• Dataset existing akan ditimpa jika ada
• Backup otomatis: {'✅ Ya' if config.get('backup_existing') else '❌ Tidak'}
• Validasi hasil: {'✅ Ya' if config.get('validate_download') else '❌ Tidak'}
• Format: YOLOv5 PyTorch (hardcoded)

🚀 Lanjutkan download?""",
            on_confirm=on_confirm_handler,
            on_cancel=on_cancel_handler,
            confirm_text="Ya, Download", 
            cancel_text="Batal"
        )
        
        safe_operation_or_none(lambda: _show_in_confirmation_area_safe(ui_components, confirmation_dialog))
    except Exception as e:
        error_msg = f"❌ Error saat menampilkan konfirmasi: {str(e)}"
        logger and logger.error(error_msg)
        show_status_safe(error_msg, "error", ui_components)
        # Langsung jalankan download jika konfirmasi gagal
        safe_operation_or_none(lambda: _execute_download_sync_safe(ui_components, config, logger))

def _execute_download_sync_safe(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Execute download dengan safe service integration dan proper error handling"""
    try:
        # Clear confirmation area
        safe_operation_or_none(lambda: _clear_confirmation_area_safe(ui_components))
        
        # Safe progress tracker setup
        progress_tracker = ui_components.get('progress_tracker')
        if not progress_tracker:
            logger and logger.error("❌ Progress tracker tidak ditemukan")
            show_status_safe("❌ Progress tracker tidak tersedia", "error", ui_components)
            return
        
        # Setup progress dengan safe API calls
        download_steps = ["validate", "connect", "metadata", "download", "extract", "organize"]
        step_weights = {"validate": 5, "connect": 10, "metadata": 15, "download": 50, "extract": 15, "organize": 5}
        safe_operation_or_none(lambda: progress_tracker.show("Download Dataset", download_steps, step_weights))
        
        # Validasi konfigurasi sebelum membuat download service
        required_fields = ['workspace', 'project', 'version', 'api_key']
        missing_fields = []
        
        # Gunakan nilai default jika tidak ada
        for field in required_fields:
            if field not in config or not config[field]:
                # Coba ambil dari default config
                from smartcash.ui.dataset.downloader.handlers.defaults import get_default_download_config
                default_config = get_default_download_config()
                if field in default_config and default_config[field]:
                    config[field] = default_config[field]
                    logger and logger.info(f"🔄 Menggunakan nilai default untuk {field}: {config[field]}")
                else:
                    missing_fields.append(field)
        
        if missing_fields:
            error_msg = f"❌ Konfigurasi tidak lengkap: {', '.join(missing_fields)} tidak ditemukan"
            safe_operation_or_none(lambda: progress_tracker.error(error_msg))
            show_status_safe(error_msg, "error", ui_components)
            return
        
        # Pastikan konfigurasi memiliki format yang benar untuk download service
        download_config = {
            'endpoint': 'roboflow',  # Default endpoint
            'workspace': config.get('workspace', 'smartcash-wo2us'),
            'project': config.get('project', 'rupiah-emisi-2022'),
            'version': config.get('version', '3'),
            'api_key': config.get('api_key', ''),
            'output_format': config.get('output_format', 'yolov5pytorch'),
            'validate_download': config.get('validate_download', True),
            'organize_dataset': True,
            'backup_existing': config.get('backup_existing', False)
        }
        
        # Extract config values untuk logging
        workspace = download_config.get('workspace', '')
        project = download_config.get('project', '')
        version = download_config.get('version', '')
        
        # Get downloader instance dengan config lengkap
        logger.info(f"🔄 Membuat download service untuk {workspace}/{project}:v{version}")
        downloader = get_downloader_instance(download_config, logger)
        
        if not downloader:
            error_msg = "❌ Gagal membuat download service"
            safe_operation_or_none(lambda: progress_tracker.error(error_msg))
            show_status_safe(error_msg, "error", ui_components)
            logger and logger.error(f"Detail config: {str(download_config)}")
            return
        
        # Setup progress callback
        if progress_tracker:
            progress_callback = _create_safe_progress_callback(progress_tracker, logger)
            safe_operation_or_none(lambda: downloader.set_progress_callback(progress_callback))
        
        # Execute download tanpa parameter karena config sudah diinisialisasi di constructor
        logger.info(f"📥 Memulai download dataset {workspace}/{project}:v{version}")
        result = safe_operation_or_none(lambda: downloader.download_dataset())
        
        if not result:
            error_msg = "❌ Download service tidak merespons"
            safe_operation_or_none(lambda: progress_tracker.error(error_msg))
            show_status_safe(error_msg, "error", ui_components)
            return
        
        # Handle result dengan safe status display
        if result.get('status') == 'success':
            stats = result.get('stats', {})
            total_images = stats.get('total_images', 0)
            success_msg = f"✅ Dataset berhasil didownload: {total_images:,} gambar"
            safe_operation_or_none(lambda: progress_tracker.complete(success_msg))
            show_status_safe(f"{success_msg} ke {result.get('output_dir', 'unknown location')}", "success", ui_components)
            logger and logger.success(success_msg)
        else:
            error_msg = f"❌ Download gagal: {result.get('message', 'Unknown error')}"
            safe_operation_or_none(lambda: progress_tracker.error(error_msg))
            show_status_safe(error_msg, "error", ui_components)
            logger and logger.error(error_msg)
            
    except Exception as e:
        error_msg = f"❌ Error saat download: {str(e)}"
        progress_tracker = ui_components.get('progress_tracker')
        safe_operation_or_none(lambda: progress_tracker.error(error_msg) if progress_tracker else None)
        show_status_safe(error_msg, "error", ui_components)
        logger and logger.error(error_msg)

def _create_safe_progress_callback(progress_tracker, logger) -> Callable:
    """Create safe dual-level progress callback dengan API yang benar"""
    
    def safe_progress_callback(step: str, current: int, total: int, message: str):
        """Safe dual-level progress dengan proper API calls dan error handling"""
        def callback_operation():
            percentage = min(100, max(0, int((current / total) * 100) if total > 0 else 0))
            
            # Safe step mapping ke progress levels
            step_mapping = {
                'validate': ('overall', "🔄 Validasi parameter"),
                'connect': ('overall', "🌐 Koneksi Roboflow"),
                'metadata': ('overall', "📊 Ambil metadata"), 
                'download': ('current', "📥 Download dataset"),
                'extract': ('current', "📦 Ekstrak files"),
                'organize': ('current', "🗂️ Organisir struktur")
            }
            
            if step in step_mapping:
                level_name, overall_msg = step_mapping[step]
                
                # Safe API calls dengan proper method names
                if level_name == 'overall':
                    # Gunakan metode update() yang benar untuk progress tracker
                    safe_operation_or_none(lambda: progress_tracker.update('overall', percentage, overall_msg))
                else:
                    # Gunakan metode update() yang benar untuk progress tracker
                    safe_operation_or_none(lambda: progress_tracker.update('current', percentage, message))
            else:
                # Generic progress update untuk unknown steps
                safe_operation_or_none(lambda: progress_tracker.update('current', percentage, message))
            
            # Log progress untuk debugging
            logger and logger.debug(f"Progress {step}: {percentage}% - {message}")
        
        safe_operation_or_none(callback_operation)
    
    return safe_progress_callback

def _bind_button_handlers_safe(ui_components: Dict[str, Any], handlers: Dict[str, Callable]) -> None:
    """Bind button handlers dengan safe approach dan error handling"""
    button_mappings = [
        ('check_button', handlers.get('check_handler')),
        ('download_button', handlers.get('download_handler')),
        ('cleanup_button', handlers.get('cleanup_handler'))
    ]
    
    # Safe binding dengan proper error handling
    for btn_name, handler in button_mappings:
        if btn_name in ui_components and handler and callable(handler):
            button = ui_components[btn_name]
            if button and hasattr(button, 'on_click'):
                safe_widget_operation(button, 'on_click', handler)

def _show_in_confirmation_area_safe(ui_components: Dict[str, Any], dialog_widget) -> None:
    """Show dialog dalam confirmation area dengan safe display dan error handling"""
    def show_operation():
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'layout'):
            confirmation_area.layout.display = 'block'
            confirmation_area.layout.visibility = 'visible'
            
            if hasattr(confirmation_area, 'clear_output'):
                confirmation_area.clear_output(wait=True)
                
            from IPython.display import display
            with confirmation_area:
                display(dialog_widget)
    
    safe_operation_or_none(show_operation)

def _clear_confirmation_area_safe(ui_components: Dict[str, Any]) -> None:
    """Clear confirmation area dengan safe approach dan error handling"""
    def clear_operation():
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'clear_output'):
            confirmation_area.clear_output(wait=True)
            
            if hasattr(confirmation_area, 'layout'):
                confirmation_area.layout.display = 'none'
                confirmation_area.layout.visibility = 'hidden'
    
    safe_operation_or_none(clear_operation)

# Safe utilities dengan improved error handling dan status checks
def get_download_status_safe(ui: Dict[str, Any]) -> Dict[str, Any]:
    """Get download status dengan safe validation dan fallback"""
    def status_operation():
        return {
            'ready': bool(ui.get('progress_tracker')) and bool(ui.get('config_handler')),
            'handlers_count': len([k for k in ui.keys() if k.endswith('_handler')]),
            'buttons_available': all(btn in ui for btn in ['check_button', 'download_button', 'cleanup_button']),
            'progress_tracker_available': bool(ui.get('progress_tracker')),
            'config_handler_available': bool(ui.get('config_handler'))
        }
    
    return safe_operation_or_none(status_operation) or {
        'ready': False, 'handlers_count': 0, 'buttons_available': False,
        'progress_tracker_available': False, 'config_handler_available': False
    }

def validate_handlers_setup_safe(ui: Dict[str, Any]) -> bool:
    """Validate handlers setup dengan safe checking"""
    def validate_operation():
        required_handlers = ['check_handler', 'download_handler', 'cleanup_handler']
        return all(handler in ui and callable(ui[handler]) for handler in required_handlers)
    
    return bool(safe_operation_or_none(validate_operation))

def get_handler_summary_safe(ui: Dict[str, Any]) -> str:
    """Get handler summary dengan safe status formatting"""
    def summary_operation():
        handlers_count = len([k for k in ui.keys() if k.endswith('_handler')])
        progress_status = '✅' if 'progress_tracker' in ui else '❌'
        config_status = '✅' if 'config_handler' in ui else '❌'
        buttons_status = '✅' if all(btn in ui for btn in ['check_button', 'download_button', 'cleanup_button']) else '❌'
        
        return f"📊 Handlers: {handlers_count} | Progress: {progress_status} | Config: {config_status} | Buttons: {buttons_status}"
    
    return safe_operation_or_none(summary_operation) or "❌ Error getting handler summary"

def reset_download_handlers_safe(ui_components: Dict[str, Any]) -> bool:
    """Reset download handlers state dengan safe operations"""
    def reset_operation():
        # Clear confirmation area
        _clear_confirmation_area_safe(ui_components)
        
        # Reset progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
        
        # Enable all buttons
        buttons = ['check_button', 'download_button', 'cleanup_button']
        for btn_name in buttons:
            btn = ui_components.get(btn_name)
            if btn and hasattr(btn, 'disabled'):
                btn.disabled = False
        
        return True
    
    return bool(safe_operation_or_none(reset_operation))