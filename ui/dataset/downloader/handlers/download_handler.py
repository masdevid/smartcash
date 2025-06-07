"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Main handler untuk download operations dengan integrasi backend services dan UI progress
"""

from typing import Dict, Any
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.ui.components.dialogs import confirm

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup download handlers dengan simplified progress callback"""
    
    # Simplified progress callback
    def create_progress_callback():
        def progress_callback(step: str, current: int, total: int, message: str):
            try:
                progress_tracker = ui_components.get('progress_tracker')
                if progress_tracker and hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(current, message)
                
                # Simple logging
                logger = ui_components.get('logger')
                if logger:
                    logger.info(message)
            except Exception:
                pass  # Silent fail to prevent blocking
        
        return progress_callback
    
    ui_components['progress_callback'] = create_progress_callback()
    
    # Setup individual handlers
    setup_download_handler(ui_components, config)
    setup_check_handler(ui_components, config)
    setup_cleanup_handler(ui_components, config)
    
    # Setup save/reset handlers
    def save_config_handler(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _handle_ui_error(ui_components, "Config handler tidak tersedia", button, None)
                return
            
            success = config_handler.save_config(ui_components)
            if success:
                _show_ui_success(ui_components, "✅ Konfigurasi berhasil disimpan", button, None)
            else:
                _handle_ui_error(ui_components, "❌ Gagal menyimpan konfigurasi", button, None)
        except Exception as e:
            _handle_ui_error(ui_components, f"❌ Error saat save: {str(e)}", button, None)
    
    def reset_config_handler(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _handle_ui_error(ui_components, "Config handler tidak tersedia", button, None)
                return
            
            success = config_handler.reset_config(ui_components)
            if success:
                _show_ui_success(ui_components, "🔄 Konfigurasi berhasil direset", button, None)
            else:
                _handle_ui_error(ui_components, "❌ Gagal reset konfigurasi", button, None)
        except Exception as e:
            _handle_ui_error(ui_components, f"❌ Error saat reset: {str(e)}", button, None)
    
    # Bind save/reset handlers
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    if save_button:
        save_button.on_click(save_config_handler)
    if reset_button:
        reset_button.on_click(reset_config_handler)
    
    return ui_components

def setup_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup download handler dengan backend service integration"""
    
    def execute_download(button=None):
        button_manager = _get_button_manager(ui_components)
        logger = ui_components.get('logger')
        
        _clear_outputs(ui_components)
        button_manager.disable_buttons('download_button')
        
        try:
            # Extract dan validate config dari UI
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _handle_ui_error(ui_components, "Config handler tidak tersedia", button, button_manager)
                return
            
            ui_config = config_handler.extract_config(ui_components)
            validation = config_handler.validate_config(ui_config)
            
            if not validation['valid']:
                error_msg = f"Konfigurasi tidak valid:\n• {chr(10).join(validation['errors'])}"
                _handle_ui_error(ui_components, error_msg, button, button_manager)
                return
            
            if logger:
                logger.info("🚀 Memulai download dataset")
            
            # Show progress tracker
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                progress_tracker.show("Dataset Download")
                progress_tracker.update_overall(0, "🚀 Memulai download...")
            
            # Check existing dataset dan konfirmasi
            _check_and_confirm_download(ui_config, ui_components, button, button_manager)
            
        except Exception as e:
            _handle_ui_error(ui_components, f"Error download handler: {str(e)}", button, button_manager)
    
    # Bind handler
    download_button = ui_components.get('download_button')
    if download_button:
        download_button.on_click(execute_download)

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup check handler dengan backend scanner integration"""
    
    def execute_check(button=None):
        button_manager = _get_button_manager(ui_components)
        logger = ui_components.get('logger')
        
        _clear_outputs(ui_components)
        button_manager.disable_buttons('check_button')
        
        try:
            if logger:
                logger.info("🔍 Checking dataset status")
            
            # Show progress tracker
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                progress_tracker.show("Dataset Check")
                progress_tracker.update_overall(0, "🔍 Memulai scan...")
            
            # Use backend scanner
            from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
            scanner = create_dataset_scanner(logger)
            scanner.set_progress_callback(ui_components['progress_callback'])
            
            # Execute scan
            result = scanner.scan_existing_dataset()
            
            # Display results
            if result.get('status') == 'success':
                _display_check_results(ui_components, result)
                _show_ui_success(ui_components, "Dataset check selesai", button, button_manager)
            else:
                _handle_ui_error(ui_components, result.get('message', 'Scan failed'), button, button_manager)
                
        except Exception as e:
            _handle_ui_error(ui_components, f"Error check handler: {str(e)}", button, button_manager)
    
    # Bind handler
    check_button = ui_components.get('check_button')
    if check_button:
        check_button.on_click(execute_check)

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup cleanup handler dengan backend cleanup service"""
    
    def execute_cleanup(button=None):
        button_manager = _get_button_manager(ui_components)
        logger = ui_components.get('logger')
        
        _clear_outputs(ui_components)
        button_manager.disable_buttons('cleanup_button')
        
        try:
            # Get cleanup targets dari backend
            from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
            scanner = create_dataset_scanner(logger)
            targets_result = scanner.get_cleanup_targets()
            
            if targets_result.get('status') != 'success':
                _handle_ui_error(ui_components, "Gagal mendapatkan cleanup targets", button, button_manager)
                return
            
            summary = targets_result.get('summary', {})
            total_files = summary.get('total_files', 0)
            
            if total_files == 0:
                _show_ui_success(ui_components, "Tidak ada file untuk dibersihkan", button, button_manager)
                return
            
            # Show confirmation dialog
            _show_cleanup_confirmation(ui_components, targets_result, button, button_manager)
            
        except Exception as e:
            _handle_ui_error(ui_components, f"Error cleanup handler: {str(e)}", button, button_manager)
    
    # Bind handler
    cleanup_button = ui_components.get('cleanup_button')
    if cleanup_button:
        cleanup_button.on_click(execute_cleanup)

def _check_and_confirm_download(ui_config: Dict[str, Any], ui_components: Dict[str, Any], button, button_manager):
    """Enhanced check dengan proper content validation"""
    try:
        logger = ui_components.get('logger')
        if logger:
            logger.info("🔍 Checking for existing dataset...")
        
        # Enhanced check - validate actual content, not just directories
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        dataset_path = env_manager.get_dataset_path()
        
        # Check for actual files in splits
        has_content = False
        total_images = 0
        
        for split in ['train', 'valid', 'test']:
            split_dir = dataset_path / split / 'images'
            if split_dir.exists():
                images = list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.png'))
                if images:
                    total_images += len(images)
                    has_content = True
        
        if logger:
            logger.info(f"📊 Existing dataset check: {'Found' if has_content else 'Empty'} ({total_images} images)")
        
        if has_content:
            # Show explicit confirmation dengan file count
            _show_explicit_download_confirmation(ui_config, ui_components, button, button_manager, total_images)
        else:
            # No existing content, proceed directly
            _execute_download_with_backend(ui_config, ui_components, button, button_manager)
            
    except Exception as e:
        if ui_components.get('logger'):
            ui_components['logger'].error(f"❌ Error checking existing dataset: {str(e)}")
        # Proceed anyway jika check gagal
        _execute_download_with_backend(ui_config, ui_components, button, button_manager)

def _show_explicit_download_confirmation(ui_config: Dict[str, Any], ui_components: Dict[str, Any], button, button_manager, existing_count: int):
    """Show explicit confirmation dengan IPython display"""
    try:
        from IPython.display import display, HTML
        
        roboflow = ui_config.get('data', {}).get('roboflow', {})
        download = ui_config.get('download', {})
        
        # Show confirmation in log_output
        log_output = ui_components.get('log_output')
        if log_output:
            confirmation_html = f"""
            <div style="padding: 15px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px; margin: 10px 0;">
                <h4 style="color: #856404; margin-top: 0;">⚠️ Konfirmasi Download Dataset</h4>
                <p><strong>Dataset existing akan ditimpa!</strong></p>
                <p>📊 Dataset saat ini: <strong>{existing_count:,} gambar</strong></p>
                <p>🎯 Target: <strong>{roboflow.get('workspace')}/{roboflow.get('project')}:v{roboflow.get('version')}</strong></p>
                <p>🔄 UUID Renaming: {'✅' if download.get('rename_files', True) else '❌'}</p>
                <p>💾 Backup: {'✅' if download.get('backup_existing', False) else '❌'}</p>
                <div style="margin-top: 15px;">
                    <button onclick="confirm_download()" style="background: #dc3545; color: white; border: none; padding: 8px 16px; border-radius: 4px; margin-right: 10px; cursor: pointer;">Ya, Lanjutkan Download</button>
                    <button onclick="cancel_download()" style="background: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Batal</button>
                </div>
            </div>
            <script>
                function confirm_download() {{
                    // Call Python function
                    IPython.notebook.kernel.execute("_confirmed_download()");
                }}
                function cancel_download() {{
                    IPython.notebook.kernel.execute("_cancelled_download()");
                }}
            </script>
            """
            
            with log_output:
                display(HTML(confirmation_html))
        
        # Set up response handlers
        def confirmed_download():
            _execute_download_with_backend(ui_config, ui_components, button, button_manager)
        
        def cancelled_download():
            logger = ui_components.get('logger')
            if logger:
                logger.info("🚫 Download dibatalkan oleh user")
            button_manager.enable_buttons()
        
        # Store handlers globally for JavaScript access
        import builtins
        builtins._confirmed_download = confirmed_download
        builtins._cancelled_download = cancelled_download
        
    except Exception as e:
        if ui_components.get('logger'):
            ui_components['logger'].error(f"❌ Error showing confirmation: {str(e)}")
        # Fallback: proceed with download
        _execute_download_with_backend(ui_config, ui_components, button, button_manager)

def _show_download_confirmation(ui_config: Dict[str, Any], ui_components: Dict[str, Any], button, button_manager):
    """Show download confirmation dialog"""
    roboflow = ui_config.get('data', {}).get('roboflow', {})
    download = ui_config.get('download', {})
    
    message_lines = [
        "Dataset existing akan ditimpa!",
        "",
        f"🎯 Target: {roboflow.get('workspace')}/{roboflow.get('project')}:v{roboflow.get('version')}",
        f"🔄 UUID Renaming: {'✅' if download.get('rename_files', True) else '❌'}",
        f"✅ Validasi: {'✅' if download.get('validate_download', True) else '❌'}",
        f"💾 Backup: {'✅' if download.get('backup_existing', False) else '❌'}",
        "",
        "Lanjutkan download?"
    ]
    
    confirm(
        "Konfirmasi Download Dataset",
        '\n'.join(message_lines),
        on_yes=lambda btn: _execute_download_with_backend(ui_config, ui_components, button, button_manager),
        on_no=lambda btn: (
            _log_to_accordion(ui_components, "🚫 Download dibatalkan", 'info'),
            button_manager.enable_buttons()
        )
    )

def _execute_download_with_backend(ui_config: Dict[str, Any], ui_components: Dict[str, Any], button, button_manager):
    """Execute download dengan enhanced debugging dan safe error handling"""
    try:
        from smartcash.dataset.downloader import get_downloader_instance, create_ui_compatible_config
        
        logger = ui_components.get('logger')
        
        # Debug: Config conversion
        if logger:
            logger.info("🔧 Converting UI config to backend format...")
        
        service_config = create_ui_compatible_config(ui_config)
        
        # Debug: Service creation
        if logger:
            logger.info("🏗️ Creating downloader service...")
        
        downloader = get_downloader_instance(service_config, logger)
        if not downloader:
            _handle_ui_error(ui_components, "❌ Gagal membuat download service", button, button_manager)
            return
        
        if logger:
            logger.info(f"✅ Downloader service created: {type(downloader).__name__}")
        
        # Debug: Progress callback setup
        if hasattr(downloader, 'set_progress_callback'):
            if logger:
                logger.info("📊 Setting up progress callback...")
            downloader.set_progress_callback(ui_components['progress_callback'])
            if logger:
                logger.info("✅ Progress callback configured")
        else:
            if logger:
                logger.warning("⚠️ Downloader tidak support progress callback")
        
        # Log download config
        _log_download_config(ui_components, ui_config)
        
        # Debug: Starting download
        if logger:
            logger.info("🚀 Starting backend download operation...")
        
        result = downloader.download_dataset()
        
        # Debug: Download result
        if logger:
            logger.info(f"📋 Download result status: {result.get('status') if result else 'None'}")
        
        # Handle result
        if result and result.get('status') == 'success':
            _show_download_success(ui_components, result, button, button_manager)
        else:
            error_msg = result.get('message', 'Download gagal - no error message') if result else 'No response from backend service'
            _handle_ui_error(ui_components, error_msg, button, button_manager)
            
    except Exception as e:
        _handle_ui_error(ui_components, f"❌ Exception in download execution: {str(e)}", button, button_manager)

def _show_cleanup_confirmation(ui_components: Dict[str, Any], targets_result: Dict[str, Any], button, button_manager):
    """Show cleanup confirmation dengan detail"""
    summary = targets_result.get('summary', {})
    targets = targets_result.get('targets', {})
    
    message_lines = [
        f"Akan menghapus {summary.get('total_files', 0):,} file ({summary.get('size_formatted', '0 B')})",
        "",
        "📂 Target cleanup:"
    ]
    
    # Add target details
    for target_name, target_info in targets.items():
        file_count = target_info.get('file_count', 0)
        size_formatted = target_info.get('size_formatted', '0 B')
        message_lines.append(f"  • {target_name}: {file_count:,} file ({size_formatted})")
    
    message_lines.extend(["", "⚠️ Direktori akan tetap dipertahankan", "Lanjutkan cleanup?"])
    
    confirm(
        "Konfirmasi Cleanup Dataset",
        '\n'.join(message_lines),
        on_yes=lambda btn: _execute_cleanup_with_backend(targets_result, ui_components, button, button_manager),
        on_no=lambda btn: (
            _log_to_accordion(ui_components, "🚫 Cleanup dibatalkan", 'info'),
            button_manager.enable_buttons()
        )
    )

def _execute_cleanup_with_backend(targets_result: Dict[str, Any], ui_components: Dict[str, Any], button, button_manager):
    """Execute cleanup dengan backend service"""
    try:
        logger = ui_components.get('logger')
        
        # Show progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.show("Dataset Cleanup")
        
        # Use backend cleanup service
        from smartcash.dataset.downloader.cleanup_service import create_cleanup_service
        cleanup_service = create_cleanup_service(logger)
        cleanup_service.set_progress_callback(ui_components['progress_callback'])
        
        # Execute cleanup
        result = cleanup_service.cleanup_dataset_files(targets_result.get('targets', {}))
        
        if result.get('status') == 'success':
            cleaned_count = len(result.get('cleaned_targets', []))
            _show_ui_success(ui_components, f"Cleanup selesai: {cleaned_count} direktori dibersihkan", button, button_manager)
        else:
            _handle_ui_error(ui_components, result.get('message', 'Cleanup failed'), button, button_manager)
            
    except Exception as e:
        _handle_ui_error(ui_components, f"Error saat cleanup: {str(e)}", button, button_manager)

# Helper functions
def _log_download_config(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Log download configuration dalam format yang rapi"""
    roboflow = config.get('data', {}).get('roboflow', {})
    download = config.get('download', {})
    
    # Mask API key untuk security
    api_key = roboflow.get('api_key', '')
    masked_key = f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}" if len(api_key) > 8 else '****'
    
    config_lines = [
        "🔧 Konfigurasi Download:",
        f"🎯 Target: {roboflow.get('workspace')}/{roboflow.get('project')}:v{roboflow.get('version')}",
        f"🔑 API Key: {masked_key}",
        f"📦 Format: {roboflow.get('output_format', 'yolov5pytorch')}",
        f"🔄 UUID Rename: {'✅' if download.get('rename_files', True) else '❌'}",
        f"✅ Validasi: {'✅' if download.get('validate_download', True) else '❌'}",
        f"💾 Backup: {'✅' if download.get('backup_existing', False) else '❌'}"
    ]
    
    _log_to_accordion(ui_components, '\n'.join(config_lines), 'info')

def _display_check_results(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Display scan results dalam format yang rapi"""
    summary = result.get('summary', {})
    
    # Main summary
    summary_lines = [
        "📊 Ringkasan Dataset:",
        f"📂 Path: {result.get('dataset_path')}",
        f"🖼️ Total Gambar: {summary.get('total_images', 0):,}",
        f"🏷️ Total Label: {summary.get('total_labels', 0):,}",
        f"📦 Download Files: {summary.get('download_files', 0):,}"
    ]
    
    # Splits detail
    splits = result.get('splits', {})
    if splits:
        summary_lines.append("\n📊 Detail per Split:")
        for split_name, split_data in splits.items():
            if split_data.get('status') == 'success':
                img_count = split_data.get('images', 0)
                label_count = split_data.get('labels', 0)
                size_formatted = split_data.get('size_formatted', '0 B')
                summary_lines.append(f"  • {split_name}: {img_count:,} gambar, {label_count:,} label ({size_formatted})")
    
    # Downloads detail
    downloads = result.get('downloads', {})
    if downloads.get('status') == 'success':
        download_count = downloads.get('file_count', 0)
        download_size = downloads.get('size_formatted', '0 B')
        summary_lines.append(f"\n📦 Downloads: {download_count:,} file ({download_size})")
    
    _log_to_accordion(ui_components, '\n'.join(summary_lines), 'success')

def _show_download_success(ui_components: Dict[str, Any], result: Dict[str, Any], button, button_manager):
    """Show download success dengan detailed stats"""
    stats = result.get('stats', {})
    
    # Format success message
    success_lines = [
        f"✅ Download selesai: {stats.get('total_images', 0):,} gambar, {stats.get('total_labels', 0):,} label",
        f"📂 Output: {result.get('output_dir', 'N/A')}",
        f"⏱️ Durasi: {result.get('duration', 0):.1f} detik"
    ]
    
    # Add split details
    splits = stats.get('splits', {})
    if splits:
        success_lines.append("📊 Detail splits:")
        for split_name, split_stats in splits.items():
            img_count = split_stats.get('images', 0)
            label_count = split_stats.get('labels', 0)
            success_lines.append(f"  • {split_name}: {img_count:,} gambar, {label_count:,} label")
    
    # Add UUID renaming info
    if stats.get('uuid_renamed'):
        naming_stats = stats.get('naming_stats', {})
        if naming_stats:
            success_lines.append(f"🔄 UUID Renaming: {naming_stats.get('total_files', 0)} files")
    
    success_message = '\n'.join(success_lines)
    _show_ui_success(ui_components, success_message, button, button_manager)

# UI state management helpers
class SimpleButtonManager:
    """Simple button state management"""
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.disabled_buttons = []
    
    def disable_buttons(self, exclude_button: str = None):
        """Disable all buttons except exclude_button"""
        buttons = ['download_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
        for btn_key in buttons:
            if btn_key != exclude_button and btn_key in self.ui_components:
                btn = self.ui_components[btn_key]
                if btn and hasattr(btn, 'disabled') and not btn.disabled:
                    btn.disabled = True
                    self.disabled_buttons.append(btn_key)
    
    def enable_buttons(self):
        """Re-enable previously disabled buttons"""
        for btn_key in self.disabled_buttons:
            if btn_key in self.ui_components:
                btn = self.ui_components[btn_key]
                if btn and hasattr(btn, 'disabled'):
                    btn.disabled = False
        self.disabled_buttons.clear()

def _get_button_manager(ui_components: Dict[str, Any]) -> SimpleButtonManager:
    """Get button manager instance"""
    if 'button_manager' not in ui_components:
        ui_components['button_manager'] = SimpleButtonManager(ui_components)
    return ui_components['button_manager']

def _log_to_accordion(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log message menggunakan ui_logger yang sudah ada"""
    logger = ui_components.get('logger')
    if logger:
        # Map level ke ui_logger method
        log_methods = {
            'info': logger.info,
            'success': logger.success,
            'warning': logger.warning,
            'error': logger.error
        }
        log_method = log_methods.get(level, logger.info)
        log_method(message)
    
    # Auto-expand untuk errors/warnings
    if level in ['error', 'warning'] and 'log_accordion' in ui_components:
        if hasattr(ui_components['log_accordion'], 'selected_index'):
            ui_components['log_accordion'].selected_index = 0

def _get_log_level(step: str) -> str:
    """Determine log level berdasarkan step"""
    if any(keyword in step.lower() for keyword in ['error', 'fail', 'failed']):
        return 'error'
    elif any(keyword in step.lower() for keyword in ['warning', 'warn', 'issue']):
        return 'warning'
    elif any(keyword in step.lower() for keyword in ['success', 'complete', 'done', 'selesai']):
        return 'success'
    else:
        return 'info'

def _clear_outputs(ui_components: Dict[str, Any]):
    """Clear UI output areas"""
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        with ui_components['log_output']:
            ui_components['log_output'].clear_output(wait=True)

def _handle_ui_error(ui_components: Dict[str, Any], error_msg: str, button=None, button_manager=None):
    """Handle error dengan UI updates"""
    logger = ui_components.get('logger')
    if logger:
        logger.error(f"❌ {error_msg}")
    
    # Update progress tracker
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        progress_tracker.error(error_msg)
    
    # Log to accordion
    _log_to_accordion(ui_components, f"❌ {error_msg}", 'error')
    
    # Show status
    show_status_safe(error_msg, 'error', ui_components)
    
    # Enable buttons
    if button_manager:
        button_manager.enable_buttons()

def _show_ui_success(ui_components: Dict[str, Any], message: str, button=None, button_manager=None):
    """Show success dengan UI updates"""
    logger = ui_components.get('logger')
    if logger:
        logger.success(f"✅ {message}")
    
    # Update progress tracker
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        progress_tracker.complete(message)
    
    # Log to accordion
    _log_to_accordion(ui_components, f"✅ {message}", 'success')
    
    # Show status
    show_status_safe(message, 'success', ui_components)
    
    # Enable buttons
    if button_manager:
        button_manager.enable_buttons()