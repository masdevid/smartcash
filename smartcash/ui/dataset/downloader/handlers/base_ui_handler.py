"""
File: smartcash/ui/dataset/downloader/handlers/base_ui_handler.py
Deskripsi: Base UI handler dengan fokus pada UI logic, progress, dan logging
"""

from typing import Dict, Any, Callable, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.utils.fallback_utils import show_status_safe

class BaseUIHandler:
    """Base handler untuk UI operations dengan fokus pada display logic"""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        self.ui_components = ui_components
        self.logger = logger or get_logger(self.__class__.__name__)
        self.progress_tracker = ui_components.get('progress_tracker')
        self._is_processing = False
    
    def _setup_progress_callback(self) -> Callable[[str, int, int, str], None]:
        """Create UI progress callback untuk backend services"""
        def progress_callback(step: str, current: int, total: int, message: str):
            try:
                # Update dual progress tracker
                if self.progress_tracker:
                    # Overall progress mapping
                    self.progress_tracker.update_overall(current, message)
                    
                    # Step progress untuk detailed tracking
                    if hasattr(self.progress_tracker, 'update_step'):
                        step_progress = self._calculate_step_progress(step, current)
                        self.progress_tracker.update_step(step_progress, f"Step: {step}")
                
                # Log to accordion
                self._log_to_accordion(message, self._get_log_level(step))
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Progress callback error: {str(e)}")
        
        return progress_callback
    
    def _calculate_step_progress(self, step: str, overall_progress: int) -> int:
        """Calculate step progress dari overall progress"""
        # Map tahapan ke step progress
        step_ranges = {
            'init': (0, 100), 'metadata': (0, 100), 'backup': (0, 100),
            'download': (0, 100), 'extract': (0, 100), 'organize': (0, 100),
            'uuid_rename': (0, 100), 'validate': (0, 100), 'cleanup': (0, 100)
        }
        
        # Normalize step name
        normalized_step = step.lower().split('_')[0]
        
        if normalized_step in step_ranges:
            return min(100, max(0, overall_progress))
        
        return overall_progress
    
    def _get_log_level(self, step: str) -> str:
        """Determine log level based on step"""
        if any(keyword in step.lower() for keyword in ['error', 'fail', 'failed']):
            return 'error'
        elif any(keyword in step.lower() for keyword in ['warning', 'warn', 'issue']):
            return 'warning'
        elif any(keyword in step.lower() for keyword in ['success', 'complete', 'done', 'selesai']):
            return 'success'
        else:
            return 'info'
    
    def _log_to_accordion(self, message: str, level: str = 'info', expand: bool = False) -> None:
        """Log message to accordion dengan auto-expand untuk errors"""
        if 'log_output' in self.ui_components and hasattr(self.ui_components['log_output'], 'clear_output'):
            # Use log_output for newer implementations
            from IPython.display import display, HTML
            
            level_colors = {
                'info': '#007bff', 'success': '#28a745', 
                'warning': '#ffc107', 'error': '#dc3545'
            }
            
            color = level_colors.get(level, '#007bff')
            timestamp = __import__('datetime').datetime.now().strftime('%H:%M:%S')
            
            html = f"""
            <div style="margin:2px 0;padding:4px 8px;border-radius:4px;
                       background-color:rgba(248,249,250,0.8);border-left:3px solid {color};">
                <span style="color:#6c757d;font-size:11px;">[{timestamp}]</span>
                <span style="color:{color};margin-left:4px;">{message}</span>
            </div>
            """
            
            with self.ui_components['log_output']:
                display(HTML(html))
        
        # Auto-expand untuk errors/warnings
        if level in ['error', 'warning'] and 'log_accordion' in self.ui_components:
            if hasattr(self.ui_components['log_accordion'], 'selected_index'):
                self.ui_components['log_accordion'].selected_index = 0
    
    def _prepare_button_state(self, button, processing_text: str = "Processing...") -> None:
        """Prepare button untuk processing state"""
        if button and hasattr(button, 'disabled'):
            if not hasattr(button, '_original_description'):
                setattr(button, '_original_description', button.description)
            button.disabled = True
            button.description = processing_text
        
        self._is_processing = True
    
    def _restore_button_state(self, button=None) -> None:
        """Restore button ke original state"""
        if button and hasattr(button, 'disabled'):
            button.disabled = False
            if hasattr(button, '_original_description'):
                button.description = getattr(button, '_original_description')
        
        self._is_processing = False
    
    def _handle_ui_error(self, error_msg: str, button=None) -> None:
        """Handle error dengan UI updates"""
        self.logger.error(f"❌ {error_msg}")
        
        # Update progress tracker
        if self.progress_tracker:
            self.progress_tracker.error(error_msg)
        
        # Log to accordion
        self._log_to_accordion(f"❌ {error_msg}", 'error', expand=True)
        
        # Show status
        show_status_safe(error_msg, 'error', self.ui_components)
        
        # Restore button
        if button:
            self._restore_button_state(button)
    
    def _show_ui_success(self, message: str, button=None) -> None:
        """Show success dengan UI updates"""
        self.logger.success(f"✅ {message}")
        
        # Update progress tracker
        if self.progress_tracker:
            self.progress_tracker.complete(message)
        
        # Log to accordion
        self._log_to_accordion(f"✅ {message}", 'success')
        
        # Show status
        show_status_safe(message, 'success', self.ui_components)
        
        # Restore button
        if button:
            self._restore_button_state(button)
    
    def _clear_ui_output(self) -> None:
        """Clear UI output areas"""
        if 'log_output' in self.ui_components and hasattr(self.ui_components['log_output'], 'clear_output'):
            with self.ui_components['log_output']:
                self.ui_components['log_output'].clear_output(wait=True)
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size untuk UI display"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


class DownloadUIHandler(BaseUIHandler):
    """UI handler untuk download operations"""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        super().__init__(ui_components, logger)
    
    def handle_download_click(self, button) -> None:
        """Handle download button click dengan UI logic"""
        try:
            self._prepare_button_state(button, "📥 Downloading...")
            self._clear_ui_output()
            
            # Show progress tracker
            if self.progress_tracker:
                self.progress_tracker.show("Dataset Download")
                self.progress_tracker.update_overall(0, "🚀 Memulai download...")
            
            # Get config
            config_handler = self.ui_components.get('config_handler')
            if not config_handler:
                self._handle_ui_error("Config handler tidak tersedia", button)
                return
            
            # Extract dan validate config
            ui_config = config_handler.extract_config_from_ui(self.ui_components)
            validation = config_handler.validate_config(ui_config)
            
            if not validation['valid']:
                error_msg = f"Config tidak valid:\n• {chr(10).join(validation['errors'])}"
                self._handle_ui_error(error_msg, button)
                return
            
            # Log consolidated config
            self._log_download_config(ui_config)
            
            # Check existing dataset
            from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
            scanner = create_dataset_scanner(self.logger)
            has_existing = scanner.quick_check_existing()
            
            if has_existing:
                self._show_confirmation_dialog(ui_config, button)
            else:
                self._execute_download(ui_config, button)
                
        except Exception as e:
            self._handle_ui_error(f"Error download handler: {str(e)}", button)
    
    def _log_download_config(self, config: Dict[str, Any]) -> None:
        """Log download configuration dalam format consolidated"""
        roboflow = config.get('data', {}).get('roboflow', {})
        download = config.get('download', {})
        
        # Mask API key
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
        
        self._log_to_accordion('\n'.join(config_lines), 'info')
    
    def _execute_download(self, config: Dict[str, Any], button) -> None:
        """Execute download dengan backend service"""
        try:
            from smartcash.dataset.downloader import get_downloader_instance, create_ui_compatible_config
            
            # Convert config
            service_config = create_ui_compatible_config(config)
            
            # Create service
            downloader = get_downloader_instance(service_config, self.logger)
            if not downloader:
                self._handle_ui_error("Gagal membuat download service", button)
                return
            
            # Set progress callback
            if hasattr(downloader, 'set_progress_callback'):
                downloader.set_progress_callback(self._setup_progress_callback())
            
            # Execute download
            self._log_to_accordion("🚀 Memulai download dataset...", 'info')
            result = downloader.download_dataset()
            
            # Handle result
            if result and result.get('status') == 'success':
                stats = result.get('stats', {})
                success_msg = f"Dataset berhasil didownload: {stats.get('total_images', 0):,} gambar"
                self._show_download_success(result, button)
            else:
                error_msg = result.get('message', 'Download gagal') if result else 'No response from service'
                self._handle_ui_error(error_msg, button)
                
        except Exception as e:
            self._handle_ui_error(f"Error saat download: {str(e)}", button)
    
    def _show_download_success(self, result: Dict[str, Any], button) -> None:
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
        self._show_ui_success(success_message, button)
    
    def _show_confirmation_dialog(self, config: Dict[str, Any], button) -> None:
        """Show confirmation dialog untuk existing dataset"""
        from smartcash.ui.components.dialogs import confirm
        
        roboflow = config.get('data', {}).get('roboflow', {})
        download = config.get('download', {})
        
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
            on_yes=lambda btn: self._execute_download(config, button),
            on_no=lambda btn: (
                self._log_to_accordion("🚫 Download dibatalkan", 'info'),
                self._restore_button_state(button)
            )
        )


class CheckUIHandler(BaseUIHandler):
    """UI handler untuk check operations"""
    
    def handle_check_click(self, button) -> None:
        """Handle check button click dengan backend scanning"""
        try:
            self._prepare_button_state(button, "🔍 Checking...")
            self._clear_ui_output()
            
            # Show progress tracker
            if self.progress_tracker:
                self.progress_tracker.show("Dataset Check")
            
            # Use backend scanner
            from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
            scanner = create_dataset_scanner(self.logger)
            scanner.set_progress_callback(self._setup_progress_callback())
            
            # Execute scan
            result = scanner.scan_existing_dataset()
            
            # Display results
            if result.get('status') == 'success':
                self._display_scan_results(result)
                self._show_ui_success("Dataset check selesai", button)
            else:
                self._handle_ui_error(result.get('message', 'Scan failed'), button)
                
        except Exception as e:
            self._handle_ui_error(f"Error saat check: {str(e)}", button)
    
    def _display_scan_results(self, result: Dict[str, Any]) -> None:
        """Display scan results dengan format yang rapi"""
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
        
        self._log_to_accordion('\n'.join(summary_lines), 'success')


class CleanupUIHandler(BaseUIHandler):
    """UI handler untuk cleanup operations"""
    
    async def handle_cleanup_click(self, button) -> None:
        """Handle cleanup button click dengan confirmation"""
        try:
            self._prepare_button_state(button, "🔍 Scanning...")
            
            # Get cleanup targets from backend
            from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
            scanner = create_dataset_scanner(self.logger)
            targets_result = scanner.get_cleanup_targets()
            
            if targets_result.get('status') != 'success':
                self._handle_ui_error("Gagal mendapatkan cleanup targets", button)
                return
            
            summary = targets_result.get('summary', {})
            total_files = summary.get('total_files', 0)
            
            if total_files == 0:
                self._show_ui_success("Tidak ada file untuk dibersihkan", button)
                return
            
            # Show confirmation
            await self._show_cleanup_confirmation(targets_result, button)
            
        except Exception as e:
            self._handle_ui_error(f"Error saat cleanup: {str(e)}", button)
    
    async def _show_cleanup_confirmation(self, targets_result: Dict[str, Any], button) -> None:
        """Show cleanup confirmation dengan detail"""
        from smartcash.ui.components.dialogs import confirm
        
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
        
        confirmed = await confirm("Konfirmasi Cleanup", '\n'.join(message_lines))
        
        if confirmed:
            await self._execute_cleanup(targets_result, button)
        else:
            self._log_to_accordion("🚫 Cleanup dibatalkan", 'info')
            self._restore_button_state(button)
    
    async def _execute_cleanup(self, targets_result: Dict[str, Any], button) -> None:
        """Execute cleanup operation"""
        try:
            self._prepare_button_state(button, "🧹 Cleaning...")
            
            if self.progress_tracker:
                self.progress_tracker.show("Dataset Cleanup")
            
            # Use backend cleanup service
            from smartcash.dataset.downloader.cleanup_service import create_cleanup_service
            cleanup_service = create_cleanup_service(self.logger)
            cleanup_service.set_progress_callback(self._setup_progress_callback())
            
            # Execute cleanup
            result = cleanup_service.cleanup_dataset_files(targets_result.get('targets', {}))
            
            if result.get('status') == 'success':
                cleaned_count = len(result.get('cleaned_targets', []))
                self._show_ui_success(f"Cleanup selesai: {cleaned_count} direktori dibersihkan", button)
            else:
                self._handle_ui_error(result.get('message', 'Cleanup failed'), button)
                
        except Exception as e:
            self._handle_ui_error(f"Error saat cleanup: {str(e)}", button)