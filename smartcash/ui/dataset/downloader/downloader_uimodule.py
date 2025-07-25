"""
Dataset Downloader UIModule - New Core Pattern
Following new UIModule architecture with clean implementation.
"""
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.decorators import suppress_ui_init_logs
from smartcash.ui.dataset.downloader.components.downloader_ui import create_downloader_ui_components
from smartcash.ui.dataset.downloader.configs.downloader_defaults import get_default_downloader_config
from smartcash.ui.dataset.downloader.operations import (
    create_download_operation,
    create_check_operation,
    create_cleanup_operation
)

class DownloaderUIModule(BaseUIModule):
    """
    Dataset Downloader UIModule following BaseUIModule pattern.
    
    Features:
    - 📥 Dataset download from Roboflow
    - 🔍 Dataset validation and checking
    - 🧹 Dataset cleanup operations
    - 📊 Real-time progress tracking
    - 🇮🇩 Bahasa Indonesia interface
    """
    
    def __init__(self):
        """Initialize downloader module."""
        super().__init__(
            module_name='downloader',
            parent_module='dataset',
            enable_environment=True  # Enable environment management features
        )
        
        # Set required components for validation
        self._required_components = [
            'main_container',
            'header_container',
            'form_container', 
            'action_container',
            'summary_container',
            'operation_container'
        ]
        
        self.log_debug("✅ DownloaderUIModule initialized")
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components for Downloader module (BaseUIModule requirement)."""
        try:
            self.log_debug("Creating Downloader UI components...")
            ui_components = create_downloader_ui_components(module_config=config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            self.log_debug(f"✅ Created {len(ui_components)} UI components")
            return ui_components
            
        except Exception as e:
            self.log_error(f"Failed to create UI components: {e}")
            raise
        
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get Downloader module-specific button handlers."""
        # Call parent method to get base handlers (save, reset)
        handlers = super()._get_module_button_handlers()
        
        # Add Downloader-specific handlers
        downloader_handlers = {
            'download': self._operation_download,
            'check': self._operation_check,
            'cleanup': self._operation_cleanup
        }
        
        handlers.update(downloader_handlers)
        return handlers
    
    
    def _extract_ui_config(self) -> Dict[str, Any]:
        """Extract configuration from UI components.
        
        Returns:
            Dictionary containing the extracted configuration
        """
        try:
            if not hasattr(self, '_config_handler') or not self._config_handler:
                self.log("❌ Config handler not available for UI config extraction", "warning")
                return {}
                
            # Delegate to config handler's extract_config method
            if hasattr(self._config_handler, 'extract_config'):
                return self._config_handler.extract_config(self._ui_components)
                
            self.log("❌ Config handler does not implement extract_config", "warning")
            return {}
            
        except Exception as e:
            self.log(f"❌ Error extracting UI config: {e}", "error")
            return {}
    
    def _operation_download(self, button=None) -> Dict[str, Any]:
        """Handle download operation with confirmation dialog for existing data."""
        def validate_download():
            return {'valid': True}
        
        def execute_download_with_confirmation():
            return self._execute_download_with_confirmation()
        
        return self._execute_operation_with_wrapper(
            operation_name="Download Dataset",
            operation_func=execute_download_with_confirmation,
            button=button,
            validation_func=validate_download,
            success_message="Download dataset berhasil diselesaikan",
            error_message="Download dataset gagal"
        )
    
    def _operation_check(self, button=None) -> Dict[str, Any]:
        """Handle check operation with proper button state management."""
        def validate_check():
            return {'valid': True}
        
        def execute_check():
            self.log("🔍 Memeriksa status dataset...", "info")
            
            # Create check operation with summary callback
            current_config = self.get_current_config()
            callbacks = {'on_success': self._update_operation_summary}
            operation = create_check_operation(self, current_config, callbacks)
            result = operation.execute()
            
            if result.get("success", False):
                file_count = result.get("file_count", 0)
                success_message = f"Pengecekan selesai: {file_count} file"
                return {'success': True, 'message': success_message, 'file_count': file_count}
            else:
                error_msg = result.get("error", "Pengecekan gagal")
                return {'success': False, 'message': error_msg}
        
        return self._execute_operation_with_wrapper(
            operation_name="Cek Dataset",
            operation_func=execute_check,
            button=button,
            validation_func=validate_check,
            success_message="Pengecekan dataset berhasil diselesaikan",
            error_message="Pengecekan dataset gagal"
        )
    
    def _operation_cleanup(self, button=None) -> Dict[str, Any]:
        """Handle cleanup operation with confirmation dialog."""
        def validate_cleanup():
            return {'valid': True}
        
        def execute_cleanup_with_confirmation():
            return self._execute_cleanup_with_confirmation()
        
        return self._execute_operation_with_wrapper(
            operation_name="Pembersihan Dataset",
            operation_func=execute_cleanup_with_confirmation,
            button=button,
            validation_func=validate_cleanup,
            success_message="Pembersihan dataset berhasil diselesaikan",
            error_message="Pembersihan dataset gagal"
        )
    
    def _execute_download_with_confirmation(self) -> Dict[str, Any]:
        """Execute download with user confirmation if existing data is detected."""
        try:
            # Check for existing data
            existing_data = self._check_existing_data()
            
            # Build confirmation message
            base_message = "Anda akan memulai download dataset dari Roboflow.\n\n"
            
            if existing_data['has_data']:
                total_files = existing_data['total_files']
                total_size = existing_data['total_size_formatted']
                message = (
                    base_message +
                    f"⚠️ PERHATIAN: Ditemukan {total_files:,} file yang sudah ada ({total_size})!\n"
                    f"Download ini akan menimpa data yang ada dan tidak dapat diurungkan.\n\n"
                    f"Detail data yang akan ditimpa:\n" +
                    self._format_existing_data_details(existing_data) +
                    f"\n\nYakin ingin melanjutkan download?"
                )
                danger_mode = True
                confirm_text = "Ya, Timpa Data yang Ada"
            else:
                message = base_message + "Lanjutkan dengan download dataset?"
                danger_mode = False
                confirm_text = "Lanjutkan"
            
            return self._show_confirmation_dialog(
                title="Konfirmasi Download Dataset",
                message=message,
                confirm_action=self._execute_download_operation,
                confirm_text=confirm_text,
                danger_mode=danger_mode
            )
        except Exception as e:
            return {'success': False, 'message': f"Gagal memeriksa status data: {e}"}

    def _execute_cleanup_with_confirmation(self) -> Dict[str, Any]:
        """Execute cleanup with user confirmation showing detailed file information."""
        try:
            # Get cleanup targets first
            current_config = self.get_current_config()
            callbacks = {'on_success': self._update_operation_summary}
            operation = create_cleanup_operation(self, current_config, callbacks)
            
            targets_result = operation.get_cleanup_targets()
            
            if not targets_result.get("success", False):
                error_msg = targets_result.get("error", "Gagal mendapatkan target pembersihan")
                return {'success': False, 'message': error_msg}
            
            total_files = targets_result.get("summary", {}).get("total_files", 0)
            if total_files == 0:
                return self._show_info_dialog(
                    title="Tidak Ada untuk Dibersihkan",
                    message="Tidak ada file dataset yang ditemukan untuk dibersihkan."
                )
            
            total_size = targets_result.get("summary", {}).get("size_formatted", "0B")
            targets = targets_result.get("targets", {})
            
            # Build detailed cleanup message
            message = (
                f"Anda akan menghapus {total_files:,} file dataset ({total_size}).\n\n"
                f"⚠️ Operasi ini tidak dapat dibatalkan!\n\n"
                f"Detail file yang akan dihapus:\n" +
                self._format_cleanup_targets_details(targets) +
                f"\n\nLanjutkan dengan pembersihan?"
            )
            
            def execute_cleanup():
                return self._execute_cleanup_operation_with_targets(operation, targets_result)
            
            return self._show_confirmation_dialog(
                title="⚠️ Konfirmasi Pembersihan Dataset", 
                message=message,
                confirm_action=execute_cleanup,
                confirm_text=f"🗑️ Hapus {total_files:,} File",
                danger_mode=True
            )
        except Exception as e:
            return {'success': False, 'message': f"Gagal memeriksa target pembersihan: {e}"}

    def _check_existing_data(self) -> Dict[str, Any]:
        """Check for existing data in the dataset directories using the scanner."""
        try:
            # Use the existing dataset scanner to check for data
            from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
            
            scanner = create_dataset_scanner(self.logger)
            scan_result = scanner.scan_existing_dataset_parallel()
            
            if scan_result.get('status') == 'success':
                summary = scan_result.get('summary', {})
                total_images = summary.get('total_images', 0)
                download_files = summary.get('download_files', 0)
                total_files = total_images + download_files
                
                # Get size information from splits
                splits_result = scan_result.get('splits', {})
                total_size = 0
                for split_data in splits_result.values():
                    total_size += split_data.get('total_size', 0)
                
                # Add downloads size
                downloads_result = scan_result.get('downloads', {})
                total_size += downloads_result.get('total_size', 0)
                
                return {
                    'has_data': total_files > 0,
                    'total_files': total_files,
                    'total_images': total_images,
                    'download_files': download_files,
                    'total_size': total_size,
                    'total_size_formatted': self._format_file_size(total_size),
                    'splits': splits_result,
                    'downloads': downloads_result
                }
            else:
                # Scanner failed or found no data
                return {
                    'has_data': False,
                    'total_files': 0,
                    'total_images': 0,
                    'download_files': 0,
                    'total_size': 0,
                    'total_size_formatted': '0B',
                    'splits': {},
                    'downloads': {}
                }
        except Exception as e:
            self.log_error(f"Error checking existing data: {e}")
            # Return safe defaults on error
            return {
                'has_data': False,
                'total_files': 0,
                'total_images': 0,
                'download_files': 0,
                'total_size': 0,
                'total_size_formatted': '0B',
                'splits': {},
                'downloads': {}
            }

    def _format_existing_data_details(self, existing_data: Dict[str, Any]) -> str:
        """Format existing data details for display in confirmation dialog."""
        details = []
        
        # Add split information
        splits = existing_data.get('splits', {})
        for split_name, split_data in splits.items():
            if split_data.get('images', 0) > 0 or split_data.get('labels', 0) > 0:
                images = split_data.get('images', 0)
                labels = split_data.get('labels', 0)
                details.append(f"• {split_name.title()}: {images} gambar, {labels} label")
        
        # Add downloads information
        downloads = existing_data.get('downloads', {})
        if downloads.get('file_count', 0) > 0:
            file_count = downloads.get('file_count', 0)
            details.append(f"• Downloads: {file_count} file")
        
        return '\n'.join(details) if details else "• Data terdeteksi tetapi detail tidak tersedia"

    def _format_cleanup_targets_details(self, targets: Dict[str, Any]) -> str:
        """Format cleanup targets details for display in confirmation dialog."""
        details = []
        
        for target_name, target_info in targets.items():
            file_count = target_info.get('file_count', 0)
            if file_count > 0:
                # Make target names more user-friendly
                friendly_name = target_name.replace('_', ' ').title()
                details.append(f"• {friendly_name}: {file_count:,} file")
        
        return '\n'.join(details) if details else "• Tidak ada detail target tersedia"

    def _execute_download_operation(self) -> Dict[str, Any]:
        """Execute the actual download operation."""
        try:
            self.log("📥 Memulai download dataset...", "info")
            
            # Create download operation with merged config
            ui_config = self._extract_ui_config()
            current_config = self.get_current_config()
            merged_config = {**current_config, **ui_config}
            
            # Create download operation with summary callback
            callbacks = {'on_success': self._update_operation_summary}
            operation = create_download_operation(self, merged_config, callbacks)
            result = operation.execute()
            
            if result.get("success", False):
                file_count = result.get("file_count", 0)
                success_message = f"Download selesai: {file_count} file"
                return {'success': True, 'message': success_message, 'file_count': file_count}
            else:
                error_msg = result.get("error", "Download gagal")
                return {'success': False, 'message': error_msg}
        except Exception as e:
            return {'success': False, 'message': f"Error download: {e}"}

    def _execute_cleanup_operation_with_targets(self, operation, targets_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the cleanup operation with pre-scanned targets."""
        try:
            self.log("🧹 Memulai pembersihan dataset...", "info")
            
            result = operation.execute(targets_result.get("targets"))
            
            if result.get("success", False):
                deleted_files = result.get("deleted_files", 0)
                freed_space = result.get("freed_space", "0B")
                success_message = f"Pembersihan selesai: {deleted_files} file dihapus ({freed_space})"
                return {'success': True, 'message': success_message, 'deleted_files': deleted_files, 'freed_space': freed_space}
            else:
                error_msg = result.get("error", "Pembersihan gagal")
                return {'success': False, 'message': error_msg}
        except Exception as e:
            return {'success': False, 'message': f"Error pembersihan: {e}"}

    def _show_confirmation_dialog(self, title: str, message: str, confirm_action, 
                                 confirm_text: str = "Confirm", danger_mode: bool = False) -> Dict[str, Any]:
        """Show confirmation dialog and execute action on confirm."""
        op_container = self.get_component('operation_container')
        if op_container and hasattr(op_container, 'show_dialog'):
            op_container.show_dialog(
                title=title,
                message=message,
                on_confirm=confirm_action,
                confirm_text=confirm_text,
                cancel_text="Batal",
                danger_mode=danger_mode
            )
            return {'success': True, 'message': 'Dialog konfirmasi ditampilkan'}
        else:
            self.log_error("Dialog not available, cannot proceed with confirmation required action")
            return {'success': False, 'message': 'Dialog konfirmasi tidak tersedia. Operasi tidak dapat dilanjutkan.'}

    def _show_info_dialog(self, title: str, message: str) -> Dict[str, Any]:
        """Show information dialog."""
        op_container = self.get_component('operation_container')
        if op_container and hasattr(op_container, 'show_dialog'):
            op_container.show_dialog(
                title=title,
                message=message,
                confirm_text="OK",
                on_cancel=None
            )
        return {'success': True, 'message': message}

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size from bytes."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    
    def _get_selected_packages(self) -> List[str]:
        """Get list of selected packages from UI."""
        # Implementation here
        return []
    
    def _update_operation_summary(self, content: str) -> None:
        """Update the operation summary container with new content."""
        try:
            summary_container = self.get_component('summary_container')
            if summary_container and hasattr(summary_container, 'set_content'):
                formatted_content = f"""
                <div style="padding: 10px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
                    <h4 style="color: #2c3e50; margin: 0 0 10px 0;">📊 Operation Summary</h4>
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745;">
                        {content}
                    </div>
                </div>
                """
                summary_container.set_content(formatted_content)
                self.log_debug("✅ Summary container updated with operation results")
            else:
                # Fallback to operation summary updater
                updater = self.get_component('operation_summary_updater')
                if updater and callable(updater):
                    updater(content)
                else:
                    self.log_warning("Summary container tidak ditemukan atau tidak dapat dipanggil.")
        except Exception as e:
            self.log_error(f"Failed to update operation summary: {e}")
        
    def cleanup(self) -> None:
        """Widget lifecycle cleanup - optimization.md compliance."""
        try:
            # Cleanup any downloader-specific resources
            if hasattr(self, '_download_status'):
                self._download_status.clear()
            
            # Cleanup UI components if they have cleanup methods
            if hasattr(self, '_ui_components') and self._ui_components:
                # Call component-specific cleanup if available
                if hasattr(self._ui_components, '_cleanup'):
                    self._ui_components._cleanup()
                
                # Close individual widgets
                for component_name, component in self._ui_components.items():
                    if hasattr(component, 'close'):
                        try:
                            component.close()
                        except Exception:
                            pass  # Ignore cleanup errors
            
            # Call parent cleanup
            if hasattr(super(), 'cleanup'):
                super().cleanup()
            
            # Minimal logging for cleanup completion
            if hasattr(self, 'logger'):
                self.logger.info("Downloader module cleanup completed")
                
        except Exception as e:
            # Critical errors always logged
            if hasattr(self, 'logger'):
                self.logger.error(f"Downloader module cleanup failed: {e}")
    
    def __del__(self):
        """Memory management - ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during deletion
    
    def create_config_handler(self, config: Dict[str, Any]):
        """Create config handler instance for Downloader module (BaseUIModule requirement)."""
        from smartcash.ui.dataset.downloader.configs.downloader_config_handler import DownloaderConfigHandler
        handler = DownloaderConfigHandler(config)
        return handler
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Downloader module (BaseUIModule requirement)."""
        return get_default_downloader_config()
