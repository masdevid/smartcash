"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: FIXED download handler dengan proper state management, log clearing, dan parameter fixes
"""

from typing import Dict, Any, Callable, Optional
from pathlib import Path
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.ui.components.dialogs import confirm
from smartcash.dataset.downloader import get_downloader_instance, create_ui_compatible_config


class DownloadHandler:
    """FIXED handler dengan proper state management dan log clearing"""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any], logger):
        self.ui_components = ui_components
        self.config = config
        self.logger = logger
        self.progress_tracker = ui_components.get('progress_tracker')
        
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup handlers dengan proper binding dan state management"""
        handlers = [
            ('download_button', self._handle_download_click),
            ('check_button', self._handle_check_click),
            ('cleanup_button', self._handle_cleanup_click)
        ]
        
        for button_name, handler in handlers:
            button = self.ui_components.get(button_name)
            if button:
                self._clear_button_handlers(button)
                button.on_click(handler)
        
        self.logger.success("✅ Handlers berhasil disetup dengan state management")
        return self.ui_components
    
    def _clear_button_handlers(self, button) -> None:
        """Clear existing handlers"""
        try:
            if hasattr(button, '_click_handlers') and hasattr(button._click_handlers, 'callbacks'):
                button._click_handlers.callbacks.clear()
        except Exception:
            pass
    
    def _prepare_button_state(self, active_button) -> None:
        """Prepare button state dan clear log"""
        # Clear log output
        log_output = self.ui_components.get('log_output')
        if log_output and hasattr(log_output, 'clear_output'):
            with log_output:
                log_output.clear_output(wait=True)
        
        # Disable semua buttons
        all_buttons = [
            self.ui_components.get('download_button'),
            self.ui_components.get('check_button'), 
            self.ui_components.get('cleanup_button')
        ]
        
        for btn in all_buttons:
            if btn and hasattr(btn, 'disabled'):
                btn.disabled = True
    
    def _restore_button_state(self) -> None:
        """Restore button state setelah operation"""
        all_buttons = [
            self.ui_components.get('download_button'),
            self.ui_components.get('check_button'),
            self.ui_components.get('cleanup_button')
        ]
        
        for btn in all_buttons:
            if btn and hasattr(btn, 'disabled'):
                btn.disabled = False
    
    def _handle_download_click(self, button) -> None:
        """FIXED download handler dengan proper parameter handling"""
        try:
            self._prepare_button_state(button)
            
            if self.progress_tracker:
                self.progress_tracker.show("Dataset Download")
                self.progress_tracker.update_overall(10, "🔧 Validating configuration...")
            
            # Extract dan validate config
            config_handler = self.ui_components.get('config_handler')
            if not config_handler:
                self._handle_error("Config handler tidak tersedia", button)
                return
            
            ui_config = config_handler.extract_config_from_ui(self.ui_components)
            
            # FIXED: Ensure rename_files is boolean, not callable
            service_config = create_ui_compatible_config(ui_config)
            service_config['rename_files'] = bool(service_config.get('rename_files', True))  # Force boolean
            service_config['organize_dataset'] = bool(service_config.get('organize_dataset', True))
            service_config['validate_download'] = bool(service_config.get('validate_download', True))
            service_config['backup_existing'] = bool(service_config.get('backup_existing', False))
            
            # Validate config
            validation = config_handler.validate_config(ui_config)
            if not validation['valid']:
                error_msg = f"Config tidak valid: {'; '.join(validation['errors'])}"
                self._handle_error(error_msg, button)
                return
            
            if self.progress_tracker:
                self.progress_tracker.update_overall(30, "✅ Configuration valid")
            
            # Check existing dataset
            has_existing = self._check_existing_dataset_quick()
            
            if has_existing:
                self._show_download_confirmation(service_config, button)
            else:
                self._execute_download(service_config, button)
                
        except Exception as e:
            self._handle_error(f"Error download handler: {str(e)}", button)
    
    def _execute_download(self, service_config: Dict[str, Any], button) -> None:
        """FIXED execute download dengan proper parameter validation"""
        try:
            # Validate required fields
            required = ['workspace', 'project', 'version', 'api_key']
            missing = [f for f in required if not service_config.get(f, '').strip()]
            
            if missing:
                self._handle_error(f"Field wajib kosong: {', '.join(missing)}", button)
                return
            
            if self.progress_tracker:
                self.progress_tracker.update_overall(40, "🏭 Creating download service...")
            
            # Log service config untuk debugging
            self.logger.info(f"🔧 Service config: rename_files={service_config.get('rename_files')}, type={type(service_config.get('rename_files'))}")
            
            # Create service dengan config yang sudah validated
            downloader = get_downloader_instance(service_config, self.logger)
            if not downloader:
                self._handle_error("Gagal membuat download service", button)
                return
            
            # Setup progress callback
            if hasattr(downloader, 'set_progress_callback'):
                downloader.set_progress_callback(self._create_progress_callback())
            
            if self.progress_tracker:
                self.progress_tracker.update_overall(50, "📥 Starting download...")
            
            # Execute download
            result = downloader.download_dataset()
            
            # Handle response
            if result and result.get('status') == 'success':
                stats = result.get('stats', {})
                total_images = stats.get('total_images', 0)
                success_msg = f"Dataset berhasil didownload: {total_images:,} gambar"
                
                if self.progress_tracker:
                    self.progress_tracker.complete(success_msg)
                
                show_status_safe(success_msg, "success", self.ui_components)
                self.logger.success(f"✅ {success_msg}")
                
                # Log additional stats jika tersedia
                if stats.get('uuid_renamed'):
                    naming_stats = stats.get('naming_stats', {})
                    if naming_stats:
                        self.logger.info(f"🔄 UUID renaming: {naming_stats.get('total_files', 0)} files processed")
                
            else:
                error_msg = f"Download gagal: {result.get('message', 'Unknown error') if result else 'No response from service'}"
                self._handle_error(error_msg, button)
                
        except Exception as e:
            self._handle_error(f"Error saat download: {str(e)}", button)
        finally:
            self._restore_button_state()
    
    def _create_progress_callback(self) -> Callable[[str, int, int, str], None]:
        """Create progress callback dengan signature yang standardized"""
        def progress_callback(step: str, current: int, total: int, message: str):
            """Standardized progress callback: (step, current, total, message)"""
            if self.progress_tracker and total > 0:
                # Map download progress ke overall progress (50-90%)
                base_progress = 50
                download_range = 40
                percentage = base_progress + int((current / total) * download_range)
                self.progress_tracker.update_overall(percentage, f"{step}: {message}")
        
        return progress_callback
    
    def _show_download_confirmation(self, service_config: Dict[str, Any], button) -> None:
        """Show confirmation dialog"""
        workspace = service_config.get('workspace', '')
        project = service_config.get('project', '')
        version = service_config.get('version', '')
        
        dataset_id = f"{workspace}/{project}:v{version}"
        
        message = f"""Dataset existing akan ditimpa!

🎯 Target: {dataset_id}
🔄 UUID Renaming: {'✅' if service_config.get('rename_files', True) else '❌'}
✅ Validasi: {'✅' if service_config.get('validate_download', True) else '❌'}
💾 Backup: {'✅' if service_config.get('backup_existing', False) else '❌'}

Lanjutkan download?"""
        
        confirm(
            "Konfirmasi Download Dataset", 
            message,
            on_yes=lambda btn: self._execute_download(service_config, button),
            on_no=lambda btn: (
                self.logger.info("🚫 Download dibatalkan"),
                self._restore_button_state(),
                self.progress_tracker and self.progress_tracker.hide()
            )
        )
    
    def _handle_check_click(self, button) -> None:
        """Handle check button click dengan log clearing"""
        try:
            self._prepare_button_state(button)
            
            if self.progress_tracker:
                self.progress_tracker.show("Check Dataset")
                self.progress_tracker.update_overall(50, "🔍 Scanning dataset...")
            
            dataset_status = self._check_existing_dataset_detailed()
            
            if self.progress_tracker:
                self.progress_tracker.update_overall(100, "✅ Check completed")
                self.progress_tracker.complete("Dataset check selesai")
            
            self._display_dataset_summary(dataset_status)
            
        except Exception as e:
            self._handle_error(f"Error check handler: {str(e)}", button)
        finally:
            self._restore_button_state()
    
    def _handle_cleanup_click(self, button) -> None:
        """Handle cleanup button click dengan log clearing"""
        try:
            self._prepare_button_state(button)
            
            cleanup_info = self._scan_cleanup_directories()
            
            if not cleanup_info['has_files']:
                self.logger.info("✨ Tidak ada file untuk dibersihkan")
                self._restore_button_state()
                return
            
            confirm(
                "Konfirmasi Cleanup Dataset",
                self._create_cleanup_message(cleanup_info),
                on_yes=lambda btn: self._execute_cleanup(cleanup_info),
                on_no=lambda btn: (
                    self.logger.info("🚫 Cleanup dibatalkan"),
                    self._restore_button_state()
                )
            )
            
        except Exception as e:
            self._handle_error(f"Error cleanup handler: {str(e)}", button)
            self._restore_button_state()
    
    def _check_existing_dataset_quick(self) -> bool:
        """Quick check existing dataset"""
        data_path = Path('data')
        return any(
            (data_path / split).exists() and 
            len(list((data_path / split).rglob('*'))) > 0 
            for split in ['train', 'valid', 'test']
        )
    
    def _check_existing_dataset_detailed(self) -> Dict[str, Any]:
        """Detailed check existing dataset"""
        data_path = Path('data')
        dataset_dirs = ['train', 'valid', 'test']
        
        dataset_status = {
            'data_dir_exists': data_path.exists(),
            'splits': {},
            'downloads': self._check_downloads_dir(),
            'total_files': 0,
            'has_dataset': False
        }
        
        for split in dataset_dirs:
            split_status = self._check_split_directory(data_path / split)
            dataset_status['splits'][split] = split_status
            dataset_status['total_files'] += split_status['total_files']
            
            if split_status['has_data']:
                dataset_status['has_dataset'] = True
        
        return dataset_status
    
    def _check_split_directory(self, split_path: Path) -> Dict[str, Any]:
        """Check single split directory"""
        if not split_path.exists():
            return {'exists': False, 'has_data': False, 'images': 0, 'labels': 0, 'total_files': 0}
        
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        image_count = len(list(images_dir.glob('*'))) if images_dir.exists() else 0
        label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        
        return {
            'exists': True,
            'has_data': image_count > 0 or label_count > 0,
            'images': image_count,
            'labels': label_count,
            'total_files': image_count + label_count,
            'images_dir_exists': images_dir.exists(),
            'labels_dir_exists': labels_dir.exists()
        }
    
    def _check_downloads_dir(self) -> Dict[str, Any]:
        """Check downloads directory"""
        downloads_path = Path('data/downloads')
        
        if not downloads_path.exists():
            return {'exists': False, 'files': 0, 'size_mb': 0}
        
        try:
            all_files = list(downloads_path.rglob('*'))
            files_only = [f for f in all_files if f.is_file()]
            total_size = sum(f.stat().st_size for f in files_only if f.exists())
            
            return {
                'exists': True,
                'files': len(files_only),
                'size_mb': total_size / (1024 * 1024)
            }
        except Exception:
            return {'exists': True, 'files': 0, 'size_mb': 0, 'error': True}
    
    def _display_dataset_summary(self, dataset_status: Dict[str, Any]) -> None:
        """Display dataset summary"""
        if not dataset_status['data_dir_exists']:
            self.logger.info("📁 Data directory belum ada - siap untuk download pertama")
            return
        
        if not dataset_status['has_dataset']:
            self.logger.info("📂 Data directory kosong - siap untuk download")
            return
        
        # Dataset exists - show summary
        self.logger.info("📊 Dataset Summary:")
        
        for split_name, split_info in dataset_status['splits'].items():
            if split_info['has_data']:
                self.logger.info(f"  • {split_name.title()}: {split_info['images']} images, {split_info['labels']} labels")
            else:
                self.logger.info(f"  • {split_name.title()}: Kosong")
        
        total_files = dataset_status['total_files']
        self.logger.info(f"📈 Total: {total_files} files")
        
        downloads = dataset_status['downloads']
        if downloads['exists'] and downloads['files'] > 0:
            self.logger.info(f"💾 Downloads: {downloads['files']} files ({downloads['size_mb']:.1f}MB)")
        
        if dataset_status['has_dataset']:
            self.logger.success("✅ Dataset sudah ada - download akan menimpa data existing")
        else:
            self.logger.info("🆕 Siap untuk download dataset baru")
    
    def _scan_cleanup_directories(self) -> Dict[str, Any]:
        """Scan directories untuk cleanup"""
        data_path = Path('data')
        cleanup_dirs = ['train', 'valid', 'test', 'downloads']
        
        scan_results = {
            'directories': {},
            'total_files': 0,
            'total_size_mb': 0,
            'has_files': False
        }
        
        for dir_name in cleanup_dirs:
            dir_path = data_path / dir_name
            dir_info = self._scan_directory(dir_path)
            
            scan_results['directories'][dir_name] = dir_info
            scan_results['total_files'] += dir_info['file_count']
            scan_results['total_size_mb'] += dir_info['size_mb']
            
            if dir_info['file_count'] > 0:
                scan_results['has_files'] = True
        
        return scan_results
    
    def _scan_directory(self, dir_path: Path) -> Dict[str, Any]:
        """Scan single directory"""
        if not dir_path.exists():
            return {'exists': False, 'file_count': 0, 'size_mb': 0}
        
        try:
            all_files = list(dir_path.rglob('*'))
            files_only = [f for f in all_files if f.is_file()]
            total_size = sum(f.stat().st_size for f in files_only if f.exists())
            
            return {
                'exists': True,
                'file_count': len(files_only),
                'size_mb': total_size / (1024 * 1024)
            }
        except Exception as e:
            return {'exists': True, 'file_count': 0, 'size_mb': 0, 'error': str(e)}
    
    def _create_cleanup_message(self, cleanup_info: Dict[str, Any]) -> str:
        """Create cleanup confirmation message"""
        total_files = cleanup_info['total_files']
        total_size = cleanup_info['total_size_mb']
        
        message_parts = [
            f"Total Files: {total_files}",
            f"Total Size: {total_size:.1f}MB",
            "",
            "Directories to clean:"
        ]
        
        for dir_name, dir_info in cleanup_info['directories'].items():
            if dir_info['file_count'] > 0:
                message_parts.append(f"• {dir_name.title()}: {dir_info['file_count']} files ({dir_info['size_mb']:.1f}MB)")
        
        return "\n".join(message_parts)
    
    def _execute_cleanup(self, cleanup_info: Dict[str, Any]) -> None:
        """Execute cleanup dengan progress"""
        try:
            if self.progress_tracker:
                self.progress_tracker.show("Cleanup Dataset")
                self.progress_tracker.update_overall(20, "🧹 Starting cleanup...")
            
            self.logger.info("🧹 Memulai cleanup...")
            
            import shutil
            cleaned_dirs = 0
            cleaned_files = 0
            total_dirs = len([d for d in cleanup_info['directories'].values() if d['file_count'] > 0])
            
            for i, (dir_name, dir_info) in enumerate(cleanup_info['directories'].items()):
                if not dir_info['exists'] or dir_info['file_count'] == 0:
                    continue
                
                dir_path = Path('data') / dir_name
                
                try:
                    if dir_path.exists():
                        shutil.rmtree(dir_path)
                        cleaned_dirs += 1
                        cleaned_files += dir_info['file_count']
                        self.logger.info(f"🗑️ Cleaned {dir_name}: {dir_info['file_count']} files")
                        
                        if self.progress_tracker:
                            progress = 20 + int((i + 1) / total_dirs * 70)
                            self.progress_tracker.update_overall(progress, f"🗑️ Cleaned {dir_name}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error cleaning {dir_name}: {str(e)}")
            
            if cleaned_files > 0:
                success_msg = f"Cleanup selesai: {cleaned_files} files dari {cleaned_dirs} directories"
                
                if self.progress_tracker:
                    self.progress_tracker.complete(success_msg)
                
                self.logger.success(f"✅ {success_msg}")
            else:
                self.logger.info("ℹ️ Tidak ada file yang dibersihkan")
                
                if self.progress_tracker:
                    self.progress_tracker.complete("No files to clean")
                
        except Exception as e:
            error_msg = f"Error during cleanup: {str(e)}"
            
            if self.progress_tracker:
                self.progress_tracker.error(error_msg)
            
            self.logger.error(f"❌ {error_msg}")
        finally:
            self._restore_button_state()
    
    def _handle_error(self, message: str, button) -> None:
        """Handle error dengan unified feedback"""
        if self.progress_tracker:
            self.progress_tracker.error(message)
        
        show_status_safe(f"❌ {message}", "error", self.ui_components)
        self.logger.error(f"❌ {message}")
        self._restore_button_state()


def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup download handlers dengan proper integration"""
    logger = ui_components.get('logger')
    
    try:
        # Create handler
        handler = DownloadHandler(ui_components, config, logger)
        
        # Setup handlers
        ui_components = handler.setup_handlers()
        
        # Add handler references
        ui_components.update({
            'download_handler': handler,
            'check_handler': handler, 
            'cleanup_handler': handler,
            'handler': handler
        })
        
        logger.success("✅ Download handlers berhasil disetup")
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error setup handlers: {str(e)}")
        return ui_components


# Export
__all__ = ['setup_download_handlers', 'DownloadHandler']