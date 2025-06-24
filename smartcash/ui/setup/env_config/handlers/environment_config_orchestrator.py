"""
File: smartcash/ui/setup/env_config/handlers/environment_config_orchestrator.py
Deskripsi: Complete orchestrator dengan DRY utils integration dan proper error handling
"""

from typing import Dict, Any, Tuple

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.setup.env_config.handlers.environment_status_checker import EnvironmentStatusChecker
from smartcash.ui.setup.env_config.handlers.drive_setup_handler import DriveSetupHandler
from smartcash.ui.setup.env_config.utils import (
    update_progress_safe, hide_progress_safe, reset_progress_safe,
    refresh_environment_state_silent, get_system_summary_minimal,
    get_status_message, get_progress_message
)

class EnvironmentConfigOrchestrator:
    """Complete orchestrator dengan DRY utils integration untuk environment setup"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = None
        self.status_checker = EnvironmentStatusChecker()
        self.drive_handler = DriveSetupHandler(ui_components)
        self._init_environment_manager_silent()
    
    def _init_environment_manager_silent(self):
        """Initialize environment manager tanpa logging premature - one-liner"""
        try:
            from smartcash.ui.setup.env_config.helpers.silent_environment_manager import get_silent_environment_manager
            self.env_manager = get_silent_environment_manager()
            self.ui_components['env_manager'] = self.env_manager
        except Exception:
            self.env_manager = None
    
    def init_logger(self):
        """Initialize logger setelah UI ready dengan environment details"""
        if self.logger is None:
            try:
                self.logger = create_ui_logger_bridge(self.ui_components, "env_config")
                self._log_environment_details()
                return self.logger
            except Exception as e:
                error_msg = f"Gagal menginisialisasi logger: {str(e)}"
                if 'status_message' in self.ui_components:
                    self.ui_components['status_message'].value = f"⚠️ {error_msg}"
                raise RuntimeError(error_msg) from e
        return self.logger
    
    def check_environment_status(self) -> Dict[str, Any]:
        """Check status environment dengan silent Drive refresh dan minimal logging"""
        try:
            if self.logger is None:
                self.init_logger()
            
            # Silent refresh Drive state
            refresh_environment_state_silent(self.env_manager)
            
            # Get status dengan refreshed state
            status = self.status_checker.get_environment_status(self.env_manager)
            
            # Log status
            if self.logger and 'status_message' in status:
                self.logger.info(f"Status environment: {status['status_message']}")
            
            return status
            
        except Exception as e:
            error_msg = f"Gagal memeriksa status environment: {str(e)}"
            if self.logger:
                self.logger.error(error_msg, exc_info=True)
            if 'status_message' in self.ui_components:
                self.ui_components['status_message'].value = f"⚠️ {error_msg}"
            return {'status': 'error', 'status_message': error_msg}
            
    def _log_environment_details(self):
        """Log environment details dengan format yang rapi"""
        if not self.logger:
            return
            
        try:
            # Get comprehensive status
            status = self.status_checker.get_comprehensive_status()
            
            # Log environment details with proper formatting
            self.logger.info("=" * 40)
            self.logger.info("ENVIRONMENT STATUS")
            self.logger.info("=" * 40)
            
            # Log basic info
            self.logger.info(f"Status: {status.get('status', 'unknown')}")
            self.logger.info(f"Pesan: {status.get('status_message', 'Tidak ada pesan')}")
            
            # Log system info if available
            if 'system_info' in status:
                self.logger.info("\n" + "-" * 40)
                self.logger.info("SISTEM INFORMASI")
                self.logger.info("-" * 40)
                sys_info = status['system_info']
                for key, value in sys_info.items():
                    self.logger.info(f"{key}: {value}")
            
            self.logger.info("=" * 40 + "\n")
            
            # Check for any issues and log them
            if not status.get('ready', False):
                missing_count = (len(status.get('missing_drive_folders', [])) +
                              len(status.get('missing_drive_configs', [])) +
                              len(status.get('missing_files', [])))
                
                if missing_count > 0:
                    self.logger.warning(f"⚠️ Ditemukan {missing_count} masalah yang perlu diperbaiki")
                    
                    # Log missing items with proper indentation
                    if 'missing_drive_folders' in status and status['missing_drive_folders']:
                        self.logger.warning("  Folder yang hilang:")
                        for folder in status['missing_drive_folders']:
                            self.logger.warning(f"    - {folder}")
                            
                    if 'missing_drive_configs' in status and status['missing_drive_configs']:
                        self.logger.warning("  Konfigurasi yang hilang:")
                        for config in status['missing_drive_configs']:
                            self.logger.warning(f"    - {config}")
                            
                    if 'missing_files' in status and status['missing_files']:
                        self.logger.warning("  File yang hilang:")
                        for file in status['missing_files']:
                            self.logger.warning(f"    - {file}")
                    
                    self.logger.info("")
                    
            return True
                    
        except Exception as e:
            error_msg = f"Gagal mencatat detail environment: {str(e)}"
            if self.logger:
                self.logger.error(error_msg, exc_info=True)
            return False
            
    def _log_setup_required(self, status):
        """Log pesan setup yang diperlukan"""
        if not self.logger:
            return
            
        try:
            missing_items = []
            
            # Collect all missing items
            if 'missing_drive_folders' in status and status['missing_drive_folders']:
                missing_items.extend([f"Folder: {f}" for f in status['missing_drive_folders']])
                
            if 'missing_drive_configs' in status and status['missing_drive_configs']:
                missing_items.extend([f"Konfigurasi: {c}" for c in status['missing_drive_configs']])
                
            if 'missing_files' in status and status['missing_files']:
                missing_items.extend([f"File: {f}" for f in status['missing_files']])
                
            if 'invalid_symlinks' in status and status['invalid_symlinks']:
                missing_items.extend([f"Symlink tidak valid: {s}" for s in status['invalid_symlinks']])
            
            # Log the summary
            if missing_items:
                self.logger.warning("🔧 Setup diperlukan:")
                for item in missing_items:
                    self.logger.warning(f"  - {item}")
                return True
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Gagal mencatat item setup yang diperlukan: {str(e)}", exc_info=True)
                
        return False
    
    def perform_environment_setup(self) -> bool:
        """Environment setup dengan complete progress tracking dan error handling"""
        if self.logger is None:
            self.logger = self.init_logger()
        
        try:
            # Initialize progress tracking
            if 'progress_tracker' not in self.ui_components:
                self.logger.error("Progress tracker tidak tersedia di UI components")
                return False
                
            progress_tracker = self.ui_components['progress_tracker']
            progress_tracker.reset()
            progress_tracker.visible = True
            
            # Update UI status
            if 'status_message' in self.ui_components:
                self.ui_components['status_message'].value = "🚀 Memulai setup environment..."
            
            self.logger.info("=" * 50)
            self.logger.info("🚀 MEMULAI SETUP ENVIRONMENT")
            self.logger.info("=" * 50)
            
            # Step 1: Check current environment status
            self._update_progress("Memeriksa status environment...", 10)
            self.logger.info("🔍 Memeriksa status environment...")
            
            # Refresh environment state
            refresh_environment_state_silent(self.env_manager)
            status = self.status_checker.get_comprehensive_status()
            
            if status.get('ready', False):
                self.logger.info("✅ Environment sudah siap digunakan")
                self._update_progress("✅ Environment sudah siap", 100)
                return True
            
            # Step 2: Setup Google Drive
            self._update_progress("Menyiapkan Google Drive...", 20)
            self.logger.info("🔍 Memeriksa koneksi Google Drive...")
            
            drive_setup_success = self.drive_handler.setup_google_drive()
            if not drive_setup_success:
                error_msg = "❌ Gagal menyiapkan Google Drive"
                self._update_progress(error_msg, 0, is_error=True)
                self.logger.error(error_msg)
                return False
            
            self.logger.info("✅ Google Drive berhasil disiapkan")
            
            # Step 3: Validate Drive path
            drive_path = status.get('drive', {}).get('path')
            if not drive_path and self.env_manager:
                try:
                    drive_path = self.env_manager.get_drive_path()
                    if drive_path:
                        status['drive']['path'] = drive_path
                except Exception as e:
                    self.logger.warning(f"Peringatan saat mendapatkan path Drive: {str(e)}")
            
            if not drive_path:
                error_msg = "❌ Tidak dapat menentukan path Google Drive"
                self._update_progress(error_msg, 0, is_error=True)
                self.logger.error(error_msg)
                return False
            
            self.logger.info(f"🎯 Target setup path: {drive_path}")
            
            # Step 4: Perform complete setup
            self._update_progress("Melakukan setup lengkap...", 50)
            self.logger.info("🔧 Melakukan setup lengkap...")
            
            setup_results = self.drive_handler.perform_complete_setup(drive_path)
            self._log_setup_results(setup_results)
            
            # Step 5: Initialize managers
            self._update_progress("Menginisialisasi komponen...", 80)
            self.logger.info("⚙️ Menginisialisasi komponen...")
            
            self._initialize_managers()
            
            # Step 6: Final verification
            self._update_progress("Memverifikasi setup...", 90)
            self.logger.info("🔍 Memverifikasi setup...")
            
            refresh_environment_state_silent(self.env_manager)
            final_status = self.status_checker.get_comprehensive_status()
            
            # Log final status
            self._log_environment_details()
            
            if final_status.get('ready', False) or setup_results.get('success', False):
                success_msg = "✅ Setup environment berhasil!"
                self._update_progress(success_msg, 100)
                self.logger.info("=" * 50)
                self.logger.info("✅ SETUP ENVIRONMENT BERHASIL")
                self.logger.info("=" * 50)
                
                if 'status_message' in self.ui_components:
                    self.ui_components['status_message'].value = success_msg
                
                return True
            else:
                warning_msg = "⚠️ Setup selesai dengan beberapa komponen belum optimal"
                self._update_progress(warning_msg, 100, is_warning=True)
                self.logger.warning(warning_msg)
                self.logger.info("💡 Environment tetap bisa digunakan untuk development")
                
                if 'status_message' in self.ui_components:
                    self.ui_components['status_message'].value = warning_msg
                
                return True
            
        except Exception as e:
            error_msg = f"❌ Terjadi kesalahan saat setup environment: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._update_progress(error_msg, 0, is_error=True)
            
            if 'status_message' in self.ui_components:
                self.ui_components['status_message'].value = error_msg
                
            return False
    
    def _update_progress(self, message: str, progress: int, is_error: bool = False, is_warning: bool = False):
        """Update progress bar and log message with appropriate formatting"""
        try:
            # Update progress bar if available
            if 'progress_tracker' in self.ui_components and progress >= 0:
                progress_tracker = self.ui_components['progress_tracker']
                progress_tracker.value = progress
                progress_tracker.description = f"{progress}% - {message}"
                
                # Update progress bar color based on status
                if is_error:
                    progress_tracker.bar_style = 'danger'
                elif is_warning:
                    progress_tracker.bar_style = 'warning'
                elif progress >= 100:
                    progress_tracker.bar_style = 'success'
                else:
                    progress_tracker.bar_style = 'info'
            
            # Log the message with appropriate level
            if self.logger:
                if is_error:
                    self.logger.error(message)
                elif is_warning:
                    self.logger.warning(message)
                else:
                    self.logger.info(message)
                    
            # Force UI update
            if hasattr(self, '_last_progress') and self._last_progress == progress:
                # Small adjustment to force UI update
                self.ui_components['progress_tracker'].value = progress - 0.1
                self.ui_components['progress_tracker'].value = progress
            
            self._last_progress = progress
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Gagal memperbarui progress: {str(e)}", exc_info=True)
    
    def _log_current_status_minimal(self, status: Dict[str, Any]):
        """Log current status dengan format minimal untuk mengurangi noise"""
        try:
            if not self.logger:
                return
                
            # Hanya log info essential dengan one-liner
            drive_info = status.get('drive', {})
            if drive_info.get('type') == 'colab':
                mount_status = "✅" if drive_info.get('mounted') else "❌"
                self.logger.info(f"💾 Drive: {mount_status}")
            
            # Summary missing items tanpa detail berlebihan
            missing_total = (len(status.get('missing_drive_folders', [])) +
                           len(status.get('missing_drive_configs', [])) +
                           len(status.get('invalid_symlinks', [])))
            
            if missing_total > 0:
                self.logger.info(f"🔧 Missing items: {missing_total}")
        except Exception:
            pass
    
    def _log_setup_results(self, results: Dict[str, Any]):
        """Log hasil setup dengan detail informatif menggunakan one-liner"""
        try:
            if not self.logger:
                return
                
            # Log results dengan one-liner pattern
            result_summaries = [
                f"📁 Folders: {sum(results['folders'].values())}/{len(results['folders'])} berhasil dibuat" if 'folders' in results else "",
                f"📋 Configs: {sum(results['configs'].values())}/{len(results['configs'])} berhasil disalin" if 'configs' in results else "",
                f"🔗 Symlinks: {sum(results['symlinks'].values())}/{len(results['symlinks'])} berhasil dibuat" if 'symlinks' in results else ""
            ]
            
            # Log non-empty summaries
            [self.logger.info(summary) for summary in result_summaries if summary]
            
            # Log overall success dengan one-liner
            if results.get('success'):
                self.logger.success("🎯 Setup components berhasil dikonfigurasi")
            else:
                self.logger.warning("⚠️ Setup selesai dengan beberapa issue minor")
        except Exception:
            pass
    
    def _initialize_managers(self):
        """Initialize environment dan config managers dengan minimal logging"""
        try:
            # Initialize config manager dengan one-liner
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            self.ui_components['config_manager'] = config_manager
            
            if self.logger:
                self.logger.info("⚙️ Config manager ready")
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Manager init warning: {str(e)}")
                self.logger.info("💡 Setup akan tetap dilanjutkan...")
    
    def _log_environment_details(self):
        """Log environment details dengan format minimal menggunakan utils"""
        try:
            if not self.logger or not self.env_manager:
                return
            
            # Get minimal summary dengan utils
            env_summary = get_system_summary_minimal(self.env_manager)
            if env_summary:
                self.logger.info(f"🌍 {env_summary}")
            
        except Exception:
            # Silent jika error untuk mencegah log noise
            pass