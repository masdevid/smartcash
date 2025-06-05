"""
File: smartcash/ui/setup/env_config/handlers/environment_config_orchestrator.py
Deskripsi: Orchestrator yang mengoordinasikan setup environment dengan logging minimal dan state management
"""

from typing import Dict, Any, Tuple

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.setup.env_config.handlers.environment_status_checker import EnvironmentStatusChecker
from smartcash.ui.setup.env_config.handlers.drive_setup_handler import DriveSetupHandler

class EnvironmentConfigOrchestrator:
    """Orchestrator untuk mengoordinasikan setup environment dengan logging minimal"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi orchestrator dengan silent environment manager integration"""
        self.ui_components = ui_components
        # Logger akan diinit setelah UI ready
        self.logger = None
        self.status_checker = EnvironmentStatusChecker()
        self.drive_handler = DriveSetupHandler(ui_components)
        
        # Initialize environment manager secara silent
        self._init_environment_manager_silent()
    
    def _init_environment_manager_silent(self):
        """Initialize environment manager tanpa logging untuk menghindari premature output"""
        try:
            from smartcash.ui.setup.env_config.helpers.silent_environment_manager import get_silent_environment_manager
            self.env_manager = get_silent_environment_manager()
            self.ui_components['env_manager'] = self.env_manager
        except Exception:
            # Silent failure, akan di-handle saat logger ready
            self.env_manager = None
    
    def init_logger(self):
        """Initialize logger setelah UI ready"""
        if self.logger is None:
            self.logger = create_ui_logger_bridge(self.ui_components, "env_config")
            
            # Log environment details setelah logger ready
            if hasattr(self, 'env_manager') and self.env_manager:
                self._log_environment_details()
    
    def check_environment_status(self) -> Dict[str, Any]:
        """Check status environment dengan silent Drive state refresh"""
        # Ensure logger is initialized
        if self.logger is None:
            self.init_logger()
        
        # Refresh Drive state secara silent
        if hasattr(self, 'env_manager') and self.env_manager:
            try:
                # Use silent refresh to prevent log leakage
                self.env_manager.refresh_drive_status()
                
                # Small delay untuk memastikan state terupdate
                import time
                time.sleep(0.5)
            except Exception:
                pass
        
        # Get status dengan refreshed state
        status = self.status_checker.get_comprehensive_status()
        
        # Log hanya status penting tanpa detail berlebihan
        if not status.get('ready', False):
            missing_count = (
                len(status.get('missing_drive_folders', [])) +
                len(status.get('missing_drive_configs', [])) +
                len(status.get('invalid_symlinks', []))
            )
            if missing_count > 0 and self.logger:
                self.logger.info(f"🔧 Setup diperlukan: {missing_count} item perlu dikonfigurasi")
        
        return status
    
    def perform_environment_setup(self) -> bool:
        """Lakukan setup environment dengan Drive state management yang proper"""
        # Ensure logger is initialized
        if self.logger is None:
            self.init_logger()
            
        self.logger.info("🚀 Memulai konfigurasi environment SmartCash...")
        self._update_progress(0.1, "Memulai setup...")
        
        try:
            # Step 1: Pre-setup Drive state refresh (silent)
            self._update_progress(0.15, "🔄 Refreshing environment state...")
            if hasattr(self, 'env_manager') and self.env_manager:
                try:
                    # Silent refresh to prevent log leakage
                    self.env_manager.refresh_drive_status()
                except Exception:
                    pass
            
            # Step 2: Check current status dengan refreshed state
            self._update_progress(0.2, "🔍 Analyzing environment...")
            status = self.status_checker.get_comprehensive_status()
            
            # Log current status dengan minimal detail
            self._log_current_status_minimal(status)
            
            if status['ready']:
                self.logger.success("✅ Environment sudah siap digunakan!")
                self._hide_progress()
                return True
            
            # Step 3: Ensure Drive mounted dengan state management
            self._update_progress(0.3, "📱 Menghubungkan Google Drive...")
            if status['drive']['type'] == 'colab':
                success, message = self.drive_handler.ensure_drive_mounted()
                if success:
                    self.logger.success(f"📱 Drive: {message}")
                    
                    # Critical: Refresh state setelah mount dengan delay (silent)
                    import time
                    time.sleep(2)  # Wait untuk mount completion
                    
                    if hasattr(self, 'env_manager') and self.env_manager:
                        try:
                            self.env_manager.refresh_drive_status()
                            self.logger.info("🔄 Drive state refreshed")
                        except Exception:
                            pass
                    
                    # Re-check status setelah mount
                    status = self.status_checker.get_comprehensive_status()
                    
                else:
                    self.logger.error(f"❌ Drive Error: {message}")
                    self._reset_progress("Drive connection failed")
                    return False
            
            # Step 4: Validate Drive path setelah refresh
            drive_path = status['drive']['path']
            if not drive_path:
                # Retry get path setelah refresh (silent)
                if hasattr(self, 'env_manager') and self.env_manager:
                    try:
                        drive_path = self.env_manager.get_drive_path()
                    except Exception:
                        pass
                
                if not drive_path:
                    self.logger.error("❌ Drive path tidak dapat diakses")
                    self._reset_progress("Setup gagal")
                    return False
            
            self.logger.info(f"🎯 Target setup path: {drive_path}")
            
            # Step 5: Perform complete setup
            self.logger.info("🔧 Melakukan setup lengkap...")
            setup_results = self.drive_handler.perform_complete_setup(drive_path)
            
            # Step 6: Log setup results dengan detail
            self._log_setup_results(setup_results)
            
            # Step 7: Initialize managers dengan state refresh
            self._update_progress(0.8, "🔧 Inisialisasi managers...")
            self._initialize_managers()
            
            # Step 8: Final verification dengan comprehensive refresh (silent)
            self._update_progress(0.9, "✅ Verifikasi final...")
            if hasattr(self, 'env_manager') and self.env_manager:
                try:
                    self.env_manager.refresh_drive_status()
                    import time
                    time.sleep(1)  # Final settling time
                except Exception:
                    pass
            
            final_status = self.status_checker.get_comprehensive_status()
            
            if final_status['ready'] or setup_results['success']:
                self.logger.success("🎉 Setup environment berhasil selesai!")
                self.logger.info("🔗 Symlinks aktif, data akan tersimpan di Drive")
                self._update_progress(1.0, "Setup selesai")
                self._hide_progress()
                return True
            else:
                self.logger.warning("⚠️ Setup selesai dengan beberapa komponen belum optimal")
                self.logger.info("💡 Environment tetap bisa digunakan untuk development")
                self._update_progress(1.0, "Setup selesai dengan warning")
                return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error setup environment: {str(e)}")
            self._reset_progress("Setup gagal")
            return False
    
    def _log_current_status_minimal(self, status: Dict[str, Any]):
        """Log current status dengan format minimal untuk mengurangi noise"""
        try:
            if not self.logger:
                return
                
            # Hanya log info essential
            drive_info = status.get('drive', {})
            if drive_info.get('type') == 'colab':
                mount_status = "✅" if drive_info.get('mounted') else "❌"
                self.logger.info(f"💾 Drive: {mount_status}")
            
            # Summary missing items tanpa detail
            missing_total = (
                len(status.get('missing_drive_folders', [])) +
                len(status.get('missing_drive_configs', [])) +
                len(status.get('invalid_symlinks', []))
            )
            
            if missing_total > 0:
                self.logger.info(f"🔧 Missing items: {missing_total}")
        except Exception:
            pass
    
    def _log_setup_results(self, results: Dict[str, Any]):
        """Log hasil setup dengan detail yang informatif"""
        try:
            if not self.logger:
                return
                
            if 'folders' in results:
                success_folders = sum(results['folders'].values())
                total_folders = len(results['folders'])
                self.logger.info(f"📁 Folders: {success_folders}/{total_folders} berhasil dibuat")
            
            if 'configs' in results:
                success_configs = sum(results['configs'].values())
                total_configs = len(results['configs'])
                self.logger.info(f"📋 Configs: {success_configs}/{total_configs} berhasil disalin")
            
            if 'symlinks' in results:
                success_symlinks = sum(results['symlinks'].values())
                total_symlinks = len(results['symlinks'])
                self.logger.info(f"🔗 Symlinks: {success_symlinks}/{total_symlinks} berhasil dibuat")
            
            # Log overall success
            if results.get('success'):
                self.logger.success("🎯 Setup components berhasil dikonfigurasi")
            else:
                self.logger.warning("⚠️ Setup selesai dengan beberapa issue minor")
        except Exception:
            pass
    
    def _initialize_managers(self):
        """Initialize environment dan config managers dengan minimal logging"""
        try:
            # Initialize config manager (tidak perlu environment manager karena sudah ada silent version)
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
        """Log environment details dengan format minimal"""
        try:
            if not self.logger or not hasattr(self, 'env_manager') or not self.env_manager:
                return
                
            system_info = self.env_manager.get_system_info()
            
            # Single line summary untuk mengurangi log noise
            env_summary = []
            env_summary.append(system_info.get('environment', 'Unknown'))
            
            if system_info.get('cuda_available'):
                env_summary.append("GPU✅")
            else:
                env_summary.append("CPU")
            
            if 'available_memory_gb' in system_info:
                memory_gb = system_info['available_memory_gb']
                env_summary.append(f"{memory_gb:.1f}GB")
            
            self.logger.info(f"🌍 {' | '.join(env_summary)}")
            
        except Exception:
            # Silent jika error untuk mencegah log noise
            pass
    
    def _update_progress(self, value: float, message: str = ""):
        """Update progress dengan error handling"""
        try:
            # Cek apakah progress_tracker tersedia
            if 'progress_tracker' in self.ui_components and self.ui_components['progress_tracker']:
                # Gunakan progress tracker baru
                progress_tracker = self.ui_components['progress_tracker']
                progress_tracker.update('level1', int(value * 100), message)
            # Fallback ke progress bar lama
            elif 'progress_bar' in self.ui_components:
                from smartcash.ui.components.progress_tracker import update_progress
                progress_bar = self.ui_components.get('progress_bar')
                if progress_bar:
                    progress_bar.value = int(value * 100)
                    progress_bar.description = f"{int(value * 100)}%"
                    
                if 'progress_message' in self.ui_components and message:
                    self.ui_components['progress_message'].value = message
        except Exception as e:
            # Silent failure untuk mencegah error pada proses utama
            pass
    
    def _reset_progress(self, message: str = ""):
        """Reset progress ke 0"""
        try:
            # Cek apakah progress_tracker tersedia
            if 'progress_tracker' in self.ui_components and self.ui_components['progress_tracker']:
                # Gunakan progress tracker baru
                progress_tracker = self.ui_components['progress_tracker']
                progress_tracker.reset()
            # Fallback ke progress bar lama
            elif 'progress_bar' in self.ui_components:
                from smartcash.ui.components.progress_tracker import reset_progress
                progress_bar = self.ui_components.get('progress_bar')
                if progress_bar:
                    progress_bar.value = 0
                    progress_bar.description = "0%"
                
                if 'progress_message' in self.ui_components:
                    self.ui_components['progress_message'].value = message or ""
                
                # Show progress container pada reset
                if 'progress_container' in self.ui_components:
                    self.ui_components['progress_container'].layout.visibility = 'visible'
        except Exception as e:
            # Silent failure untuk mencegah error pada proses utama
            pass
    
    def _hide_progress(self):
        """Sembunyikan progress bar setelah setup berhasil"""
        try:
            # Cek apakah progress_tracker tersedia
            if 'progress_tracker' in self.ui_components and self.ui_components['progress_tracker']:
                # Gunakan progress tracker baru - reset akan menyembunyikan
                progress_tracker = self.ui_components['progress_tracker']
                progress_tracker.reset()
            # Fallback ke progress bar lama
            elif 'progress_container' in self.ui_components:
                self.ui_components['progress_container'].layout.visibility = 'hidden'
        except Exception as e:
            # Silent failure untuk mencegah error pada proses utama
            pass