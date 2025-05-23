"""
File: smartcash/ui/setup/env_config/handlers/environment_config_orchestrator.py
Deskripsi: Orchestrator yang mengoordinasikan setup environment dengan logging informatif dan integrasi environment manager
"""

from typing import Dict, Any, Tuple

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.setup.env_config.handlers.environment_status_checker import EnvironmentStatusChecker
from smartcash.ui.setup.env_config.handlers.drive_setup_handler import DriveSetupHandler

class EnvironmentConfigOrchestrator:
    """Orchestrator untuk mengoordinasikan setup environment dengan logging informatif"""
    
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
            from smartcash.common.environment import get_environment_manager
            self.env_manager = get_environment_manager()
            self.ui_components['env_manager'] = self.env_manager
        except Exception:
            # Silent failure, akan di-handle saat logger ready
            pass
    
    def init_logger(self):
        """Initialize logger setelah UI ready"""
        if self.logger is None:
            self.logger = create_ui_logger_bridge(self.ui_components, "env_config")
            
            # Log environment details setelah logger ready
            if hasattr(self, 'env_manager'):
                self._log_environment_details()
    
    def _log_environment_details(self):
        """Log detail environment dari environment manager"""
        try:
            system_info = self.env_manager.get_system_info()
            
            # Log environment details dengan format yang rapi
            self.logger.info(f"🌍 Platform: {system_info.get('environment', 'Unknown')}")
            self.logger.info(f"📁 Base Directory: {system_info.get('base_directory', 'N/A')}")
            
            if self.env_manager.is_colab:
                drive_status = "✅ Terhubung" if self.env_manager.is_drive_mounted else "❌ Belum Terhubung"
                self.logger.info(f"💾 Google Drive: {drive_status}")
                
                if self.env_manager.drive_path:
                    self.logger.info(f"🎯 Drive SmartCash Path: {self.env_manager.drive_path}")
            
            # Log GPU info
            if system_info.get('cuda_available'):
                gpu_info = f"Device: {system_info.get('cuda_device_name', 'Unknown')[:40]}"
                self.logger.info(f"🎮 GPU Available: {gpu_info}")
            else:
                self.logger.info("🎮 GPU: CPU-only mode")
            
            # Log memory info jika tersedia
            if 'available_memory_gb' in system_info:
                memory_gb = system_info['available_memory_gb']
                memory_status = "🟢" if memory_gb > 8 else "🟡" if memory_gb > 4 else "🔴"
                self.logger.info(f"🧠 Memory: {memory_status} {memory_gb:.1f}GB available")
            
        except Exception as e:
            self.logger.debug(f"🔍 Detail logging error: {str(e)}")
    
    def check_environment_status(self) -> Dict[str, Any]:
        """Check status environment dengan Drive state refresh dan retry mechanism"""
        # Ensure logger is initialized
        if self.logger is None:
            self.init_logger()
        
        # Refresh Drive state sebelum check jika ada environment manager
        if hasattr(self, 'env_manager') and self.env_manager.is_colab:
            try:
                # Force refresh Drive status
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
            if missing_count > 0:
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
            # Step 1: Pre-setup Drive state refresh
            self._update_progress(0.15, "🔄 Refreshing environment state...")
            if hasattr(self, 'env_manager') and self.env_manager.is_colab:
                self.env_manager.refresh_drive_status()
            
            # Step 2: Check current status dengan refreshed state
            self._update_progress(0.2, "🔍 Analyzing environment...")
            status = self.status_checker.get_comprehensive_status()
            
            # Log current status dengan detail
            self._log_current_status(status)
            
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
                    
                    # Critical: Refresh state setelah mount dengan delay
                    import time
                    time.sleep(2)  # Wait untuk mount completion
                    
                    if hasattr(self, 'env_manager'):
                        self.env_manager.refresh_drive_status()
                        self.logger.info("🔄 Drive state refreshed setelah mount")
                    
                    # Re-check status setelah mount
                    status = self.status_checker.get_comprehensive_status()
                    
                else:
                    self.logger.error(f"❌ Drive Error: {message}")
                    self._reset_progress("Drive connection failed")
                    return False
            
            # Step 4: Validate Drive path setelah refresh
            drive_path = status['drive']['path']
            if not drive_path:
                # Retry get path setelah refresh
                if hasattr(self, 'env_manager') and self.env_manager.drive_path:
                    drive_path = str(self.env_manager.drive_path)
                else:
                    self.logger.error("❌ Drive path tidak dapat diakses setelah refresh")
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
            
            # Step 8: Final verification dengan comprehensive refresh
            self._update_progress(0.9, "✅ Verifikasi final...")
            if hasattr(self, 'env_manager'):
                self.env_manager.refresh_drive_status()
                import time
                time.sleep(1)  # Final settling time
            
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
            self.logger.error(f"❌ Error setup environment: {str(e)}")
            self._reset_progress("Setup gagal")
            return False
    
    def _log_current_status(self, status: Dict[str, Any]):
        """Log current status dengan format yang informatif"""
        # Drive status
        drive_info = status.get('drive', {})
        if drive_info.get('type') == 'colab':
            mount_status = "✅ Mounted" if drive_info.get('mounted') else "❌ Not Mounted"
            self.logger.info(f"💾 Google Drive: {mount_status}")
        
        # Missing items summary
        missing_folders = len(status.get('missing_drive_folders', []))
        missing_configs = len(status.get('missing_drive_configs', []))
        invalid_symlinks = len(status.get('invalid_symlinks', []))
        
        if missing_folders > 0:
            self.logger.info(f"📁 Missing folders: {missing_folders} items")
        if missing_configs > 0:
            self.logger.info(f"📋 Missing configs: {missing_configs} files")
        if invalid_symlinks > 0:
            self.logger.info(f"🔗 Invalid symlinks: {invalid_symlinks} links")
    
    def _log_setup_results(self, results: Dict[str, Any]):
        """Log hasil setup dengan detail yang informatif"""
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
    
    def _initialize_managers(self):
        """Initialize environment dan config managers dengan error handling"""
        try:
            # Initialize environment manager (refresh jika sudah ada)
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            env_manager.refresh_drive_status()  # Refresh status setelah setup
            self.ui_components['env_manager'] = env_manager
            self.logger.info("🌍 Environment manager terinisialisasi")
            
            # Initialize config manager
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            self.ui_components['config_manager'] = config_manager
            self.logger.info("⚙️ Config manager terinisialisasi")
            
            # Log updated system info
            system_info = env_manager.get_system_info()
            if env_manager.is_drive_mounted:
                self.logger.success(f"💾 Drive terhubung: {system_info.get('drive_path', 'N/A')}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Manager initialization warning: {str(e)}")
            self.logger.info("💡 Setup akan tetap dilanjutkan...")
    
    def _update_progress(self, value: float, message: str = ""):
        """Update progress dengan error handling"""
        if 'progress_bar' in self.ui_components:
            try:
                from smartcash.ui.setup.env_config.components.progress_tracking import update_progress
                update_progress(self.ui_components, int(value * 100), 100, message)
            except ImportError:
                pass
    
    def _reset_progress(self, message: str = ""):
        """Reset progress ke 0"""
        if 'progress_bar' in self.ui_components:
            try:
                from smartcash.ui.setup.env_config.components.progress_tracking import reset_progress
                reset_progress(self.ui_components, message)
            except ImportError:
                pass
    
    def _hide_progress(self):
        """Sembunyikan progress bar setelah setup berhasil"""
        if 'progress_container' in self.ui_components:
            self.ui_components['progress_container'].layout.visibility = 'hidden'