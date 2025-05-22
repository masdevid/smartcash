"""
File: smartcash/ui/setup/env_config/handlers/environment_config_orchestrator.py
Deskripsi: Orchestrator yang mengoordinasikan setup environment dengan logging minimal dan flow yang jelas
"""

from typing import Dict, Any, Tuple

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.setup.env_config.handlers.environment_status_checker import EnvironmentStatusChecker
from smartcash.ui.setup.env_config.handlers.drive_setup_handler import DriveSetupHandler

class EnvironmentConfigOrchestrator:
    """Orchestrator untuk mengoordinasikan setup environment dengan flow yang jelas"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi orchestrator"""
        self.ui_components = ui_components
        self.logger = create_ui_logger_bridge(ui_components, "env_orchestrator")
        self.status_checker = EnvironmentStatusChecker()
        self.drive_handler = DriveSetupHandler(ui_components)
    
    def check_environment_status(self) -> Dict[str, Any]:
        """Check status environment tanpa logging berlebihan"""
        return self.status_checker.get_comprehensive_status()
    
    def perform_environment_setup(self) -> bool:
        """Lakukan setup environment dengan flow yang jelas"""
        self.logger.info("🚀 Memulai setup environment...")
        self._update_progress(0.1, "Memulai setup...")
        
        try:
            # Step 1: Check current status
            self._update_progress(0.15, "🔍 Checking status...")
            status = self.status_checker.get_comprehensive_status()
            
            if status['ready']:
                self.logger.success("✅ Environment sudah siap!")
                self._hide_progress()
                return True
            
            # Step 2: Ensure Drive mounted
            self._update_progress(0.2, "📱 Menghubungkan Drive...")
            if status['drive']['type'] == 'colab':
                success, message = self.drive_handler.ensure_drive_mounted()
                if not success:
                    self.logger.error(f"❌ {message}")
                    self._reset_progress("Setup gagal")
                    return False
            
            # Step 3: Get Drive path
            drive_path = status['drive']['path']
            if not drive_path:
                self.logger.error("❌ Drive path tidak tersedia")
                self._reset_progress("Setup gagal")
                return False
            
            # Step 4: Perform complete setup
            self.logger.info("🔧 Melakukan setup lengkap...")
            setup_results = self.drive_handler.perform_complete_setup(drive_path)
            
            # Step 5: Initialize managers
            self._update_progress(0.8, "🔧 Inisialisasi managers...")
            self._initialize_managers()
            
            # Step 6: Final verification
            self._update_progress(0.9, "✅ Verifikasi final...")
            final_status = self.status_checker.get_comprehensive_status()
            
            if final_status['ready'] or setup_results['success']:
                self.logger.success("🎉 Setup environment berhasil!")
                self._update_progress(1.0, "Setup selesai")
                self._hide_progress()
                return True
            else:
                self.logger.warning("⚠️ Setup selesai dengan beberapa masalah")
                self._update_progress(1.0, "Setup selesai dengan warning")
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Error setup: {str(e)}")
            self._reset_progress("Setup gagal")
            return False
    
    def _initialize_managers(self):
        """Initialize environment dan config managers"""
        try:
            # Initialize environment manager
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            self.ui_components['env_manager'] = env_manager
            
            # Initialize config manager
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            self.ui_components['config_manager'] = config_manager
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal inisialisasi managers: {str(e)}")
    
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