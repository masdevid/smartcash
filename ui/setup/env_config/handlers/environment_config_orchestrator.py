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
            self.logger = create_ui_logger_bridge(self.ui_components, "env_config")
            self._log_environment_details()
    
    def check_environment_status(self) -> Dict[str, Any]:
        """Check status environment dengan silent Drive refresh dan minimal logging"""
        if self.logger is None:
            self.init_logger()
        
        # Silent refresh Drive state
        refresh_environment_state_silent(self.env_manager)
        
        # Get status dengan refreshed state
        status = self.status_checker.get_comprehensive_status()
        
        # Log hanya status penting tanpa detail berlebihan
        if not status.get('ready', False) and self.logger:
            missing_count = (len(status.get('missing_drive_folders', [])) +
                           len(status.get('missing_drive_configs', [])) +
                           len(status.get('invalid_symlinks', [])))
            if missing_count > 0:
                self.logger.info(f"🔧 Setup diperlukan: {missing_count} item perlu dikonfigurasi")
        
        return status
    
    def perform_environment_setup(self) -> bool:
        """Environment setup dengan complete progress tracking dan error handling"""
        if self.logger is None:
            self.init_logger()
            
        self.logger.info(get_status_message('setup_start'))
        update_progress_safe(self.ui_components, 1, get_progress_message('start'))
        
        try:
            # Step 1: Pre-setup refresh (silent)
            update_progress_safe(self.ui_components, 2, get_progress_message('refresh'))
            refresh_environment_state_silent(self.env_manager)
            
            # Step 2: Check current status
            update_progress_safe(self.ui_components, 5, get_progress_message('analysis'))
            status = self.status_checker.get_comprehensive_status()
            self._log_current_status_minimal(status)
            
            if status['ready']:
                self.logger.success("✅ Environment sudah siap digunakan!")
                update_progress_safe(self.ui_components, 100, "✅ Environment ready")
                hide_progress_safe(self.ui_components)
                return True
            
            # Step 3: Ensure Drive mounted
            update_progress_safe(self.ui_components, 10, get_progress_message('drive_connect'))
            if status['drive']['type'] == 'colab':
                success, message = self.drive_handler.ensure_drive_mounted()
                if success:
                    self.logger.success(f"📱 Drive: {message}")
                    
                    # Refresh state setelah mount dengan proper delay
                    import time
                    time.sleep(2)
                    refresh_environment_state_silent(self.env_manager)
                    self.logger.info("🔄 Drive state refreshed")
                    
                    # Re-check status setelah mount
                    status = self.status_checker.get_comprehensive_status()
                    
                else:
                    self.logger.error(f"❌ Drive Error: {message}")
                    reset_progress_safe(self.ui_components, "Drive connection failed")
                    return False
            
            # Step 4: Validate Drive path
            drive_path = status['drive']['path']
            if not drive_path:
                # Retry get path setelah refresh
                if self.env_manager:
                    try:
                        drive_path = self.env_manager.get_drive_path()
                    except Exception:
                        pass
                
                if not drive_path:
                    self.logger.error(get_status_message('drive_error'))
                    reset_progress_safe(self.ui_components, "Setup gagal")
                    return False
            
            self.logger.info(f"🎯 Target setup path: {drive_path}")
            
            # Step 5: Perform complete setup (progress 30-100% handled by drive_handler)
            self.logger.info("🔧 Melakukan setup lengkap...")
            setup_results = self.drive_handler.perform_complete_setup(drive_path)
            
            # Step 6: Log setup results dengan detail
            self._log_setup_results(setup_results)
            
            # Step 7: Initialize managers
            self._initialize_managers()
            
            # Step 8: Final verification dengan comprehensive refresh
            refresh_environment_state_silent(self.env_manager)
            final_status = self.status_checker.get_comprehensive_status()
            
            if final_status['ready'] or setup_results['success']:
                self.logger.success(get_status_message('setup_success'))
                self.logger.info("🔗 Symlinks aktif, data akan tersimpan di Drive")
                hide_progress_safe(self.ui_components)
                return True
            else:
                self.logger.warning("⚠️ Setup selesai dengan beberapa komponen belum optimal")
                self.logger.info("💡 Environment tetap bisa digunakan untuk development")
                return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error setup environment: {str(e)}")
            reset_progress_safe(self.ui_components, "Setup gagal")
            return False
    
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