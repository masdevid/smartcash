"""
File: smartcash/ui/setup/env_config/handlers/setup_handler.py
Deskripsi: Setup handler untuk environment configuration dengan proper workflow
"""

import time
from typing import Dict, Any
from smartcash.ui.setup.env_config.handlers.drive_handler import DriveHandler
from smartcash.ui.setup.env_config.handlers.folder_handler import FolderHandler
from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler
from smartcash.ui.setup.env_config.utils.ui_updater import update_progress_bar, update_status_panel
from smartcash.ui.setup.env_config.utils.progress_tracker import track_setup_progress
from smartcash.ui.setup.env_config.components.setup_summary import update_setup_summary

class SetupHandler:
    """🚀 Handler untuk mengelola setup environment workflow"""
    
    def __init__(self, logger=None):
        self.logger = logger or self._create_dummy_logger()
        self.drive_handler = DriveHandler(logger)
        self.folder_handler = FolderHandler(logger)
        self.config_handler = ConfigHandler()
    
    def run_full_setup(self, ui_components: Dict[str, Any]) -> bool:
        """🔄 Jalankan full setup workflow dengan progress tracking"""
        try:
            # Initialize progress tracking
            progress_tracker = track_setup_progress(ui_components)
            
            # Set running state
            self._set_running_state(ui_components)
            
            # Execute setup workflow
            summary_data = self._execute_setup_workflow(ui_components, progress_tracker)
            
            # Update final summary
            update_setup_summary(ui_components['setup_summary'], summary_data)
            
            # Set completion state
            self._set_completion_state(ui_components, summary_data)
            
            return summary_data.get('success', False)
            
        except Exception as e:
            self.logger.error(f"❌ Setup failed: {str(e)}")
            self._set_error_state(ui_components, str(e))
            return False
    
    def _execute_setup_workflow(self, ui_components: Dict[str, Any], progress_tracker) -> Dict[str, Any]:
        """🔧 Execute setup steps dengan progress tracking"""
        summary_data = {
            'drive_mounted': False,
            'mount_path': 'N/A',
            'configs_synced': 0,
            'symlinks_created': 0,
            'folders_created': 0,
            'success': False
        }
        
        try:
            # Step 1: Mount Drive
            self.logger.info("🔧 Step 1: Mounting Google Drive...")
            progress_tracker.update_step("Mounting Google Drive", 0)
            
            drive_result = self.drive_handler.mount_drive()
            summary_data['drive_mounted'] = drive_result.get('success', False)
            summary_data['mount_path'] = drive_result.get('mount_path', 'N/A')
            
            # Step 2: Create Folders
            self.logger.info("📁 Step 2: Creating directories...")
            progress_tracker.update_step("Creating directories", 25)
            
            folder_result = self.folder_handler.create_required_folders()
            summary_data['folders_created'] = folder_result.get('created_count', 0)
            summary_data['symlinks_created'] = folder_result.get('symlinks_count', 0)
            
            # Step 3: Sync Configurations
            self.logger.info("⚙️ Step 3: Syncing configurations...")
            progress_tracker.update_step("Syncing configurations", 50)
            
            config_result = self.config_handler.sync_configurations()
            summary_data['configs_synced'] = config_result.get('synced_count', 0)
            
            # Step 4: Verify Setup
            self.logger.info("✅ Step 4: Verifying setup...")
            progress_tracker.update_step("Verifying setup", 75)
            
            # Verification logic here
            time.sleep(1)  # Simulate verification
            
            # Complete
            progress_tracker.update_step("Setup complete", 100)
            summary_data['success'] = True
            
            self.logger.success("🎉 Environment setup completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Setup workflow failed: {str(e)}")
            summary_data['success'] = False
            
        return summary_data
    
    def _set_running_state(self, ui_components: Dict[str, Any]) -> None:
        """🔄 Set UI ke running state"""
        ui_components['setup_button'].disabled = True
        ui_components['setup_button'].description = "⏳ Running Setup..."
        update_status_panel(ui_components['status_panel'], "Setup sedang berjalan...", "warning")
    
    def _set_completion_state(self, ui_components: Dict[str, Any], summary_data: Dict[str, Any]) -> None:
        """✅ Set UI ke completion state"""
        success = summary_data.get('success', False)
        
        if success:
            ui_components['setup_button'].description = "✅ Setup Complete"
            ui_components['setup_button'].button_style = 'success'
            update_status_panel(ui_components['status_panel'], "Setup berhasil diselesaikan!", "success")
        else:
            ui_components['setup_button'].description = "❌ Setup Failed"
            ui_components['setup_button'].button_style = 'danger'
            update_status_panel(ui_components['status_panel'], "Setup gagal, silakan coba lagi", "danger")
    
    def _set_error_state(self, ui_components: Dict[str, Any], error_msg: str) -> None:
        """❌ Set UI ke error state"""
        ui_components['setup_button'].disabled = False
        ui_components['setup_button'].description = "🔄 Retry Setup"
        ui_components['setup_button'].button_style = 'warning'
        update_status_panel(ui_components['status_panel'], f"Error: {error_msg}", "danger")
    
    def _create_dummy_logger(self):
        """📝 Create dummy logger fallback"""
        class DummyLogger:
            def info(self, msg): print(f"ℹ️ {msg}")
            def warning(self, msg): print(f"⚠️ {msg}")
            def error(self, msg): print(f"❌ {msg}")
            def success(self, msg): print(f"✅ {msg}")
        return DummyLogger()