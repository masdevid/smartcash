"""
File: smartcash/ui/setup/env_config/handlers/setup_handler.py
Deskripsi: Fixed setup handler dengan proper imports dan UI state management
"""

import time
from typing import Dict, Any
from smartcash.ui.setup.env_config.handlers.environment_handler import EnvironmentHandler
from smartcash.ui.setup.env_config.handlers.drive_handler import DriveHandler
from smartcash.ui.setup.env_config.handlers.folder_handler import FolderHandler
from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler
from smartcash.ui.setup.env_config.constants import PROGRESS_STEPS, STATUS_MESSAGES
from smartcash.ui.setup.env_config.utils.ui_updater import update_progress_bar, update_status_panel


class SetupHandler:
    """🚀 Setup handler dengan comprehensive workflow tanpa circular import"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.env_handler = EnvironmentHandler(logger)
        self.drive_handler = DriveHandler(logger)
        self.folder_handler = FolderHandler()  # Doesn't accept logger parameter
        self.config_handler = ConfigHandler()
    
    def run_full_setup(self, ui_components: Dict[str, Any]) -> bool:
        """🔄 Run full environment setup dengan progress tracking"""
        try:
            # Set initial state
            self._set_setup_running_state(ui_components)
            
            # Execute setup steps
            success = self._execute_setup_workflow(ui_components)
            
            if success:
                self._set_setup_complete_state(ui_components)
                return True
            else:
                self._set_setup_failed_state(ui_components)
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Setup failed: {str(e)}")
            self._set_setup_failed_state(ui_components, str(e))
            return False
    
    def _execute_setup_workflow(self, ui_components: Dict[str, Any]) -> bool:
        """Execute complete setup workflow dengan progress updates"""
        steps = [
            ('analysis', self._analyze_environment),
            ('drive_mount', self._ensure_drive_connection),
            ('folders', self._create_folder_structure),
            ('configs', self._copy_config_templates),
            ('symlinks', self._create_symlinks),
            ('validation', self._validate_setup)
        ]
        
        for step_name, step_func in steps:
            self._update_step_progress(ui_components, step_name, 'start')
            
            if not step_func():
                self.logger.error(f"❌ Setup failed at step: {step_name}")
                return False
            
            self._update_step_progress(ui_components, step_name, 'complete')
            time.sleep(0.5)  # Brief pause for UX
        
        return True
    
    def _analyze_environment(self) -> bool:
        """Analyze current environment"""
        try:
            self.logger.info("🔍 Menganalisis environment saat ini...")
            env_status = self.env_handler.get_environment_status()
            return env_status.get('ready', False)
        except Exception as e:
            self.logger.error(f"❌ Environment analysis failed: {str(e)}")
            return False
    
    def _ensure_drive_connection(self) -> bool:
        """Ensure Google Drive is connected"""
        try:
            mount_result = self.drive_handler.mount_drive()
            return mount_result.get('success', False)
        except Exception as e:
            self.logger.error(f"❌ Drive connection failed: {str(e)}")
            return False
    
    def _create_folder_structure(self) -> bool:
        """Create required folder structure"""
        try:
            self.folder_handler.create_folder_structures(self.logger)
            return True
        except Exception as e:
            self.logger.error(f"❌ Folder creation failed: {str(e)}")
            return False
    
    def _copy_config_templates(self) -> bool:
        """Copy configuration templates"""
        try:
            self.config_handler.setup_configurations(self.logger)
            return True
        except Exception as e:
            self.logger.error(f"❌ Config copy failed: {str(e)}")
            return False
    
    def _create_symlinks(self) -> bool:
        """Create symbolic links"""
        try:
            return self.env_handler.create_project_symlinks(self.logger)
        except Exception as e:
            self.logger.error(f"❌ Symlink creation failed: {str(e)}")
            return False
    
    def _validate_setup(self) -> bool:
        """Validate complete setup"""
        try:
            validation = self.env_handler.validate_post_setup()
            return validation.get('valid', False)
        except Exception as e:
            self.logger.error(f"❌ Setup validation failed: {str(e)}")
            return False
    
    def _set_setup_running_state(self, ui_components: Dict[str, Any]) -> None:
        """Set UI to setup running state"""
        update_status_panel(ui_components, STATUS_MESSAGES['setup_running'], 'info')
        update_progress_bar(ui_components, 0, "Memulai setup...")
        
        if 'setup_button' in ui_components:
            ui_components['setup_button'].disabled = True
            ui_components['setup_button'].description = "Setting up..."
    
    def _set_setup_complete_state(self, ui_components: Dict[str, Any]) -> None:
        """Set UI to setup complete state"""
        update_status_panel(ui_components, STATUS_MESSAGES['setup_success'], 'success')
        update_progress_bar(ui_components, 100, "✅ Setup lengkap!")
        
        if 'setup_button' in ui_components:
            ui_components['setup_button'].disabled = False
            ui_components['setup_button'].description = "Setup Environment"
            ui_components['setup_button'].button_style = 'success'
    
    def _set_setup_failed_state(self, ui_components: Dict[str, Any], error_msg: str = "") -> None:
        """Set UI to setup failed state"""
        msg = f"{STATUS_MESSAGES['setup_failed']}: {error_msg}" if error_msg else STATUS_MESSAGES['setup_failed']
        update_status_panel(ui_components, msg, 'error')
        update_progress_bar(ui_components, 0, "❌ Setup gagal", is_error=True)
        
        if 'setup_button' in ui_components:
            ui_components['setup_button'].disabled = False  
            ui_components['setup_button'].description = "Retry Setup"
            ui_components['setup_button'].button_style = 'danger'
    
    def _update_step_progress(self, ui_components: Dict[str, Any], step_name: str, phase: str) -> None:
        """Update progress untuk specific step"""
        if step_name not in PROGRESS_STEPS:
            return
        
        step_info = PROGRESS_STEPS[step_name]
        
        if phase == 'start':
            progress = step_info['range'][0]
            message = step_info['label']
        else:  # complete
            progress = step_info['range'][1]
            message = f"✅ {step_info['label']} selesai"
        
        update_progress_bar(ui_components, progress, message)
        
        if self.logger:
            self.logger.info(f"📊 {message} ({progress}%)")