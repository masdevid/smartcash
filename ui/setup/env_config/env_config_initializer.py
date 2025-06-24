"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Main initializer untuk environment configuration UI
"""

from typing import Dict, Any, Optional
from smartcash.ui.setup.env_config.components.env_config_component import create_env_config_component
from smartcash.ui.setup.env_config.handlers.status_handler import StatusHandler
from smartcash.ui.setup.env_config.handlers.drive_handler import DriveHandler
from smartcash.ui.setup.env_config.utils.progress_tracker import ProgressTracker
from smartcash.ui.setup.env_config.utils.ui_updater import (
    update_status_panel, update_progress_bar, update_setup_button, 
    update_summary_panel, append_to_log
)
from smartcash.ui.setup.env_config.constants import STATUS_MESSAGES

class UILogger:
    """🔧 Logger yang mengarahkan output ke UI log accordion"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        
    def info(self, message: str) -> None:
        """Log info message ke UI"""
        append_to_log(self.ui_components, message, 'info')
        
    def warning(self, message: str) -> None:
        """Log warning message ke UI"""
        append_to_log(self.ui_components, message, 'warning')
        
    def error(self, message: str, exc_info=None) -> None:
        """Log error message ke UI"""
        append_to_log(self.ui_components, message, 'error')
        
    def success(self, message: str) -> None:
        """Log success message ke UI"""
        append_to_log(self.ui_components, message, 'success')


class EnvironmentConfigOrchestrator:
    """🎯 Main orchestrator untuk environment configuration"""
    
    def __init__(self):
        self.ui_components = None
        self.status_handler = None
        self.drive_handler = None
        self.progress_tracker = None
        self.logger = None
        self.setup_in_progress = False
        
    def initialize_ui(self) -> None:
        """🎨 Initialize UI components"""
        try:
            # Setup logger setelah UI ready
            self.logger = UILogger(self.ui_components)
            
            # Initialize handlers dengan UI logger
            self.status_handler = StatusHandler(self.logger)
            self.drive_handler = DriveHandler(self.logger)
            
            # Initialize progress tracker
            self.progress_tracker = ProgressTracker(self._update_progress_callback)
            
            # Create UI components
            self.ui_components = create_env_config_component(self._handle_setup_click)
            
            # Initial status check
            self._perform_initial_status_check()
            
            # Display UI
            self._display_ui()
            
            self.logger.info("✅ Environment config UI initialized successfully")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ UI initialization failed: {str(e)}")
            # Fallback ke print hanya untuk error initialization
            print(f"❌ Error initializing environment config: {str(e)}")
    
    def _setup_logger(self) -> None:
        """🔧 Setup UI logger (tidak ada console output)"""
        # Logger akan diset setelah UI components ready
        pass
    
    def _perform_initial_status_check(self) -> None:
        """🔍 Perform initial status check"""
        try:
            update_status_panel(self.ui_components, STATUS_MESSAGES['checking'], 'info')
            
            # Get comprehensive status dengan error handling
            status = self.status_handler.get_comprehensive_status()
            
            # Safe check untuk status
            if status and status.get('ready', False):
                update_status_panel(self.ui_components, STATUS_MESSAGES['ready'], 'success')
                update_setup_button(self.ui_components, False, "Environment Ready")
            else:
                update_status_panel(self.ui_components, STATUS_MESSAGES['setup_needed'], 'warning')
                update_setup_button(self.ui_components, True, "Setup Environment")
            
            # Update summary jika status valid
            if status:
                self._update_environment_summary(status)
            
        except Exception as e:
            error_msg = f"❌ Initial status check failed: {str(e)}"
            append_to_log(self.ui_components, error_msg, 'error')
            update_status_panel(self.ui_components, "❌ Error checking status", 'error')
    
    def _handle_setup_click(self) -> None:
        """🔘 Handle setup button click"""
        if self.setup_in_progress:
            append_to_log(self.ui_components, "⚠️ Setup sudah berjalan, mohon tunggu...", 'warning')
            return
        
        # Perform setup directly (no threading)
        self._perform_environment_setup()
    
    def _perform_environment_setup(self) -> None:
        """🚀 Perform complete environment setup"""
        self.setup_in_progress = True
        
        try:
            # Update UI untuk setup mode
            update_setup_button(self.ui_components, False, "⚙️ Setting up...")
            update_status_panel(self.ui_components, STATUS_MESSAGES['setup_running'], 'info')
            
            # Step 1: Check requirements
            self.progress_tracker.start_step('analysis')
            append_to_log(self.ui_components, "🔍 Analyzing setup requirements...", 'info')
            
            requirements = self.status_handler.check_setup_requirements()
            
            if not requirements.get('setup_needed', False):
                self._handle_setup_success("Environment sudah terkonfigurasi!")
                return
            
            self.progress_tracker.complete_step(True)
            
            # Step 2: Mount Drive (if needed)
            if 'mount_drive' in requirements.get('setup_steps', []):
                self.progress_tracker.start_step('drive_mount')
                append_to_log(self.ui_components, "📱 Mounting Google Drive...", 'info')
                
                mount_result = self.drive_handler.mount_drive()
                if not mount_result.get('success', False):
                    self._handle_setup_error(f"Drive mount failed: {mount_result.get('error', 'Unknown')}")
                    return
                
                self.progress_tracker.complete_step(True)
                append_to_log(self.ui_components, "✅ Google Drive mounted successfully", 'success')
            
            # Step 3: Complete Drive setup
            self.progress_tracker.start_step('folders')
            append_to_log(self.ui_components, "🔧 Performing complete Drive setup...", 'info')
            
            setup_results = self.drive_handler.perform_complete_setup()
            
            if not setup_results.get('success', False):
                failed_steps = setup_results.get('steps_failed', [])
                self._handle_setup_error(f"Setup failed at steps: {', '.join(failed_steps)}")
                return
            
            # Progress through remaining steps
            self.progress_tracker.complete_step(True)
            self.progress_tracker.start_step('configs')
            self.progress_tracker.complete_step(True)
            self.progress_tracker.start_step('symlinks')
            self.progress_tracker.complete_step(True)
            
            # Step 4: Final validation
            self.progress_tracker.start_step('validation')
            append_to_log(self.ui_components, "✅ Validating setup...", 'info')
            
            validation = self.status_handler.validate_post_setup()
            
            if validation.get('valid', False):
                self.progress_tracker.complete_step(True)
                self.progress_tracker.start_step('complete')
                self.progress_tracker.complete_step(True)
                self._handle_setup_success()
            else:
                issues = validation.get('issues', [])
                self._handle_setup_error(f"Validation failed: {', '.join(issues)}")
            
        except Exception as e:
            self._handle_setup_error(f"Setup error: {str(e)}")
        finally:
            self.setup_in_progress = False
            
    def _handle_setup_success(self, message: str = None) -> None:
        """✅ Handle successful setup completion"""
        success_msg = message or STATUS_MESSAGES['setup_success']
        
        update_status_panel(self.ui_components, success_msg, 'success')
        update_setup_button(self.ui_components, False, "Environment Ready")
        append_to_log(self.ui_components, success_msg, 'success')
        
        # Update summary dengan status terbaru
        try:
            status = self.status_handler.get_comprehensive_status()
            self._update_environment_summary(status)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update summary: {str(e)}")
    
    def _handle_setup_error(self, error_message: str) -> None:
        """❌ Handle setup error"""
        update_status_panel(self.ui_components, STATUS_MESSAGES['setup_failed'], 'error')
        update_setup_button(self.ui_components, True, "Retry Setup")
        append_to_log(self.ui_components, f"❌ {error_message}", 'error')
        
        if self.logger:
            self.logger.error(f"Setup failed: {error_message}")
    
    def _update_progress_callback(self, message: str, progress: int, is_error: bool) -> None:
        """📊 Callback untuk progress updates"""
        update_progress_bar(self.ui_components, progress, message, is_error)
        if not is_error:
            append_to_log(self.ui_components, message, 'info')
    
    def _update_environment_summary(self, status: Dict[str, Any]) -> None:
        """📋 Update environment summary panel"""
        try:
            summary_data = {
                'Environment Ready': status.get('ready', False),
                'Drive Mounted': status.get('drive', {}).get('drive_mounted', False),
                'Configs Available': status.get('configs', {}).get('total_configs', 0),
                'Symlinks Valid': status.get('symlinks', {}).get('valid_count', 0)
            }
            
            update_summary_panel(self.ui_components, summary_data)
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Failed to update summary: {str(e)}")
    
    def _display_ui(self) -> None:
        """🖥️ Display the main UI"""
        if self.ui_components and 'main_layout' in self.ui_components:
            from IPython.display import display
            display(self.ui_components['main_layout'])


# Entry point function
def initialize_environment_config_ui() -> None:
    """🚀 Initialize environment configuration UI - Entry point"""
    try:
        orchestrator = EnvironmentConfigOrchestrator()
        orchestrator.initialize_ui()
    except Exception as e:
        print(f"❌ Failed to initialize environment config UI: {str(e)}")
        # Fallback minimal display
        import ipywidgets as widgets
        from IPython.display import display
        
        error_widget = widgets.HTML(
            value=f"""
            <div style="padding: 20px; background: #ffebee; border: 1px solid #f44336; border-radius: 5px;">
                <h3 style="color: #c62828; margin-top: 0;">❌ Environment Config Error</h3>
                <p>Failed to initialize environment configuration: {str(e)}</p>
                <p><strong>Solusi:</strong> Restart runtime dan coba lagi</p>
            </div>
            """
        )
        display(error_widget)