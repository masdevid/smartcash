"""
File: smartcash/ui/setup/env_config/handlers/setup_handler.py
Deskripsi: Setup handler untuk environment configuration dengan proper workflow
"""

import os
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
            
            # Check if setup was cancelled
            if summary_data.get('cancelled', False):
                return summary_data
                
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
        from smartcash.ui.setup.env_config.constants import REQUIRED_FOLDERS, SYMLINK_MAP
        
        # Load existing summary data if available to maintain state
        summary_data = getattr(self, '_last_summary_data', {
            'drive_mounted': False,
            'mount_path': 'N/A',
            'configs_synced': 0,
            'symlinks_created': 0,
            'folders_created': 0,
            'required_folders': len(REQUIRED_FOLDERS),
            'required_symlinks': len(SYMLINK_MAP),
            'success': False,
            'verified_folders': [],
            'missing_folders': [],
            'verified_symlinks': [],
            'missing_symlinks': []
        })
        self._last_summary_data = summary_data
        
        try:
            # Step 1: Mount Drive (if not already mounted)
            if not summary_data.get('drive_mounted', False):
                self.logger.info("🔧 Step 1: Mounting Google Drive...")
                progress_tracker.update_step("Mounting Google Drive", 0)
                
                drive_result = self.drive_handler.mount_drive()
                
                # Check if user cancelled the drive mount
                if drive_result.get('cancelled', False):
                    self.logger.info("ℹ️ Drive mount was cancelled by user")
                    summary_data['success'] = False
                    summary_data['status_message'] = "Drive mount was cancelled"
                    summary_data['cancelled'] = True
                    return summary_data
                    
                summary_data['drive_mounted'] = drive_result.get('success', False)
                summary_data['mount_path'] = drive_result.get('mount_path', 'N/A')
            else:
                self.logger.info("✅ Google Drive already mounted")
            
            # Step 2: Create Folders (if not already created)
            if summary_data.get('folders_created', 0) < len(REQUIRED_FOLDERS):
                self.logger.info("📁 Step 2: Creating directories...")
                progress_tracker.update_step("Creating directories", 25)
                
                folder_result = self.folder_handler.create_required_folders()
                summary_data['folders_created'] = folder_result.get('created_count', 0)
                summary_data['symlinks_created'] = folder_result.get('symlinks_count', 0)
            else:
                self.logger.info("✅ Directories already created")
            
            config_result = self.config_handler.sync_configurations()
            summary_data['configs_synced'] = config_result.get('synced_count', 0)
            
            # Step 3: Create missing symlinks
            if 'missing_symlinks' in summary_data and summary_data['missing_symlinks']:
                self.logger.info("🔗 Step 3: Creating missing symlinks...")
                progress_tracker.update_step("Creating symlinks", 50)
                
                # Create missing symlinks
                created_symlinks = []
                for source, target in list(summary_data['missing_symlinks']):
                    try:
                        # Ensure parent directory exists
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        # Create symlink
                        os.symlink(source, target, target_is_directory=True)
                        created_symlinks.append((source, target))
                        self.logger.success(f"Created symlink: {target} -> {source}")
                    except Exception as e:
                        self.logger.warning(f"Failed to create symlink {target} -> {source}: {str(e)}")
                
                # Update verified/missing symlinks
                for symlink in created_symlinks:
                    if symlink in summary_data['missing_symlinks']:
                        summary_data['missing_symlinks'].remove(symlink)
                        summary_data['verified_symlinks'].append(symlink)
            
            # Step 4: Sync Configurations (only if all symlinks are created)
            if not summary_data.get('missing_symlinks', []):
                self.logger.info("⚙️ Step 4: Syncing configurations...")
                progress_tracker.update_step("Syncing configurations", 60)
                
                config_result = self.config_handler.sync_configurations()
                summary_data['configs_synced'] = config_result.get('synced_count', 0)
            else:
                self.logger.warning("⚠️ Skipping config sync due to missing symlinks")
                summary_data['configs_synced'] = 0
            
            # Step 5: Verify Setup
            self.logger.info("✅ Step 5: Verifying setup...")
            progress_tracker.update_step("Verifying setup", 80)
            
            # Verify all required folders exist
            verified_folders = []
            missing_folders = []
            for folder in REQUIRED_FOLDERS:
                if os.path.exists(folder) and os.path.isdir(folder):
                    verified_folders.append(folder)
                else:
                    missing_folders.append(folder)
            
            # Verify all symlinks exist and are valid
            verified_symlinks = []
            missing_symlinks = []
            for source, target in SYMLINK_MAP.items():
                if os.path.islink(target) and os.path.realpath(target) == os.path.realpath(source):
                    verified_symlinks.append((source, target))
                else:
                    missing_symlinks.append((source, target))
            
            # Update summary with verification results
            summary_data['verified_folders'] = verified_folders
            summary_data['missing_folders'] = missing_folders
            summary_data['verified_symlinks'] = verified_symlinks
            summary_data['missing_symlinks'] = missing_symlinks
            
            # Update counts for backward compatibility
            summary_data['folders_created'] = len(verified_folders)
            summary_data['symlinks_created'] = len(verified_symlinks)
            
            # Check if all required items exist
            all_verified = (
                not missing_folders and
                not missing_symlinks and
                summary_data['drive_mounted'] and
                (summary_data['configs_synced'] > 0 or not missing_symlinks)  # Config sync is optional if symlinks are okay
            )
            
            summary_data['success'] = all_verified
            
            # Log verification results
            if missing_folders:
                self.logger.warning(f"Missing folders: {', '.join(missing_folders)}")
            if missing_symlinks:
                self.logger.warning(f"Missing symlinks: {', '.join(f'{s} -> {t}' for s, t in missing_symlinks)}")
            
            # Complete
            progress_tracker.update_step("Setup complete", 100)
            
            # Update the setup summary with detailed information
            if 'setup_summary' in ui_components:
                from smartcash.ui.setup.env_config.components.setup_summary import update_setup_summary
                status_msg = (
                    "✅ Environment setup completed successfully!" if all_verified
                    else "⚠️ Setup completed with some issues"
                )
                update_setup_summary(
                    ui_components['setup_summary'],
                    status_message=status_msg,
                    status_type='success',
                    details=summary_data
                )
            
            self.logger.success("🎉 Environment setup completed successfully!")
            
        except Exception as e:
            error_msg = f"❌ Setup workflow failed: {str(e)}"
            self.logger.error(error_msg)
            summary_data['success'] = False
            
            # Update the setup summary with error information
            if 'setup_summary' in ui_components:
                from smartcash.ui.setup.env_config.components.setup_summary import update_setup_summary
                update_setup_summary(
                    ui_components['setup_summary'],
                    status_message=error_msg,
                    status_type='error',
                    details=summary_data
                )
            
        return summary_data
    
    def _set_running_state(self, ui_components: Dict[str, Any]) -> None:
        """🔄 Set UI ke running state"""
        ui_components['setup_button'].disabled = True
        ui_components['setup_button'].description = "⏳ Running Setup..."
        update_status_panel(ui_components['status_panel'], "Setup sedang berjalan...", "warning")
    
    def _set_completion_state(self, ui_components: Dict[str, Any], summary_data: Dict[str, Any]) -> None:
        """✅ Set UI ke completion state"""
        if 'setup_button' not in ui_components:
            return
            
        if summary_data.get('cancelled', False):
            # Handle cancellation case
            ui_components['setup_button'].disabled = False
            ui_components['setup_button'].button_style = ''
            ui_components['setup_button'].description = "▶️ Setup Environment"
            
            if 'status_panel' in ui_components:
                update_status_panel(ui_components['status_panel'],
                                 "ℹ️ Setup was cancelled", "info")
        elif summary_data.get('success', False):
            # Handle successful completion
            ui_components['setup_button'].disabled = False
            ui_components['setup_button'].button_style = 'success'
            ui_components['setup_button'].description = "✅ Setup Complete"
            
            if 'status_panel' in ui_components:
                update_status_panel(ui_components['status_panel'],
                                 "✅ Environment setup completed successfully!", "success")
        else:
            # Handle failure case
            ui_components['setup_button'].disabled = False
            ui_components['setup_button'].description = "🔄 Retry Setup"
            ui_components['setup_button'].button_style = 'danger'
            update_status_panel(ui_components['status_panel'], "Setup gagal, silakan coba lagi", "danger")
    
    def _set_error_state(self, ui_components: Dict[str, Any], error_msg: str) -> None:
        """Set error state in UI components"""
        if 'setup_button' in ui_components:
            ui_components['setup_button'].disabled = False
            ui_components['setup_button'].description = "❌ Retry Setup"
            ui_components['setup_button'].button_style = 'danger'
        update_status_panel(ui_components.get('status_panel'), error_msg, "danger")
        
    def setup_button_handler(self, button, ui_components: Dict[str, Any]) -> None:
        """Handle setup button click events
        
        Args:
            button: The button widget that was clicked
            ui_components: Dictionary containing UI components
        """
        try:
            # Clear log accordion if it exists
            if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'children'):
                log_output = ui_components['log_accordion'].children[0]
                if hasattr(log_output, 'clear_output'):
                    log_output.clear_output()
            
            # Disable button during setup
            button.disabled = True
            button.description = "🔄 Setting up..."
            button.button_style = 'info'
            
            # Run the setup process
            setup_result = self.run_full_setup(ui_components)
            
            # Update button state based on result
            if setup_result is True:
                button.description = "✅ Setup Complete"
                button.button_style = 'success'
            elif hasattr(setup_result, 'get') and setup_result.get('cancelled', False):
                # Handle cancellation case
                button.description = "▶️ Setup Environment"  # Reset to initial state
                button.button_style = ''  # Reset style
                button.disabled = False  # Re-enable the button
                return  # Exit early for cancellation
            else:
                button.description = "❌ Setup Failed"
                button.button_style = 'danger'
                button.disabled = False  # Allow retry on failure
        except Exception as e:
            error_msg = f"Setup error: {str(e)}"
            self.logger.error(error_msg)
            if 'status_panel' in ui_components:
                update_status_panel(ui_components['status_panel'], error_msg, "error")
            button.description = "❌ Error - Click to Retry"
            button.button_style = 'danger'
            button.disabled = False  # Allow retry on error
    
    def _create_dummy_logger(self):
        """📝 Create dummy logger fallback"""
        class DummyLogger:
            def info(self, msg): print(f"ℹ️ {msg}")
            def warning(self, msg): print(f"⚠️ {msg}")
            def error(self, msg): print(f"❌ {msg}")
            def success(self, msg): print(f"✅ {msg}")
        return DummyLogger()