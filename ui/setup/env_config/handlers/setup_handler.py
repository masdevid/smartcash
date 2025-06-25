"""
File: smartcash/ui/setup/env_config/handlers/setup_handler.py
Deskripsi: Setup handler untuk environment configuration dengan proper workflow
"""

import os
import time
import traceback
from enum import Enum
from typing import Dict, Any
from smartcash.ui.setup.env_config.handlers.drive_handler import DriveHandler
from smartcash.ui.setup.env_config.handlers.folder_handler import FolderHandler
from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler
from smartcash.ui.setup.env_config.utils.ui_updater import update_status_panel
from smartcash.ui.setup.env_config.utils.dual_progress_tracker import track_setup_progress, SetupStage
from smartcash.ui.setup.env_config.components.setup_summary import update_setup_summary

class SetupHandler:
    """🚀 Handler untuk mengelola setup environment workflow"""
    
    def __init__(self, logger=None):
        self.logger = logger or self._create_dummy_logger()
        self.drive_handler = DriveHandler(logger)
        self.folder_handler = FolderHandler(logger)
        self.config_handler = ConfigHandler()
        self._last_summary_data = None  # Store the last summary data
    
    def run_full_setup(self, ui_components: Dict[str, Any], clear_logs: bool = True) -> Dict[str, Any]:
        """🔄 Jalankan full setup workflow dengan progress tracking
        
        Args:
            ui_components: Dictionary containing UI components
            clear_logs: Whether to clear logs before starting (default: True)
            
        Returns:
            Dict containing setup results and status
        """
        # Initialize variables
        progress_tracker = None
        summary_data = {'status': 'error', 'message': 'Setup not started'}
        
        try:
            # Validate UI components
            if not ui_components or not isinstance(ui_components, dict):
                raise ValueError("Invalid UI components provided")
                
            # Initialize progress tracking with logger
            try:
                progress_tracker = track_setup_progress(ui_components, logger=self.logger)
                if not progress_tracker:
                    raise ValueError("Failed to initialize progress tracker")
            except Exception as e:
                self.logger.error(f"Failed to initialize progress tracker: {str(e)}")
                raise
            
            # Set running state
            self._set_running_state(ui_components)
            
            # Execute setup workflow
            summary_data = self._execute_setup_workflow(ui_components, progress_tracker)
            
            # Check if setup was cancelled
            if summary_data.get('cancelled', False):
                return summary_data
            
            # Store the summary data for future reference
            self._last_summary_data = summary_data
            
            # Only update the summary if we have new data
            if 'setup_summary' in ui_components:
                try:
                    status_msg = (
                        "✅ Environment setup completed successfully!" if summary_data.get('status') == 'success'
                        else "⚠️ Setup completed with some issues"
                    )
                    update_setup_summary(
                        ui_components['setup_summary'],
                        status_message=status_msg,
                        status_type='success' if summary_data.get('status') == 'success' else 'warning',
                        details=summary_data
                    )
                except Exception as e:
                    self.logger.error(f"Error updating setup summary: {str(e)}")
                    if progress_tracker:
                        progress_tracker.logger.error(f"Error updating setup summary: {str(e)}")
            
            # Set completion state if we have a successful status
            if summary_data.get('status') == 'success':
                self._set_completion_state(ui_components, summary_data)
            
            # Ensure we return a dictionary with at least a status
            if not isinstance(summary_data, dict):
                summary_data = {'status': 'success' if not summary_data else 'unknown'}
                
            return summary_data
            
        except Exception as e:
            # Get detailed error information
            error_type = type(e).__name__
            error_msg = str(e) or "No error message"
            error_traceback = traceback.format_exc()
            
            # Format the full error message
            full_error_msg = f"❌ Setup failed: {error_type}: {error_msg}"
            
            # Log the error with traceback
            self.logger.error(full_error_msg, exc_info=True)
            print(f"\n=== ERROR DETAILS ===\n{error_traceback}\n===================\n")
            
            # Try to update UI with error
            try:
                if progress_tracker and hasattr(progress_tracker, 'error'):
                    progress_tracker.error(full_error_msg)
                self._set_error_state(ui_components, full_error_msg)
            except Exception as ui_error:
                error_msg = f"Failed to update UI with error: {str(ui_error)}"
                self.logger.error(error_msg, exc_info=True)
                print(f"\n=== UI UPDATE ERROR ===\n{error_msg}\n{str(ui_error)}\n======================\n")
            
            # Return detailed error information
            return {
                'status': 'error',
                'error_type': error_type,
                'error': error_msg,
                'message': full_error_msg,
                'traceback': error_traceback,
                'success': False,
                'phase': 'initialization'
            }
        finally:
            # Clean up the progress tracker
            if progress_tracker and hasattr(progress_tracker, 'complete'):
                try:
                    progress_tracker.complete("Setup completed")
                except Exception as e:
                    self.logger.error(f"Error completing progress tracker: {str(e)}")
    
    def _execute_setup_workflow(self, ui_components: Dict[str, Any], progress_tracker) -> Dict[str, Any]:
        """🔧 Execute setup steps dengan progress tracking"""
        from smartcash.ui.setup.env_config.constants import REQUIRED_FOLDERS, SYMLINK_MAP
        
        # Load existing summary data if available to maintain state
        summary_data = getattr(self, '_last_summary_data', None)
        
        # Initialize with default values if summary_data is None or not a dictionary
        if not isinstance(summary_data, dict):
            summary_data = {
                'drive_mounted': False,
                'mount_path': 'N/A',
                'configs_synced': 0,
                'symlinks_created': 0,
                'folders_created': 0,
                'required_folders': len(REQUIRED_FOLDERS),
                'required_symlinks': len(SYMLINK_MAP),
                'status': 'pending',  # Use 'status' instead of 'success' for consistency
                'verified_folders': [],
                'missing_folders': [],
                'verified_symlinks': [],
                'missing_symlinks': []
            }
        
        # Ensure all required keys exist in the dictionary
        for key in ['drive_mounted', 'mount_path', 'configs_synced', 'symlinks_created', 
                   'folders_created', 'status', 'verified_folders', 'missing_folders',
                   'verified_symlinks', 'missing_symlinks']:
            if key not in summary_data:
                summary_data[key] = None if key == 'mount_path' else 0 if 'count' in key or 'created' in key else False
        
        # Store the initialized summary data
        self._last_summary_data = summary_data
        
        try:
            # Step 1: Mount Drive (if not already mounted)
            if not summary_data.get('drive_mounted', False):
                self.logger.info("🔧 Step 1: Mounting Google Drive...")
                progress_tracker.update_stage(SetupStage.DRIVE_MOUNT)
                progress_tracker.update_within_stage(0, "Starting drive mount...")
                
                drive_result = self.drive_handler.mount_drive()
                
                # Check if user cancelled the drive mount
                if drive_result.get('cancelled', False):
                    self.logger.info("ℹ️ Drive mount was cancelled by user")
                    summary_data['status'] = False
                    summary_data['status_message'] = "Drive mount was cancelled"
                    summary_data['cancelled'] = True
                    return summary_data
                    
                summary_data['drive_mounted'] = drive_result.get('success', False)
                summary_data['mount_path'] = drive_result.get('mount_path', 'N/A')
                
                if summary_data['drive_mounted']:
                    progress_tracker.update_within_stage(100, "Drive mounted successfully")
                    progress_tracker.complete_stage("Drive mounted successfully")
            else:
                self.logger.info("✅ Google Drive already mounted")
                progress_tracker.update_stage(SetupStage.DRIVE_MOUNT)
                progress_tracker.update_within_stage(100, "Drive already mounted")
            
            # Step 2: Create Folders (if not already created)
            try:
                progress_tracker.update_stage(SetupStage.FOLDER_SETUP)
                
                if summary_data.get('folders_created', 0) < len(REQUIRED_FOLDERS):
                    self.logger.info("📁 Step 2: Creating directories...")
                    progress_tracker.update_within_stage(0, "Starting folder creation...")
                    
                    # Execute folder creation
                    folder_result = self.folder_handler.create_required_folders()
                    created_count = folder_result.get('created_count', 0)
                    symlinks_count = folder_result.get('symlinks_count', 0)
                    
                    # Update summary data
                    summary_data['folders_created'] = created_count
                    summary_data['symlinks_created'] = symlinks_count
                    summary_data['backups_created'] = folder_result.get('backups_created', [])
                    summary_data['backups_count'] = folder_result.get('backups_count', 0)
                    
                    # Update progress based on results
                    backups_count = folder_result.get('backups_count', 0)
                    status_parts = []
                    
                    if created_count > 0:
                        status_parts.append(f"{created_count} folders")
                    if symlinks_count > 0:
                        status_parts.append(f"{symlinks_count} symlinks")
                    if backups_count > 0:
                        status_parts.append(f"{backups_count} backups")
                    
                    if status_parts:
                        status_msg = f"Created {', '.join(status_parts)}"
                        if backups_count > 0:
                            status_msg += " (backups in ~/data/backup)"
                        self.logger.info(status_msg)
                        progress_tracker.update_within_stage(100, status_msg)
                    else:
                        status_msg = "No new folders, symlinks, or backups needed"
                        self.logger.info(status_msg)
                        progress_tracker.update_within_stage(100, status_msg)
                    
                    # Complete the stage with the final status
                    progress_tracker.complete_stage("Folder setup completed")
                else:
                    self.logger.info("✅ Directories already created")
                    progress_tracker.update_within_stage(100, "All folders already exist")
                    progress_tracker.complete_stage("All folders already exist")
                    
            except Exception as e:
                error_msg = f"Error during folder creation: {str(e)}"
                self.logger.error(error_msg)
                progress_tracker.error(error_msg)
                summary_data['status'] = 'error'
                summary_data['message'] = error_msg
                return summary_data
            
            # Step 3: Sync Configurations
            progress_tracker.update_stage(SetupStage.CONFIG_SYNC)
            progress_tracker.update_within_stage(0, "Starting config sync...")
            
            config_result = self.config_handler.sync_configurations()
            synced_count = config_result.get('synced_count', 0)
            summary_data['configs_synced'] = synced_count
            
            if synced_count > 0:
                progress_tracker.update_within_stage(100, f"Synced {synced_count} configurations")
                progress_tracker.complete_stage("Configuration sync completed")
            else:
                progress_tracker.update_within_stage(100, "No configurations needed syncing")
            
            # Step 4: Create missing symlinks
            progress_tracker.update_stage(SetupStage.FOLDER_SETUP, "Verifying symlinks...")
            
            if 'missing_symlinks' in summary_data and summary_data['missing_symlinks']:
                self.logger.info("🔗 Step 4: Creating missing symlinks...")
                progress_tracker.update_within_stage(0, f"Creating {len(summary_data['missing_symlinks'])} symlinks...")
                
                # Create missing symlinks
                created_symlinks = []
                total_symlinks = len(summary_data['missing_symlinks'])
                
                for i, (source, target) in enumerate(list(summary_data['missing_symlinks'])):
                    try:
                        # Update progress
                        progress = int((i / total_symlinks) * 100)
                        progress_tracker.update_within_stage(
                            progress, 
                            f"Creating symlink {i+1}/{total_symlinks}: {os.path.basename(target)}"
                        )
                        
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
                
                if created_symlinks:
                    progress_tracker.update_within_stage(100, f"Created {len(created_symlinks)} symlinks")
                    progress_tracker.complete_stage("Symlink creation completed")
                else:
                    progress_tracker.update_within_stage(100, "No symlinks needed to be created")
            else:
                progress_tracker.update_within_stage(100, "No missing symlinks found")
            
            # Step 5: Verify Setup
            self.logger.info("✅ Step 5: Verifying setup...")
            progress_tracker.update_stage(SetupStage.ENV_SETUP)
            progress_tracker.update_within_stage(0, "Verifying environment setup...")
            
            # Verify all required folders exist
            verified_folders = []
            missing_folders = []
            total_folders = len(REQUIRED_FOLDERS)
            
            for i, folder in enumerate(REQUIRED_FOLDERS):
                progress = int((i / total_folders) * 100)
                progress_tracker.update_within_stage(progress, f"Verifying {os.path.basename(folder)}...")
                
                if os.path.exists(folder) and os.path.isdir(folder):
                    verified_folders.append(folder)
                else:
                    missing_folders.append(folder)
            
            # Verify all required symlinks exist
            verified_symlinks = []
            missing_symlinks = []
            total_symlinks = len(SYMLINK_MAP)
            
            for i, (source, target) in enumerate(SYMLINK_MAP.items()):
                progress = int((i / total_symlinks) * 100)
                progress_tracker.update_within_stage(
                    50 + (progress // 2),  # First half for folders, second half for symlinks
                    f"Verifying symlink {i+1}/{total_symlinks}: {os.path.basename(target)}"
                )
                
                if os.path.lexists(target) and os.path.islink(target):
                    verified_symlinks.append((source, target))
                else:
                    missing_symlinks.append((source, target))
            
            # Update summary data
            summary_data['verified_folders'] = verified_folders
            summary_data['missing_folders'] = missing_folders
            summary_data['verified_symlinks'] = verified_symlinks
            summary_data['missing_symlinks'] = missing_symlinks
            
            # Determine overall status
            has_errors = (
                len(missing_folders) > 0 or 
                len(missing_symlinks) > 0 or 
                not summary_data.get('drive_mounted', False)
            )
            
            if has_errors:
                summary_data['status'] = False
                summary_data['status_message'] = "Setup completed with some issues"
                progress_tracker.update_within_stage(100, "Verification completed with issues")
                progress_tracker.error("⚠️ Setup completed with some issues")
                
                # Log missing items
                if missing_folders:
                    self.logger.warning(f"Missing folders: {', '.join(os.path.basename(f) for f in missing_folders)}")
                if missing_symlinks:
                    self.logger.warning(f"Missing symlinks: {', '.join(t for _, t in missing_symlinks)}")
                if not summary_data.get('drive_mounted', False):
                    self.logger.warning("Google Drive is not mounted")
            else:
                summary_data['status'] = True
                summary_data['status_message'] = "Setup completed successfully"
                progress_tracker.update_within_stage(100, "Verification completed successfully")
                progress_tracker.complete(" Setup completed successfully")
            
            # Mark all stages as complete if successful
            if summary_data['status']:
                # Explicitly complete each stage
                for stage in [
                    SetupStage.INIT,
                    SetupStage.DRIVE_MOUNT,
                    SetupStage.CONFIG_SYNC,
                    SetupStage.FOLDER_SETUP,
                    SetupStage.ENV_SETUP
                ]:
                    progress_tracker.update_stage(stage)
                    progress_tracker.complete_stage(f"Completed {stage.name.replace('_', ' ').title()}")
                
                # Finally, set to complete stage
                progress_tracker.update_stage(SetupStage.COMPLETE)
                progress_tracker.complete("✅ All setup steps completed successfully")
            
            # Log final summary
            self.logger.info("\n📋 Setup Summary:" + "\n" + "="*50)
            self.logger.info(f"Drive Mounted: {'✅' if summary_data.get('drive_mounted', False) else '❌'}")
            self.logger.info(f"Folders Created: {len(verified_folders)}/{len(REQUIRED_FOLDERS)}")
            self.logger.info(f"Symlinks Created: {len(verified_symlinks)}/{len(SYMLINK_MAP)}")
            self.logger.info(f"Configs Synced: {summary_data.get('configs_synced', 0)}")
            
            if missing_folders:
                self.logger.warning(f"Missing Folders: {', '.join(os.path.basename(f) for f in missing_folders)}")
            if missing_symlinks:
                self.logger.warning(f"Missing Symlinks: {', '.join(f'-> {t}' for _, t in missing_symlinks)}")
            
            self.logger.info("="*50)
            
            return summary_data
            
        except Exception as e:
            error_msg = f"❌ Setup workflow failed: {str(e)}"
            self.logger.error(error_msg)
            summary_data['status'] = False
            
            # Update setup summary
            if 'setup_summary' in ui_components:
                update_setup_summary(
                    ui_components['setup_summary'],
                    status_message=error_msg,
                    status_type='error',
                    details=summary_data
                )
            
        return summary_data
    
    def _set_running_state(self, ui_components: Dict[str, Any]) -> None:
        """🔄 Set UI ke running state"""
        try:
            # Disable setup button
            ui_components['setup_button'].disabled = True
            ui_components['setup_button'].description = "⏳ Running Setup..."
            
            # Show progress container if it exists
            if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
                ui_components['progress_container'].layout.visibility = 'visible'
                ui_components['progress_container'].layout.display = 'flex'
            
            # Show progress bar if it exists in the progress tracker
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker and hasattr(progress_tracker, 'show'):
                progress_tracker.show()
            
            # Update status panel
            update_status_panel(ui_components['status_panel'], "Setup sedang berjalan...", "warning")
            
            # Force UI update
            if hasattr(ui_components.get('progress_bar', None), 'value'):
                ui_components['progress_bar'].value = 0
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in _set_running_state: {str(e)}", exc_info=True)
    
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
        elif summary_data.get('status') == 'success':
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
            ui_components['setup_button'].button_style = 'warning'  # Changed from 'danger' to 'warning'
            
            # Get error message from summary_data if available
            error_msg = summary_data.get('message', 'Setup failed, please try again')
            update_status_panel(ui_components['status_panel'], error_msg, "warning")
    
    def _set_error_state(self, ui_components: Dict[str, Any], error_msg: str) -> None:
        """Set error state in UI components"""
        if 'setup_button' in ui_components:
            ui_components['setup_button'].disabled = False
            ui_components['setup_button'].description = "❌ Retry Setup"
            ui_components['setup_button'].button_style = 'danger'
        update_status_panel(ui_components.get('status_panel'), error_msg, "danger")
        
    def setup_button_handler(self, button, ui_components: Dict[str, Any], clear_logs: bool = True) -> None:
        """Handle setup button click events
        
        Args:
            button: The button widget that was clicked
            ui_components: Dictionary containing UI components
            clear_logs: Whether to clear the log accordion before starting
        """
        try:
            # Only clear logs if explicitly requested (not on retry)
            if clear_logs and 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'children'):
                log_output = ui_components['log_accordion'].children[0]
                if hasattr(log_output, 'clear_output'):
                    log_output.clear_output()
            
            # Disable button during setup
            button.disabled = True
            button.description = "🔄 Setting up..."
            button.button_style = 'info'
            
            # Run the setup process with clear_logs parameter
            setup_result = self.run_full_setup(ui_components, clear_logs=clear_logs)
            
            # Update button state based on result
            if setup_result is True or (hasattr(setup_result, 'get') and setup_result.get('status') == 'success'):
                button.description = "✅ Setup Complete"
                button.button_style = 'success'
            elif hasattr(setup_result, 'get') and setup_result.get('cancelled', False):
                # Handle cancellation case
                button.description = "▶️ Setup Environment"  # Reset to initial state
                button.button_style = ''  # Reset style
                button.disabled = False  # Re-enable the button
                return  # Exit early for cancellation
            else:
                button.description = "🔄 Retry Setup"  # Changed from "Setup Failed" to "Retry Setup"
                button.button_style = 'warning'  # Changed from 'danger' to 'warning' for retry
                button.disabled = False  # Allow retry on failure
                
                # Update summary with error details if available
                if hasattr(setup_result, 'get') and 'message' in setup_result:
                    if 'setup_summary' in ui_components:
                        from smartcash.ui.setup.env_config.components.setup_summary import update_setup_summary
                        update_setup_summary(
                            ui_components['setup_summary'],
                            status_message=setup_result['message'],
                            status_type='error',
                            details=self._last_summary_data or {}
                        )
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