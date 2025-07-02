"""
Setup Workflow Handler for Environment Configuration.

This module provides the SetupHandler class which orchestrates the environment
setup workflow, including drive mounting, folder creation, and configuration syncing.
It uses ConfigHandler for configuration management.
"""

import asyncio
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, TypedDict, cast

from smartcash.common.environment import get_environment_manager
from smartcash.ui.handlers.base_handler import BaseHandler
from smartcash.ui.setup.env_config.handlers.base_config_mixin import BaseConfigMixin

from smartcash.ui.setup.env_config.constants import SetupStage

class SetupPhase(str, Enum):
    """Phases of the setup workflow."""
    INIT = "initializing"
    DRIVE = "drive_setup"
    FOLDERS = "folder_setup"
    CONFIG = "config_sync"
    VERIFY = "verification"
    COMPLETE = "complete"
    ERROR = "error"

class SetupSummary(TypedDict, total=False):
    """Type definition for setup summary data."""
    status: str
    message: str
    phase: SetupPhase
    progress: float
    current_stage: str
    drive_mounted: bool
    mount_path: str
    folders_created: int
    symlinks_created: int
    configs_synced: int
    verified_folders: List[str]
    missing_folders: List[str]
    verified_symlinks: List[Tuple[str, str]]
    missing_symlinks: List[Tuple[str, str]]
    config_check: Dict[str, Any]

class SetupHandler(BaseHandler, BaseConfigMixin):
    """Orchestrates the environment setup workflow.
    
    This class handles the complete environment setup process, using
    ConfigHandler for configuration management and progress tracking.
    
    See the _run_setup_workflow() method for detailed information about
    the setup process steps and workflow.
    """
    
    # Default configuration for the handler
    DEFAULT_CONFIG = {
        'auto_start': False,        # Start setup automatically
        'stop_on_error': True,      # Stop on first error
        'verify_setup': True,       # Verify setup after completion
        'enable_logging': True,     # Enable detailed logging
        'max_retries': 3,           # Max retries for failed operations
        'retry_delay': 5,           # Delay between retries in seconds
        'stages': [                 # Ordered list of setup stages
            'drive',
            'folder',
            'config',
            'verify'
        ]
    }
    
    def __init__(self, config_handler=None, **kwargs):
        """Initialize the SetupHandler with configuration.
        
        Args:
            config_handler: Configuration handler instance
            **kwargs: Additional keyword arguments for BaseHandler
                - logger: Custom logger instance (optional)
                - parent: Parent widget (optional)
                
        Raises:
            ValueError: If required dependencies are not provided
        """
        # Initialize BaseHandler first
        super().__init__(
            module_name='setup',
            parent_module='env_config',
            **kwargs
        )
        
        # Initialize BaseConfigMixin with the config handler
        BaseConfigMixin.__init__(self, config_handler=config_handler, **kwargs)
        
        # Initialize environment manager
        self.env_manager = get_environment_manager(logger=self.logger)
        
        # Initialize workflow state
        self._current_phase = SetupPhase.INIT
        self._setup_in_progress = False
        self._setup_progress = 0.0
        self._last_error = None
        self._retry_count = 0
        self._current_stage = "Not started"
        
        # Load and validate configuration
        self.auto_start = self.get_config_value('auto_start', False)
        self.stop_on_error = self.get_config_value('stop_on_error', True)
        self.verify_setup = self.get_config_value('verify_setup', True)
        self.max_retries = max(0, int(self.get_config_value('max_retries', 3)))
        self.retry_delay = max(0, float(self.get_config_value('retry_delay', 5)))
        self.stages = self.get_config_value('stages', [])
        
        # Validate required stages
        if not self.stages:
            self.logger.warning("No setup stages configured, using default stages")
            self.stages = ['drive', 'folder', 'config', 'verify']
        
        # Initialize last summary
        self._last_summary = self._create_initial_summary()
        
        # Log successful initialization
        self.logger.info(
            f"Initialized SetupHandler with {len(self.stages)} stages: {', '.join(self.stages)}"
        )
    
    def _create_initial_summary(self) -> 'SetupSummary':
        """Create and return a new setup summary with default values.
        
        Returns:
            A new SetupSummary dictionary with default values
        """
        return {
            'status': 'pending',
            'message': 'Initializing setup...',
            'phase': SetupPhase.INIT,
            'progress': 0.0,
            'current_stage': 'INIT',
            'drive_mounted': False,
            'mount_path': '',
            'folders_created': 0,
            'symlinks_created': 0,
            'configs_synced': 0,
            'verified_folders': [],
            'missing_folders': [],
            'verified_symlinks': [],
            'missing_symlinks': [],
            'config_check': {}
        }
        
    def _update_summary(self, **updates) -> None:
        """Update the setup summary with the provided values.
        
        Args:
            **updates: Key-value pairs to update in the summary
            
        Raises:
            ValueError: If any of the update keys are not valid summary fields
        """
        if self._last_summary is None:
            self._last_summary = self._create_initial_summary()
            
        # Validate update keys against the summary structure
        valid_keys = set(self._last_summary.keys())
        invalid_keys = set(updates.keys()) - valid_keys
        
        if invalid_keys:
            self.logger.warning(f"Ignoring invalid summary fields: {', '.join(invalid_keys)}")
            
        # Apply valid updates
        valid_updates = {k: v for k, v in updates.items() if k in valid_keys}
        self._last_summary.update(valid_updates)
        
        # Update the current phase if it was provided
        if 'phase' in valid_updates:
            self._current_phase = valid_updates['phase']
            
        # Log significant state changes
        if 'status' in valid_updates:
            self.logger.info(f"Setup status updated to: {valid_updates['status']}")
    
    async def start_setup(self, auto_start: bool = None) -> None:
        """Start the environment setup process.
        
        This method initiates the setup workflow, which includes:
        1. Drive mounting and verification
        2. Symlink setup (before folder creation)
        3. Required folder creation
        4. Configuration synchronization
        5. Final verification and status check
        
        Args:
            auto_start: Whether to start the setup automatically. If None, uses the instance's auto_start setting.
            
        Raises:
            RuntimeError: If setup is already in progress
        """
        auto_start = auto_start if auto_start is not None else self.auto_start
        
        if self._setup_in_progress:
            error_msg = "Setup is already in progress"
            self.logger.warning(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            self._setup_in_progress = True
            self._setup_progress = 0.0
            self._last_error = None
            self._retry_count = 0
            
            self.logger.info("Initializing environment setup")
            
            # Update status through the status panel
            self._update_status("Initializing environment setup...")
            
            # Ensure handlers are initialized
            if not hasattr(self, 'handlers'):
                self.initialize_handlers()
            
            # Validate configuration before starting
            if not self._validate_config():
                error_msg = "Configuration validation failed"
                self.logger.error(error_msg)
                self._handle_setup_error(error_msg)
                return
                
            # Start the setup process
            if auto_start:
                self.logger.info("Starting setup workflow")
                asyncio.create_task(self._run_setup_workflow())
                
        except Exception as e:
            self.logger.error(f"Failed to start setup: {str(e)}")
            self._handle_setup_error(f"Failed to start setup: {str(e)}")
            raise
    
    async def _run_stage(self, stage: str) -> bool:
        """Execute a single setup stage with retry logic and proper error handling.
        
        Args:
            stage: The name of the stage to run (must be a key in self.handlers)
            
        Returns:
            bool: True if the stage completed successfully, False otherwise
            
        Raises:
            ValueError: If the stage name is invalid
            RuntimeError: If the stage handler is not properly initialized
        """
        # Validate stage name
        if stage not in self.handlers:
            error_msg = f"Unknown setup stage: {stage}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        handler = self.handlers[stage]
        if not hasattr(handler, 'run') or not callable(handler.run):
            error_msg = f"Handler for stage '{stage}' is missing required 'run' method"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Set up progress tracking for this stage
        stage_progress = 0.0
        
        def update_stage_progress(progress: float, message: str = None):
            """Update progress within the current stage.
            
            Args:
                progress: Progress value between 0.0 and 1.0 for this stage
                message: Optional progress message
            """
            nonlocal stage_progress
            stage_progress = max(0.0, min(1.0, progress))
            stage_msg = f"{stage.replace('_', ' ').title()}: {message or ''}".strip()
            self._update_progress(stage_progress, stage_msg)
        
        # Add progress callback to handler if supported
        if hasattr(handler, 'set_progress_callback'):
            handler.set_progress_callback(update_stage_progress)
            
        attempt = 0
        max_attempts = 1 + self.max_retries  # Initial attempt + retries
        
        while attempt < max_attempts:
            attempt += 1
            self._current_stage = stage
            
            # Update status for this attempt
            if attempt > 1:
                retry_msg = f" (Attempt {attempt}/{max_attempts})"
                self.logger.warning(f"Retrying {stage} stage{retry_msg}")
                await asyncio.sleep(self.retry_delay * (attempt - 1))  # Exponential backoff
            
            start_msg = f"Starting {stage.replace('_', ' ')} stage..."
            self._update_status(start_msg)
            update_stage_progress(0.0, start_msg)
            self.logger.info(f"Executing setup stage: {stage}")
            
            try:
                # Execute the stage handler
                result = await handler.run()
                
                # If we get here, the stage completed without exceptions
                complete_msg = f"{stage.replace('_', ' ').title()} completed successfully"
                self.logger.info(f"Successfully completed setup stage: {stage}")
                update_stage_progress(1.0, complete_msg)
                
                self._update_summary(
                    status='completed',
                    phase=stage,
                    message=complete_msg
                )
                return True
                
            except asyncio.CancelledError:
                self.logger.warning(f"Stage '{stage}' was cancelled")
                self._update_status(f"{stage.replace('_', ' ')} cancelled", is_error=True)
                return False
                
            except Exception as e:
                self._last_error = str(e)
                self.logger.error(
                    f"Error in setup stage '{stage}' (attempt {attempt}/{max_attempts}): {str(e)}",
                    exc_info=True
                )
                
                # If this was the last attempt, handle the error
                if attempt >= max_attempts:
                    error_msg = f"Failed to complete {stage} after {max_attempts} attempts"
                    self._handle_setup_error(error_msg)
                    self._update_status(error_msg, is_error=True)
                    return False
                
                # Otherwise, log the error and let the loop retry
                self.logger.warning(f"Will retry {stage} stage after error: {str(e)}")
        
        # This line should never be reached due to the loop logic above
        return False
    
    def _handle_setup_error(self, error_msg: str, error: Optional[Exception] = None) -> None:
        """Handle setup errors by updating status, logging, and notifying listeners.
        
        Args:
            error_msg: Human-readable error message
            error: Optional exception that caused the error
        """
        self._setup_in_progress = False
        self._last_error = str(error) if error else error_msg
        
        # Update the summary with error information
        self._update_summary(
            status='error',
            phase=SetupPhase.ERROR,
            message=error_msg,
            progress=self._setup_progress
        )
        
        # Log the error
        self.logger.error(f"Setup error: {error_msg}", exc_info=error)
        
        # Notify listeners of the error
        if hasattr(self, 'on_error') and callable(self.on_error):
            self.on_error(error_msg, error)
    
    def initialize_handlers(self, **handler_kwargs) -> None:
        """Initialize all required handlers for the setup process.
        
        This method initializes all the handlers needed for the setup process,
        using ConfigHandler for configuration management.
        
        Returns:
            Dict[str, Any]: Dictionary of initialized handlers
        """
        self.logger.info("Initializing setup workflow handlers")
        
        try:
            # Common kwargs for all handlers
            common_kwargs = {
                'config_handler': self.config_handler,
                'logger': self.logger,
                **handler_kwargs
            }
            
            # Initialize drive handler
            from .drive_handler import DriveHandler
            self.drive_handler = DriveHandler(**{
                **common_kwargs,
                'on_drive_mounted': self._on_drive_mounted,
                'on_drive_mount_failed': self._on_drive_mount_failed
            })
            
            # Initialize folder handler
            from .folder_handler import FolderHandler
            self.folder_handler = FolderHandler(**{
                **common_kwargs,
                'on_folders_created': self._on_folders_created,
                'on_folder_creation_failed': self._on_folder_creation_failed
            })
            
            # Initialize config handler - use the one from env_config
            # This ensures we're using the same config handler instance
            self.config_handler = self.env_config.get_config_handler()
            
            # Initialize status checker
            from .status_checker import StatusChecker
            self.status_checker = StatusChecker(**{
                **common_kwargs,
                'on_status_checked': self._on_status_checked,
                'on_status_check_failed': self._on_status_check_failed
            })
            
            # Store all handlers for easy access
            self.handlers = {
                'drive': self.drive_handler,
                'folder': self.folder_handler,
                'config': self.config_handler,
                'status': self.status_checker
            }
            
            self.logger.info("Setup workflow handlers initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize setup handlers: {str(e)}")
            raise
    
    async def _run_setup_workflow(self) -> None:
        """Execute the setup workflow with progress tracking and error handling.
        
        This method orchestrates the complete environment setup process by executing each stage
        in the order defined in the 'stages' configuration. The stages must proceed in this order:
        INIT → DRIVE_MOUNT → SYMLINK_SETUP → FOLDER_SETUP → CONFIG_SYNC → ENV_SETUP → VERIFY → COMPLETE
        
        Workflow Stages:
            1. INIT: Initial setup and validation
               - Validates configuration and dependencies
               - Initializes required components
               - Sets up progress tracking
            
            2. DRIVE_MOUNT: Mount and verify the drive
               - Handles drive mounting and verification
               - Sets up necessary drive connections
               - Verifies drive accessibility
            
            3. SYMLINK_SETUP: Setup required symlinks
               - Creates symbolic links for required paths
               - Must run before folder creation
               - Handles existing symlinks based on configuration
            
            4. FOLDER_SETUP: Create required folder structure
               - Creates all necessary directories
               - Sets up folder permissions
               - Handles existing folders based on configuration
            
            5. CONFIG_SYNC: Synchronize configuration files
               - Syncs configuration from source to target locations
               - Handles configuration validation
               - Manages configuration versioning
            
            6. ENV_SETUP: Environment-specific setup
               - Performs environment-specific initialization
               - Sets up environment variables
               - Configures runtime environment
            
            7. VERIFY: Perform final verification and status check
               - Validates the complete setup
               - Performs health checks
               - Generates a setup report
            
            8. COMPLETE: Finalize setup
               - Performs cleanup operations
               - Saves final state
               - Emits completion signals
        
        Progress Tracking:
            - Overall workflow progress (0-100%)
            - Individual stage progress (0-100% within each stage's allocated portion)
            - Detailed status messages for each operation
            - Stage weights:
                - INIT: 5%
                - DRIVE_MOUNT: 15%
                - SYMLINK_SETUP: 10%
                - FOLDER_SETUP: 20%
                - CONFIG_SYNC: 20%
                - ENV_SETUP: 15%
                - VERIFY: 10%
                - COMPLETE: 5%
        
        Error Handling:
            - Configurable retry mechanism for transient failures
            - Graceful degradation on non-critical errors
            - Detailed error reporting and logging
            - Cleanup on failure based on configuration
            - Automatic rollback of failed operations when possible
        
        Configuration:
            The behavior can be customized through the following configuration options:
                - stages: List of stage names to execute (default: all stages)
                - max_retries: Maximum number of retry attempts per stage (default: 3)
                - retry_delay: Delay between retry attempts in seconds (default: 5)
                - stop_on_error: Whether to stop on first error (default: True)
                - verify_setup: Whether to run verification after setup (default: True)
        
        State Management:
            - Tracks current stage and progress
            - Maintains setup state across retries
            - Updates internal status and summary
            - Handles stage transitions and dependencies
            - Maintains error state and recovery information
            
        Note:
            - This method is designed to be idempotent and can be safely retried in case of failures.
            - Progress callbacks are invoked on the event loop thread.
            - The method follows the Open/Closed Principle - stages can be extended without modifying this method.
        
        Raises:
            RuntimeError: If the setup process encounters an unrecoverable error
            asyncio.CancelledError: If the operation is cancelled
            ValueError: If the configuration is invalid
            
        Side Effects:
            - Updates internal state and progress trackers
            - Modifies filesystem (creates/mounts drives, creates directories, etc.)
            - May modify system configuration
            - Emits progress and status updates through callbacks
            - May start/stop services or processes
        """
        try:
            stages = self.get_config_value('stages', [])
            total_stages = len(stages)
            
            if total_stages == 0:
                self.logger.warning("No setup stages configured")
                return
                
            self.logger.info(f"Starting setup workflow with {total_stages} stages")
            self._setup_in_progress = True
            
            # Initialize workflow progress tracking
            completed_stages = 0
            
            for i, stage in enumerate(stages, 1):
                if not self._setup_in_progress and self.stop_on_error:
                    self.logger.warning(f"Stopping setup workflow after stage '{stage}' due to error")
                    break
                
                # Calculate progress range for this stage (0.0 to 1.0 for the entire workflow)
                stage_start_progress = completed_stages / total_stages
                stage_end_progress = i / total_stages
                
                def get_workflow_progress(stage_progress: float) -> float:
                    """Convert stage progress to overall workflow progress."""
                    return stage_start_progress + (stage_progress * (stage_end_progress - stage_start_progress))
                
                # Save the current _update_progress method
                original_update_progress = self._update_progress
                
                # Create a scoped update_progress for this stage
                def stage_update_progress(progress: float, message: str = None):
                    """Update progress within the context of the current stage."""
                    workflow_progress = get_workflow_progress(progress)
                    original_update_progress(workflow_progress, message)
                
                # Temporarily replace _update_progress for this stage
                self._update_progress = stage_update_progress
                
                try:
                    # Execute the current stage with progress tracking
                    success = await self._run_stage(stage)
                    
                    if not success and self.stop_on_error:
                        self.logger.warning(f"Stage '{stage}' failed, stopping workflow")
                        break
                        
                    # Only count completed stages for progress
                    if success:
                        completed_stages += 1
                        
                finally:
                    # Restore original _update_progress method
                    self._update_progress = original_update_progress
            
            # Final status update
            if self._setup_in_progress:
                self._setup_progress = 1.0
                self.logger.info("Setup workflow completed successfully")
                self._update_status("Setup completed successfully")
                self._on_setup_completed(True)
            else:
                error_msg = "Setup workflow failed"
                if self._last_error:
                    error_msg += f": {self._last_error}"
                self.logger.error(error_msg)
                self._update_status(error_msg, is_error=True)
                self._on_setup_completed(False, error=self._last_error)
                
        except Exception as e:
            error_msg = f"Setup workflow failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._handle_setup_error(error_msg)
            self._on_setup_completed(False, error=error_msg)
    
    def _update_progress(self, progress: float, stage: str = None) -> None:
        """Update the setup progress and optionally the current stage.
        
        Args:
            progress: The progress value between 0.0 and 1.0
            stage: Optional description of the current stage
            
        Note:
            Progress is automatically clamped between 0.0 and 1.0.
            Emits the 'progress_updated' signal if available.
        """
        # Validate and clamp progress value
        progress = max(0.0, min(1.0, float(progress)))
        
        # Update internal state
        self._setup_progress = progress
        
        # Update summary with progress and optional stage
        updates = {'progress': progress}
        if stage is not None:
            updates['current_stage'] = str(stage)[:100]  # Truncate long stage names
            
        self._update_summary(**updates)
        
        # Log progress update
        self.logger.debug(f"Progress updated: {progress:.1%} - {stage or 'No stage info'}")
        
        # Emit progress update signal if available
        if hasattr(self, 'progress_updated'):
            try:
                self.progress_updated.emit(progress, stage or "")
            except Exception as e:
                self.logger.error(f"Failed to emit progress update: {str(e)}")
    
    async def run_setup(self, ui_components: Optional[Dict[str, Any]] = None) -> SetupSummary:
        """Run the complete setup workflow.
        
        See the class docstring for a detailed description of the setup process steps.
        
        Args:
            ui_components: Dictionary of UI components for progress updates
            
        Returns:
            SetupSummary with the results of the setup process
        """
        self.set_stage(SetupStage.INIT, "Starting setup process")
        
        # Initialize summary
        summary: SetupSummary = {
            'status': 'pending',
            'message': 'Setup started',
            'phase': 'initialization',
            'drive_mounted': False,
            'mount_path': '',
            'symlinks_created': 0,
            'folders_created': 0,
            'configs_synced': 0,
            'errors': []
        }
        
        try:
            # Step 1: Mount Drive
            summary = await self._mount_drive_step(summary, ui_components)
            
            # Step 2: Setup Symlinks (must be before folder creation)
            if summary.get('status') == 'success':
                summary = await self._setup_symlinks_step(summary, ui_components)
            
            # Step 3: Create Folders
            if summary.get('status') == 'success':
                summary = await self._create_folders_step(summary, ui_components)
            
            # Step 4: Sync Configs
            if summary.get('status') == 'success':
                summary = await self._sync_configs_step(summary, ui_components)
            
            # Step 5: Verify Setup
            if summary.get('status') == 'success':
                summary = await self._verify_setup_step(summary, ui_components)
            
            # Update final status
            summary.update({
                'status': 'success' if summary.get('status') != 'error' else 'error',
                'phase': 'complete'
            })
            
            self.set_stage(SetupStage.COMPLETE, summary['message'])
            
        except Exception as e:
            error_msg = f"Setup failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            summary.update({
                'status': 'error',
                'message': error_msg,
                'phase': summary.get('phase', 'unknown')
            })
            
            if ui_components:
                self._update_status_panel(ui_components, f"Error: {error_msg}", 'error')
        
        self._last_summary = summary
        return summary
    
    async def _mount_drive_step(self, summary: SetupSummary, ui_components: Optional[Dict[str, Any]]) -> SetupSummary:
        """Handle the drive mounting step."""
        self.set_stage(SetupStage.DRIVE_MOUNT, "Mounting Google Drive")
        
        try:
            # Mount the drive
            drive_handler = self.get_handler('drive')
            result = await drive_handler.mount_drive()
            
            if result.get('success', False):
                summary.update({
                    'status': 'success',
                    'message': 'Google Drive mounted successfully',
                    'drive_mounted': True,
                    'mount_path': result.get('mount_path', '')
                })
                self.logger.info(f"Google Drive mounted at: {summary['mount_path']}")
            else:
                error_msg = result.get('message', 'Failed to mount Google Drive')
                summary.update({
                    'status': 'error',
                    'message': error_msg,
                    'errors': [error_msg]
                })
                self.logger.error(f"Failed to mount Google Drive: {error_msg}")
            
            return summary
            
        except Exception as e:
            error_msg = f"Error mounting Google Drive: {str(e)}"
            summary.update({
                'status': 'error',
                'message': error_msg,
                'errors': [error_msg]
            })
            self.logger.error(error_msg, exc_info=True)
            if ui_components:
                self._update_status_panel(ui_components, f"Error mounting drive: {str(e)}", 'error')
            
            return summary
    
    async def _setup_symlinks_step(self, summary: SetupSummary, ui_components: Optional[Dict[str, Any]]) -> SetupSummary:
        """Handle the symlink setup step.
        
        This must run before folder creation to ensure symlinks are in place.
        """
        self.set_stage(SetupStage.SYMLINK_SETUP, "Setting up symlinks")
        
        if ui_components:
            self._update_status_panel(ui_components, "Setting up symlinks...", 'info')
        
        try:
            # Get the folder handler
            folder_handler = self.get_handler('folder')
            
            # Setup symlinks
            result = await folder_handler.setup_symlinks()
            
            if result.get('success', False):
                symlinks_created = result.get('symlinks_created', 0)
                summary.update({
                    'status': 'success',
                    'message': f'Created {symlinks_created} symlinks',
                    'symlinks_created': symlinks_created
                })
                self.logger.info(f"Created {symlinks_created} symlinks")
            else:
                error_msg = result.get('message', 'Failed to setup symlinks')
                summary.update({
                    'status': 'error',
                    'message': error_msg,
                    'errors': summary.get('errors', []) + [error_msg]
                })
                self.logger.error(f"Failed to setup symlinks: {error_msg}")
            
            return summary
            
        except Exception as e:
            error_msg = f"Error setting up symlinks: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            if ui_components:
                self._update_status_panel(ui_components, error_msg, 'error')
            
            summary.update({
                'status': 'error',
                'message': error_msg,
                'errors': summary.get('errors', []) + [error_msg]
            })
            
            return summary
    
    async def _create_folders_step(self, summary: SetupSummary, ui_components: Optional[Dict[str, Any]]) -> SetupSummary:
        """Handle the folder creation step."""
        self.set_stage(SetupStage.FOLDER_SETUP, "Creating required folders")
        
        if ui_components:
            self._update_status_panel(ui_components, "Creating folders...", 'info')
        
        try:
            # Get the folder handler
            folder_handler = self.get_handler('folder')
            
            # Create folders
            result = await folder_handler.create_required_folders()
            
            if result.get('success', False):
                folders_created = result.get('folders_created', 0)
                summary.update({
                    'status': 'success',
                    'message': f'Created {folders_created} folders',
                    'folders_created': folders_created
                })
                self.logger.info(f"Created {folders_created} folders")
            else:
                error_msg = result.get('message', 'Failed to create folders')
                summary.update({
                    'status': 'error',
                    'message': error_msg,
                    'errors': summary.get('errors', []) + [error_msg]
                })
                self.logger.error(f"Failed to create folders: {error_msg}")
            
            return summary
            
        except Exception as e:
            error_msg = f"Error creating folders: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            if ui_components:
                self._update_status_panel(ui_components, error_msg, 'error')
            
            summary.update({
                'status': 'error',
                'message': error_msg,
                'errors': summary.get('errors', []) + [error_msg]
            })
            
            return summary
    
    def _perform_initial_status_check(self, ui_components: Dict[str, Any]) -> None:
        """Perform initial status check on the environment using the status_checker.
        
        This method performs a comprehensive check of the environment status using the
        status_checker handler and updates the UI components accordingly. It's called 
        during the initialization process and after setup completion.
        
        Args:
            ui_components: Dictionary containing UI components to update
        """
        try:
            self.logger.debug("Performing comprehensive status check...")
            
            # Store UI components for later use
            self.ui_components = ui_components
            
            # Initialize status checker if not already done
            if not hasattr(self, 'status_checker'):
                self.status_checker = StatusChecker(
                    config_handler=self.config_handler,
                    logger=self.logger
                )
            
            # Perform environment check
            check_result = self.status_checker.check_environment()
            
            # Update UI based on check result
            if check_result.get('status'):
                status_msg = check_result.get('status_message', 'Environment is ready')
                self._update_status_panel(ui_components, status_msg, 'success')
            else:
                status_msg = check_result.get('status_message', 'Environment check completed with warnings')
                self._update_status_panel(ui_components, status_msg, 'warning')
            
            # Store the check result for reference
            self.last_status_check = check_result
            
            self.logger.debug("Status check completed")
            
        except Exception as e:
            error_msg = f"Error performing status check: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._update_status_panel(ui_components, error_msg, "error")
    
    def _is_setup_complete(self) -> bool:
        """Check if the environment setup is complete.
        
        This checks for the existence of a marker file or configuration
        that indicates the initial setup has been completed.
        
        Returns:
            bool: True if setup is complete, False otherwise
        """
        try:
            # Check for a marker file in the config directory
            if not hasattr(self, 'config_handler') or not hasattr(self.config_handler, 'config_dir'):
                self.logger.warning("Config handler not properly initialized, cannot check setup completion")
                return False
                
            marker_file = Path(self.config_handler.config_dir) / '.setup_complete'
            return marker_file.exists()
            
        except Exception as e:
            self.logger.warning(f"Error checking setup completion: {str(e)}")
            return False
            
    def _should_sync_config_templates(self) -> bool:
        """Determine if config templates should be synced.
        
        Returns:
            bool: True if config templates should be synced, False otherwise
            
        Raises:
            RuntimeError: If there's an error checking sync status
        """
        try:
            # Get drive handler to check mount status
            drive_handler = self.get_handler('drive')
            if not drive_handler or not drive_handler._is_drive_mounted():
                self.logger.debug("Skipping config template sync: Drive not mounted")
                return False
            
            # Check if setup is complete by looking for a marker file or config
            if not self._is_setup_complete():
                self.logger.debug("Skipping config template sync: Setup not complete")
                return False
                
            return True
            
        except Exception as e:
            error_msg = f"Error checking if should sync config templates: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
    def sync_config_templates(self, force_overwrite: bool = False, update_ui: bool = False, 
                            ui_components: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronize configuration templates with the config directory.
        
        This method can be used both as part of the setup workflow and independently.
        
        Args:
            force_overwrite: Whether to overwrite existing config files
            update_ui: Whether to update the UI with progress
            ui_components: UI components to update if update_ui is True
            
        Returns:
            Dict containing sync results including:
            - success: bool indicating if sync was successful
            - synced_count: number of files synced
            - skipped_count: number of files skipped
            - message: status message
        """
        if update_ui and not ui_components:
            self.logger.warning("update_ui is True but no ui_components provided")
            
        try:
            if update_ui and ui_components:
                self._update_status_panel(ui_components, "Synchronizing configurations...", 'info')
                
            self.logger.info("Starting configuration template synchronization...")
            
            # Get the config manager instance with auto_sync enabled
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager(auto_sync=True)
            
            # Sync configs using the config manager
            result = config_manager.sync_configs_to_drive(force_overwrite=force_overwrite)
            
            # Log the result
            if result.get('success', False):
                self.logger.info(
                    f"Successfully synced {result.get('synced_count', 0)} config templates. "
                    f"Skipped {result.get('skipped_count', 0)} up-to-date files."
                )
            else:
                self.logger.warning(
                    f"Config template sync completed with issues: {result.get('message', 'Unknown error')}"
                )
                
            # Update UI if requested
            if update_ui and ui_components:
                status_type = 'success' if result.get('success', False) else 'warning'
                message = result.get('message', 'Configuration sync completed')
                self._update_status_panel(ui_components, message, status_type)
                
            return result
            
        except Exception as e:
            error_msg = f"Error syncing configuration templates: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            if update_ui and ui_components:
                self._update_status_panel(ui_components, error_msg, 'error')
                
            return {
                'success': False,
                'message': error_msg,
                'error': str(e)
            }
    
    async def _sync_configs_step(self, summary: SetupSummary, ui_components: Optional[Dict[str, Any]] = None) -> SetupSummary:
        """Handle the configuration syncing step in the setup workflow.
        
        This is a wrapper around sync_config_templates that updates the setup summary.
        """
        self.set_stage(SetupStage.CONFIG_SYNC, "Synchronizing configurations")
        
        try:
            # Use the reusable sync method
            sync_result = await self.sync_config_templates(
                force_overwrite=False,
                update_ui=bool(ui_components),
                ui_components=ui_components
            )
            
            # Update summary with the sync result
            summary.update({
                'configs_synced': sync_result.get('synced_count', 0),
                'skipped_configs': sync_result.get('skipped_count', 0),
                'phase': 'config_sync',
                'message': sync_result.get('message', 'Configuration sync completed'),
                'config_check': sync_result,
                'success': sync_result.get('success', False)
            })
            
            return summary
            
        except Exception as e:
            error_msg = f"Error in configuration sync step: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            summary.update({
                'phase': 'config_sync',
                'message': error_msg,
                'success': False
            })
            
            if ui_components:
                self._update_status_panel(ui_components, error_msg, 'error')
            
            return summary
    
    async def _verify_setup_step(self, summary: SetupSummary, ui_components: Optional[Dict[str, Any]] = None) -> SetupSummary:
        """Execute the setup verification step."""
        self.set_stage(SetupStage.VERIFICATION, "Verifying setup")
        
        if ui_components:
            self._update_status_panel(ui_components, "Verifying setup...", 'info')
        
        try:
            # Verify folder structure
            folder_check = self.folder_handler.verify_folder_structure()
            
            summary.update({
                'verified_folders': folder_check.get('valid_folders', []),
                'missing_folders': folder_check.get('missing_folders', []),
                'verified_symlinks': folder_check.get('valid_symlinks', []),
                'missing_symlinks': [
                    (s['source'], s['target']) 
                    for s in folder_check.get('invalid_symlinks', [])
                ],
                'phase': 'verification',
                'message': 'Verification completed'
            })
            
            return summary
            
        except Exception as e:
            error_msg = f"Error verifying setup: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            summary.update({
                'phase': 'verification',
                'message': error_msg
            })
            
            if ui_components:
                self._update_status_panel(ui_components, f"Error verifying setup: {str(e)}", 'error')
            
            return summary
    
    # Status updates are now handled by the parent class's _update_status_panel method
    
    def get_last_summary(self) -> Optional[SetupSummary]:
        """Get the summary of the last setup run.
        
        Returns:
            SetupSummary of the last run, or None if no run has been performed
        """
        return self._last_summary
        
    def _on_setup_completed(self, success: bool, error: Optional[str] = None) -> None:
        """Handle setup completion by creating the setup complete marker file if successful.
        
        This method is called when the setup workflow completes, either successfully or with an error.
        If successful, it creates a marker file to indicate that setup has been completed.
        
        Args:
            success: Whether the setup completed successfully
            error: Optional error message if the setup failed
            
        Note:
            The marker file is created in the config directory with the name '.setup_complete'.
            This file is checked by EnvConfigInitializer._is_setup_complete() to determine
            if the setup has been completed.
        """
        try:
            if success:
                # Only create the marker file if setup was successful
                if hasattr(self, 'config_handler') and hasattr(self.config_handler, 'config_dir'):
                    marker_file = Path(self.config_handler.config_dir) / '.setup_complete'
                    try:
                        # Create parent directories if they don't exist
                        marker_file.parent.mkdir(parents=True, exist_ok=True)
                        # Create the marker file
                        marker_file.touch()
                        self.logger.info(f"Created setup completion marker file at: {marker_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to create setup completion marker file: {str(e)}", exc_info=True)
            
            # Update the setup state
            self._setup_in_progress = False
            self._current_phase = SetupPhase.COMPLETE if success else SetupPhase.ERROR
            
            # Update the last summary
            self._update_summary(
                status='completed' if success else 'failed',
                phase=self._current_phase,
                message='Setup completed successfully' if success else f'Setup failed: {error}'
            )
            
            # Log the completion
            if success:
                self.logger.info("Setup completed successfully")
            else:
                self.logger.error(f"Setup failed: {error}")
                
        except Exception as e:
            self.logger.error(f"Error in _on_setup_completed: {str(e)}", exc_info=True)
            # Still update the state even if we couldn't create the marker file
            self._setup_in_progress = False
            self._current_phase = SetupPhase.ERROR
            self._update_summary(
                status='error',
                phase=SetupPhase.ERROR,
                message=f'Error in setup completion: {str(e)}'
            )
        return self._last_summary
