"""
File: smartcash/ui/setup/env_config/handlers/setup_handler.py

Setup Handler - Refactored dengan arsitektur baru.

Handler untuk mengatur workflow setup environment dengan stage-based operations.
"""

import asyncio
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple, TypedDict, Union
from datetime import datetime

# Import core handlers
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.ui.core.handlers.config_handler import ConfigurableHandler

# Import environment manager
from smartcash.common.environment import get_environment_manager

# Import constants
from smartcash.ui.setup.env_config.constants import SetupStage, STAGE_WEIGHTS

# Import operation handlers
from smartcash.ui.setup.env_config.operations.drive_operation import DriveOperation
from smartcash.ui.setup.env_config.operations.folder_operation import FolderOperation
from smartcash.ui.setup.env_config.operations.config_operation import ConfigOperation


class SetupPhase(str, Enum):
    """Phases setup workflow."""
    INIT = "initializing"
    DRIVE = "drive_setup"
    FOLDERS = "folder_setup"
    CONFIG = "config_sync"
    VERIFY = "verification"
    COMPLETE = "complete"
    ERROR = "error"


class SetupSummary(TypedDict, total=False):
    """Type definition untuk setup summary data."""
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


class SetupHandler(OperationHandler, ConfigurableHandler):
    """Handler untuk setup environment workflow.
    
    Mengelola complete environment setup process dengan stage-based operations:
    - Drive mounting
    - Folder creation
    - Config synchronization
    - Verification
    """
    
    def __init__(self):
        """Initialize setup handler."""
        # Initialize parent classes
        OperationHandler.__init__(
            self,
            module_name='env_config',
            parent_module='setup'
        )
        ConfigurableHandler.__init__(
            self,
            module_name='env_config',
            parent_module='setup'
        )
        
        # Initialize environment manager
        self.env_manager = get_environment_manager()
        
        # Initialize status checker
        from smartcash.ui.setup.env_config.services.status_checker import EnvironmentStatusChecker
        self.status_checker = EnvironmentStatusChecker()
        
        # Initialize workflow state
        self._current_stage = SetupStage.INIT
        self._current_phase = SetupPhase.INIT
        self._setup_in_progress = False
        self._setup_progress = 0.0
        self._last_error = None
        self._retry_count = 0
        
        # Initialize operation handlers
        self._drive_operation = DriveOperation()
        self._folder_operation = FolderOperation()
        self._config_operation = ConfigOperation()
        
        # Initialize summary
        self._last_summary = self._create_initial_summary()
        
        # UI components akan di-set dari luar
        self._ui_components = {}
        
        # Initialize operations
        self._operations = {
            'start_setup': self._run_setup_workflow,
            'cancel_setup': self.cancel_operation,
            'verify_environment': self.verify_environment,
            'reset_environment': self.reset_environment,
            'sync_templates': self.sync_config_templates
        }
        
        self.logger.info("🔧 Setup handler initialized")
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations.
        
        Returns:
            Dictionary mapping operation names to their handler methods
        """
        return self._operations
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize the handler.
        
        Returns:
            Dictionary containing initialization status
        """
        try:
            # Initialize operation handlers
            self._drive_operation.initialize()
            self._folder_operation.initialize()
            self._config_operation.initialize()
            
            # Mark as initialized
            self._is_initialized = True
            
            return {
                'status': True,
                'message': 'Setup handler initialized successfully',
                'operations': list(self._operations.keys())
            }
            
        except Exception as e:
            error_msg = f"Failed to initialize SetupHandler: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'error': error_msg,
                'message': 'Failed to initialize setup handler'
            }
    
    def set_ui_components(self, ui_components: Dict[str, Any]):
        """Set UI components untuk updates.
        
        Args:
            ui_components: Dictionary berisi UI components
        """
        self._ui_components = ui_components
    
    def _create_initial_summary(self) -> SetupSummary:
        """Create initial setup summary.
        
        Returns:
            SetupSummary dengan default values
        """
        return {
            'status': 'pending',
            'message': 'Setup belum dimulai',
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
    
    def _update_summary(self, **updates):
        """Update setup summary.
        
        Args:
            **updates: Updates untuk summary
        """
        self._last_summary.update(updates)
    
    def _update_progress(self, stage: SetupStage, stage_progress: float = 1.0):
        """Update overall progress berdasarkan stage dengan container-aware updates.
        
        Args:
            stage: Current stage
            stage_progress: Progress dalam stage (0.0-1.0)
        """
        # Calculate cumulative progress dari stages sebelumnya
        completed_weight = sum(
            STAGE_WEIGHTS[s] for s in SetupStage 
            if s.value < stage.value and s != SetupStage.ERROR
        )
        
        # Add current stage progress
        current_weight = STAGE_WEIGHTS[stage] * stage_progress
        
        # Total progress
        total_progress = completed_weight + current_weight
        
        # Update state
        self._setup_progress = min(100.0, total_progress)
        self._current_stage = stage
        
        # Update summary
        self._update_summary(
            progress=self._setup_progress,
            current_stage=stage.name
        )
        
        # Update UI dengan container-aware access
        self._update_ui_progress(total_progress / 100.0, f"Stage: {stage.name} ({self._setup_progress:.1f}%)")
    
    def _update_ui_progress(self, progress: float, message: str):
        """Update UI progress dengan container-aware access."""
        try:
            # Try direct access first
            if 'progress_tracker' in self._ui_components:
                tracker = self._ui_components['progress_tracker']
                if hasattr(tracker, 'update'):
                    tracker.update(progress, message)
                    return
            
            # Try container-based access
            for container_key in ['main_container', 'summary_container', 'action_container']:
                if container_key in self._ui_components:
                    container = self._ui_components[container_key]
                    # Check if container has progress tracker
                    if hasattr(container, 'progress_tracker'):
                        tracker = container.progress_tracker
                        if hasattr(tracker, 'update'):
                            tracker.update(progress, message)
                            return
                    # Check if container has children with progress tracker
                    elif hasattr(container, 'children'):
                        for child in container.children:
                            if hasattr(child, 'progress_tracker') or getattr(child, '__class__', None).__name__ == 'ProgressTracker':
                                tracker = child.progress_tracker if hasattr(child, 'progress_tracker') else child
                                if hasattr(tracker, 'update'):
                                    tracker.update(progress, message)
                                    return
            
            # Fallback: log the progress
            self.logger.info(f"📊 Progress: {progress*100:.1f}% - {message}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update UI progress: {str(e)}")
            # Don't raise, just log
    
    def perform_action(self, action: str, **kwargs) -> Dict[str, Any]:
        """Perform setup action.
        
        Args:
            action: Action name
            **kwargs: Action parameters
            
        Returns:
            Dictionary berisi hasil action
        """
        if action == 'setup_environment':
            return self._run_setup_workflow()
        elif action == 'verify_setup':
            return self._verify_setup()
        elif action == 'reset_setup':
            return self._reset_setup()
        else:
            return {
                'status': False,
                'message': f'Unknown action: {action}'
            }
    
    def _run_setup_workflow(self) -> Dict[str, Any]:
        """Run complete setup workflow.
        
        Returns:
            Dictionary berisi hasil setup
        """
        try:
            self.logger.info("🚀 Starting setup workflow")
            
            # Prevent concurrent setup
            if self._setup_in_progress:
                return {
                    'status': False,
                    'message': 'Setup sudah berjalan'
                }
            
            self._setup_in_progress = True
            self._retry_count = 0
            
            # Update status
            self._update_summary(
                status='running',
                message='Setup sedang berjalan...',
                phase=SetupPhase.INIT
            )
            
            # Execute stages dalam urutan yang benar
            stages = [
                (SetupStage.INIT, self._stage_init),
                (SetupStage.DRIVE_MOUNT, self._stage_drive_mount),
                (SetupStage.CONFIG_SYNC, self._stage_config_sync),
                (SetupStage.FOLDER_SETUP, self._stage_folder_setup),
                (SetupStage.VERIFY, self._stage_verify),
                (SetupStage.COMPLETE, self._stage_complete)
            ]
            
            for stage, stage_func in stages:
                try:
                    self.logger.info(f"🔄 Executing stage: {stage.name}")
                    
                    # Update progress
                    self._update_progress(stage, 0.0)
                    
                    # Execute stage
                    result = stage_func()
                    
                    if not result.get('status', False):
                        # Stage failed
                        self._update_summary(
                            status='error',
                            message=f"Stage {stage.name} failed: {result.get('message', 'Unknown error')}",
                            phase=SetupPhase.ERROR
                        )
                        return result
                    
                    # Stage berhasil
                    self._update_progress(stage, 1.0)
                    
                except Exception as e:
                    error_msg = f"Error in stage {stage.name}: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    
                    self._update_summary(
                        status='error',
                        message=error_msg,
                        phase=SetupPhase.ERROR
                    )
                    
                    return {
                        'status': False,
                        'message': error_msg,
                        'error': str(e)
                    }
            
            # All stages completed
            self._update_summary(
                status='success',
                message='Setup berhasil diselesaikan!',
                phase=SetupPhase.COMPLETE
            )
            
            return {
                'status': True,
                'message': 'Setup berhasil diselesaikan!',
                'summary': self._last_summary
            }
            
        except Exception as e:
            error_msg = f"Setup workflow failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            self._update_summary(
                status='error',
                message=error_msg,
                phase=SetupPhase.ERROR
            )
            
            return {
                'status': False,
                'message': error_msg,
                'error': str(e)
            }
        finally:
            self._setup_in_progress = False
    
    def _stage_init(self) -> Dict[str, Any]:
        """Initialize stage."""
        try:
            self.logger.info("📋 Initializing setup...")
            
            # Validate environment
            if not self.env_manager:
                return {
                    'status': False,
                    'message': 'Environment manager tidak tersedia'
                }
            
            # Reset counters
            self._update_summary(
                folders_created=0,
                symlinks_created=0,
                configs_synced=0,
                verified_folders=[],
                missing_folders=[],
                verified_symlinks=[],
                missing_symlinks=[],
                config_check={}
            )
            
            return {
                'status': True,
                'message': 'Initialization completed'
            }
            
        except Exception as e:
            return {
                'status': False,
                'message': f'Initialization failed: {str(e)}'
            }
    
    def _stage_drive_mount(self) -> Dict[str, Any]:
        """Mount drive stage."""
        try:
            self.logger.info("💾 Mounting drive...")
            
            # Delegate ke drive operation
            result = self._drive_operation.mount_drive()
            
            if result.get('status', False):
                self._update_summary(
                    drive_mounted=True,
                    mount_path=result.get('mount_path', '')
                )
            
            return result
            
        except Exception as e:
            return {
                'status': False,
                'message': f'Drive mount failed: {str(e)}'
            }
    
    def _stage_folder_setup(self) -> Dict[str, Any]:
        """Setup folders stage."""
        try:
            self.logger.info("📁 Setting up folders...")
            
            # Delegate ke folder operation
            result = self._folder_operation.create_folders()
            
            if result.get('status', False):
                self._update_summary(
                    folders_created=result.get('folders_created', 0),
                    symlinks_created=result.get('symlinks_created', 0)
                )
            
            return result
            
        except Exception as e:
            return {
                'status': False,
                'message': f'Folder setup failed: {str(e)}'
            }
    
    def _stage_config_sync(self) -> Dict[str, Any]:
        """Sync config stage."""
        try:
            self.logger.info("⚙️ Syncing configurations...")
            
            # Delegate ke config operation
            result = self._config_operation.sync_configs()
            
            if result.get('status', False):
                self._update_summary(
                    configs_synced=result.get('configs_synced', 0)
                )
            
            return result
            
        except Exception as e:
            return {
                'status': False,
                'message': f'Config sync failed: {str(e)}'
            }
    
    def _stage_verify(self) -> Dict[str, Any]:
        """Verify setup stage."""
        try:
            self.logger.info("🔍 Verifying setup...")
            
            # Verify components
            drive_ok = self._verify_drive()
            folders_ok = self._verify_folders()
            configs_ok = self._verify_configs()
            
            all_ok = drive_ok and folders_ok and configs_ok
            
            return {
                'status': all_ok,
                'message': 'Verification completed' if all_ok else 'Verification failed',
                'drive_ok': drive_ok,
                'folders_ok': folders_ok,
                'configs_ok': configs_ok
            }
            
        except Exception as e:
            return {
                'status': False,
                'message': f'Verification failed: {str(e)}'
            }
    
    def _stage_complete(self) -> Dict[str, Any]:
        """Complete setup stage."""
        try:
            self.logger.info("✅ Completing setup...")
            
            # Final updates
            self._update_summary(
                status='success',
                message='Setup completed successfully!',
                phase=SetupPhase.COMPLETE
            )
            
            # Update UI
            if 'setup_summary' in self._ui_components:
                self._update_ui_summary()
            
            return {
                'status': True,
                'message': 'Setup completed successfully!'
            }
            
        except Exception as e:
            return {
                'status': False,
                'message': f'Completion failed: {str(e)}'
            }
    
    def _verify_drive(self) -> bool:
        """Verify drive mount menggunakan status checker."""
        try:
            drive_status = self.status_checker.check_drive_status()
            return drive_status.get('mounted', False)
        except Exception as e:
            self.logger.error(f"Drive verification failed: {str(e)}")
            return False
    
    def _verify_folders(self) -> bool:
        """Verify folder creation menggunakan status checker."""
        try:
            folders_status = self.status_checker.check_folders_status()
            return folders_status.get('all_exist', False)
        except Exception as e:
            self.logger.error(f"Folder verification failed: {str(e)}")
            return False
    
    def _verify_configs(self) -> bool:
        """Verify config sync menggunakan status checker."""
        try:
            configs_status = self.status_checker.check_configs_status()
            return configs_status.get('all_synced', False)
        except Exception as e:
            self.logger.error(f"Config verification failed: {str(e)}")
            return False
    
    def _verify_setup(self) -> Dict[str, Any]:
        """Verify current setup."""
        try:
            return self._stage_verify()
        except Exception as e:
            return {
                'status': False,
                'message': f'Verification failed: {str(e)}'
            }
    
    def _reset_setup(self) -> Dict[str, Any]:
        """Reset setup state."""
        try:
            self.logger.info("🔄 Resetting setup...")
            
            # Reset state
            self._current_stage = SetupStage.INIT
            self._current_phase = SetupPhase.INIT
            self._setup_in_progress = False
            self._setup_progress = 0.0
            self._last_error = None
            self._retry_count = 0
            
            # Reset summary
            self._last_summary = self._create_initial_summary()
            
            # Update UI
            if 'progress_tracker' in self._ui_components:
                self._ui_components['progress_tracker'].update(0.0, "Reset completed")
            
            return {
                'status': True,
                'message': 'Setup reset completed'
            }
            
        except Exception as e:
            return {
                'status': False,
                'message': f'Reset failed: {str(e)}'
            }
    
    def _update_ui_summary(self):
        """Update UI summary component dengan container-aware access."""
        try:
            # Try direct access first
            if 'setup_summary' in self._ui_components:
                summary_widget = self._ui_components['setup_summary']
                self._set_summary_content(summary_widget)
                return
            
            # Try container-based access
            container_keys = ['summary_container', 'main_container', 'form_container']
            for container_key in container_keys:
                if container_key in self._ui_components:
                    container = self._ui_components[container_key]
                    # Check for summary widget in container
                    if hasattr(container, 'setup_summary'):
                        self._set_summary_content(container.setup_summary)
                        return
                    elif hasattr(container, 'content') and hasattr(container.content, 'setup_summary'):
                        self._set_summary_content(container.content.setup_summary)
                        return
                    # Check container children
                    elif hasattr(container, 'children'):
                        for child in container.children:
                            if hasattr(child, 'setup_summary'):
                                self._set_summary_content(child.setup_summary)
                                return
            
            # Fallback: log the summary
            self.logger.info("📋 Setup Summary updated (UI widget not found)")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update UI summary: {str(e)}")
    
    def _set_summary_content(self, summary_widget):
        """Set content pada summary widget."""
        try:
            # Format summary untuk display
            summary_text = f"""
            📊 Setup Summary:
            Status: {self._last_summary.get('status', 'Unknown')}
            Progress: {self._last_summary.get('progress', 0):.1f}%
            Current Stage: {self._last_summary.get('current_stage', 'Unknown')}
            Drive Mounted: {'✅' if self._last_summary.get('drive_mounted', False) else '❌'}
            Folders Created: {self._last_summary.get('folders_created', 0)}
            Symlinks Created: {self._last_summary.get('symlinks_created', 0)}
            Configs Synced: {self._last_summary.get('configs_synced', 0)}
            """
            
            if hasattr(summary_widget, 'value'):
                summary_widget.value = summary_text
            elif hasattr(summary_widget, 'set_content'):
                summary_widget.set_content(summary_text)
            else:
                self.logger.warning("Summary widget has no supported content method")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to set summary content: {str(e)}")
    
    def get_setup_status(self) -> Dict[str, Any]:
        """Get current setup status.
        
        Returns:
            Dictionary berisi setup status
        """
        return {
            'current_stage': self._current_stage.name,
            'current_phase': self._current_phase.value,
            'progress': self._setup_progress,
            'in_progress': self._setup_in_progress,
            'summary': self._last_summary
        }
    
    def get_summary(self) -> SetupSummary:
        """Get setup summary.
        
        Returns:
            SetupSummary dictionary
        """
        return self._last_summary.copy()
    
    def should_sync_config_templates(self) -> bool:
        """Check if should sync config templates.
        
        Returns:
            True jika should sync
        """
        return (
            self._last_summary.get('drive_mounted', False) and
            self._last_summary.get('status') == 'success'
        )
    
    def sync_config_templates(
        self,
        force_overwrite: bool = False,
        update_ui: bool = True,
        ui_components: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Sync config templates.
        
        Args:
            force_overwrite: Force overwrite existing templates
            update_ui: Update UI components
            ui_components: UI components untuk update
            
        Returns:
            Dictionary berisi hasil sync
        """
        try:
            self.logger.info("📋 Syncing config templates...")
            
            # Use provided UI components atau fallback ke internal
            ui_comps = ui_components or self._ui_components
            
            # Delegate ke config operation
            result = self._config_operation.sync_templates(
                force_overwrite=force_overwrite
            )
            
            if update_ui and 'setup_summary' in ui_comps:
                self._update_ui_summary()
            
            return result
            
        except Exception as e:
            error_msg = f"Config template sync failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'message': error_msg,
                'error': str(e)
            }
    
    def perform_initial_status_check(self, ui_components: Dict[str, Any]):
        """Perform initial status check dan update UI.
        
        Args:
            ui_components: UI components untuk update
        """
        try:
            self.logger.info("🔍 Performing initial status check...")
            
            # Set UI components
            self.set_ui_components(ui_components)
            
            # Get comprehensive status
            all_status = self.status_checker.check_all_status()
            overall = all_status['overall_status']
            
            # Update summary dengan status info
            self._update_summary(
                drive_mounted=all_status['drive_status'].get('mounted', False),
                status=overall['status'],
                message=overall['message']
            )
            
            # Update UI
            self.update_summary(self.status_checker.get_status_summary())
            
            self.logger.info("✅ Initial status check completed")
            
        except Exception as e:
            self.logger.error(f"Initial status check failed: {str(e)}")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information menggunakan status checker.
        
        Returns:
            Dictionary berisi environment info
        """
        try:
            all_status = self.status_checker.check_all_status()
            return {
                'status_details': all_status,
                'drive_mounted': all_status['drive_status'].get('mounted', False),
                'folders_ok': all_status['folders_status'].get('all_exist', False),
                'configs_ok': all_status['configs_status'].get('all_synced', False),
                'setup_status': self.get_setup_status(),
                'summary': self.get_summary(),
                'readiness': all_status['overall_status']['readiness_percent']
            }
        except Exception as e:
            self.logger.error(f"Failed to get environment info: {str(e)}")
            return {
                'error': str(e),
                'drive_mounted': False,
                'folders_ok': False,
                'configs_ok': False,
                'readiness': 0
            }
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate environment.
        
        Returns:
            Dictionary berisi validation results
        """
        try:
            return self._verify_setup()
        except Exception as e:
            return {
                'status': False,
                'message': f'Environment validation failed: {str(e)}'
            }
    
    def reset_environment(self) -> Dict[str, Any]:
        """Reset environment.
        
        Returns:
            Dictionary berisi reset results
        """
        try:
            return self._reset_setup()
        except Exception as e:
            return {
                'status': False,
                'message': f'Environment reset failed: {str(e)}'
            }