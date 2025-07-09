"""
File: smartcash/ui/setup/colab/operations/operation_manager.py
Description: Manager for coordinating individual colab operations
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.ui.components.operation_container import OperationContainer

from .init_operation import InitOperation
from .drive_mount_operation import DriveMountOperation
from .symlink_operation import SymlinkOperation
from .folders_operation import FoldersOperation
from .config_sync_operation import ConfigSyncOperation
from .env_setup_operation import EnvSetupOperation
from .verify_operation import VerifyOperation

from ..constants import SetupStage, STAGE_WEIGHTS


class ColabOperationManager(OperationHandler):
    """Manager for coordinating individual colab operations."""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 operation_container: Optional[OperationContainer] = None,
                 max_workers: int = 2):
        """Initialize Colab operation manager.
        
        Args:
            config: Configuration dictionary
            operation_container: OperationContainer for UI integration
            max_workers: Maximum number of worker threads
        """
        super().__init__(
            module_name='colab_operation_manager',
            parent_module='setup',
            max_workers=max_workers,
            use_process_pool=False,
            operation_container=operation_container
        )
        
        self.config = config
        self.setup_stages = [
            'init',
            'drive', 
            'symlink',
            'folders',
            'config',
            'env',
            'verify'
        ]
        self.current_stage = 0
        
        # Initialize individual operation handlers
        self.operations = {
            'init': InitOperation(config, operation_container=operation_container),
            'drive': DriveMountOperation(config, operation_container=operation_container),
            'symlink': SymlinkOperation(config, operation_container=operation_container),
            'folders': FoldersOperation(config, operation_container=operation_container),
            'config': ConfigSyncOperation(config, operation_container=operation_container),
            'env': EnvSetupOperation(config, operation_container=operation_container),
            'verify': VerifyOperation(config, operation_container=operation_container)
        }
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations for Colab setup."""
        return {
            'init': self._init_operation,
            'drive': self._drive_mount_operation,
            'symlink': self._symlink_operation,
            'folders': self._folders_operation,
            'config': self._config_sync_operation,
            'env': self._env_setup_operation,
            'verify': self._verify_operation,
            'full_setup': self._full_setup_operation,
            'post_init_check': self._post_init_check
        }
    
    def initialize(self) -> None:
        """Initialize the operation manager and its operations."""
        self.logger.info("🚀 Initializing Colab operation manager")
        
        # Initialize all individual operations
        for operation_name, operation in self.operations.items():
            try:
                operation.initialize()
                self.logger.debug(f"✅ Initialized {operation_name} operation")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {operation_name} operation: {e}")
        
        self.logger.info("✅ Colab operation manager initialization complete")
    
    def clear_outputs(self) -> None:
        """Clear outputs from the operation container if available."""
        try:
            if hasattr(self, 'operation_container') and self.operation_container is not None:
                if hasattr(self.operation_container, 'clear_logs'):
                    self.operation_container.clear_logs()
                if hasattr(self.operation_container, 'clear_dialog'):
                    self.operation_container.clear_dialog()
                self.logger.debug("Cleared operation container outputs")
        except Exception as e:
            self.logger.error(f"Error clearing outputs: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("🧹 Cleaning up Colab operation manager")
        self.clear_outputs()
        
        for operation in self.operations.values():
            try:
                operation.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up operation {operation.__class__.__name__}: {e}")
        
        self._executor.shutdown(wait=False)
        self.logger.info("✅ Colab operation manager cleanup complete")
        
    def update_status(self, message: str, level: str = 'info') -> None:
        """Update operation status with a message.
        
        Args:
            message: Status message to display
            level: Message level ('info', 'warning', 'error', 'success')
        """
        # Log the message
        log_method = getattr(self.logger, level, self.logger.info)
        log_method(f"Status update: {message}")
        
        # Update operation container if available
        if hasattr(self, 'operation_container') and self.operation_container is not None:
            if hasattr(self.operation_container, 'update_status'):
                self.operation_container.update_status(message, level=level)
            elif hasattr(self.operation_container, 'log'):
                self.operation_container.log(message, level=level)
    
    def _init_operation(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute initialization operation."""
        return self.operations['init'].execute_init(progress_callback)
    
    def _drive_mount_operation(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute drive mount operation."""
        return self.operations['drive'].execute_mount_drive(progress_callback)
    
    def _symlink_operation(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute symlink creation operation."""
        return self.operations['symlink'].execute_create_symlinks(progress_callback)
    
    def _folders_operation(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute folder creation operation."""
        return self.operations['folders'].execute_create_folders(progress_callback)
    
    def _config_sync_operation(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute config sync operation."""
        return self.operations['config'].execute_sync_configs(progress_callback)
    
    def _env_setup_operation(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute environment setup operation."""
        return self.operations['env'].execute_setup_environment(progress_callback)
    
    def _verify_operation(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute verification operation."""
        return self.operations['verify'].execute_verify_setup(progress_callback)
    
    def _full_setup_operation(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute complete environment setup with weighted progress tracking."""
        try:
            self.log("🚀 Starting complete environment setup", 'info')
            
            stage_results = {}
            cumulative_progress = 0
            
            # Execute each stage with weighted progress
            for stage_name in self.setup_stages:
                stage_weight = STAGE_WEIGHTS.get(getattr(SetupStage, stage_name.upper()), 0)
                
                self.log(f"Executing stage: {stage_name} (weight: {stage_weight}%)", 'info')
                
                # Create stage-specific progress callback
                def stage_progress_callback(stage_progress, message):
                    nonlocal cumulative_progress
                    # Convert stage progress to overall weighted progress
                    weighted_progress = cumulative_progress + (stage_progress / 100) * stage_weight
                    if progress_callback:
                        progress_callback(weighted_progress, f"{stage_name.upper()}: {message}")
                
                # Execute stage operation
                stage_operation = self.get_operations().get(stage_name)
                if stage_operation:
                    result = stage_operation(progress_callback=stage_progress_callback)
                    stage_results[stage_name] = result
                    
                    if not result.get('success', False):
                        self.log(f"❌ Stage {stage_name} failed: {result.get('error', 'Unknown error')}", 'error')
                        return {
                            'success': False,
                            'failed_stage': stage_name,
                            'stage_results': stage_results,
                            'error': f"Setup failed at stage '{stage_name}': {result.get('error', 'Unknown error')}"
                        }
                    else:
                        self.log(f"✅ Stage {stage_name} completed successfully", 'info')
                        
                else:
                    self.log(f"⚠️ Stage operation '{stage_name}' not found", 'warning')
                
                # Update cumulative progress
                cumulative_progress += stage_weight
            
            if progress_callback:
                progress_callback(100, "🎉 Environment setup completed!")
            
            self.log("🎉 Complete environment setup finished successfully", 'info')
            
            return {
                'success': True,
                'stage_results': stage_results,
                'message': 'Environment setup completed successfully'
            }
            
        except Exception as e:
            error_msg = f'Full setup failed: {str(e)}'
            self.log(error_msg, 'error')
            return {
                'success': False,
                'error': error_msg
            }
    
    def _post_init_check(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Post-initialization check for integrity and auto-sync."""
        try:
            if progress_callback:
                progress_callback(20, "🔍 Checking folders and configs integrity...")
            
            # Check folder integrity using folders operation
            folder_verification = self.operations['folders'].verify_folders()
            missing_folders = folder_verification.get('missing_folders', [])
            
            # Check symlink integrity - would need verification method
            # TODO: Add symlink verification method to symlink operation
            broken_symlinks = []
            
            # Check config integrity using config operation
            config_verification = self.operations['config'].check_config_integrity()
            missing_configs = config_verification.get('missing_configs', [])
            
            if progress_callback:
                progress_callback(60, "⚙️ Auto-syncing configurations...")
            
            # Auto-sync configs if needed
            config_sync_result = None
            if missing_folders or broken_symlinks or missing_configs:
                self.log("Detected integrity issues, performing auto-sync", 'info')
                config_sync_result = self.operations['config'].execute_sync_configs()
            
            if progress_callback:
                progress_callback(100, "✅ Post-init check complete")
            
            return {
                'success': True,
                'missing_folders': missing_folders,
                'broken_symlinks': broken_symlinks,
                'missing_configs': missing_configs,
                'config_sync': config_sync_result,
                'message': f'Found {len(missing_folders)} missing folders, {len(broken_symlinks)} broken symlinks, {len(missing_configs)} missing configs'
            }
            
        except Exception as e:
            self.log(f"Post-init check failed: {str(e)}", 'error')
            return {
                'success': False,
                'error': f'Post-init check failed: {str(e)}'
            }
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration for all operations.
        
        Args:
            new_config: New configuration to apply
        """
        self.config = new_config
        
        # Update config for all individual operations
        for operation in self.operations.values():
            operation.config = new_config
    
    def get_stage_status(self) -> Dict[str, Any]:
        """Get current stage status information.
        
        Returns:
            Dictionary with stage status information
        """
        return {
            'current_stage': self.current_stage,
            'current_stage_name': self.setup_stages[self.current_stage] if self.current_stage < len(self.setup_stages) else 'complete',
            'total_stages': len(self.setup_stages),
            'completed_stages': self.current_stage,
            'remaining_stages': len(self.setup_stages) - self.current_stage,
            'progress_percent': (self.current_stage / len(self.setup_stages)) * 100
        }