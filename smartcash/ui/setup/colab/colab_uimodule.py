"""
Colab Module (Optimized) - BaseUIModule Pattern with Mixins
Google Colab environment detection and setup with sequential operations.
"""

from typing import Dict, Any
import os
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.model.mixins import ModelConfigSyncMixin, BackendServiceMixin
from smartcash.common.environment import get_environment_manager
from .components.colab_ui import create_colab_ui
from .configs.colab_config_handler import ColabConfigHandler
from .configs.colab_defaults import get_default_colab_config
from .operations.colab_factory import (init_environment, mount_drive, create_symlinks, 
                                      create_folders, sync_config, setup_environment, verify_setup, 
                                      execute_full_setup, detect_environment)


class ColabUIModule(ModelConfigSyncMixin, BackendServiceMixin, BaseUIModule):
    """Optimized Colab Module with mixin integration."""
    
    def __init__(self, **kwargs):
        super().__init__(module_name='colab', parent_module='setup', enable_environment=True, **kwargs)
        self._required_components = ['main_container', 'action_container', 'operation_container', 'environment_container']
        self._initialized = False
        self._ui_components = None
        self._ui_components_created = False
        self._operation_sequence = ['init', 'drive', 'symlink', 'folders', 'config', 'env', 'verify']
        self._operation_status = {op: 'pending' for op in self._operation_sequence}
        self._current_operation_index = 0
        self._is_colab_environment = self._detect_colab_environment()
        self.environment_manager = None  # Will be initialized when needed
    
    def get_default_config(self) -> Dict[str, Any]: return get_default_colab_config()
    def create_config_handler(self, config: Dict[str, Any]) -> ColabConfigHandler: return ColabConfigHandler(config)
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components with lazy initialization."""
        # Prevent double initialization
        if self._ui_components_created and self._ui_components is not None:
            self.log_debug("⏭️ Skipping UI component creation - already created")
            return self._ui_components
            
        try:
            self.log_debug("Creating Colab UI components...")
            self._ui_components = create_colab_ui(config=config)
            
            if not self._ui_components:
                raise RuntimeError("Failed to create UI components")
            
            # Mark as created to prevent reinitalization
            self._ui_components_created = True
            return self._ui_components
            
        except Exception as e:
            self.log_error(f"Failed to create UI components: {e}")
            raise
    
    def _detect_colab_environment(self) -> bool:
        """Detect if running in Google Colab environment."""
        try:
            import google.colab  # noqa: F401
            return True
        except ImportError:
            return 'COLAB_GPU' in os.environ or '/content' in os.getcwd()
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get colab-specific button handlers."""
        # Get base handlers first (save, reset)
        handlers = super()._get_module_button_handlers()
        
        # Add colab-specific handlers
        colab_handlers = {
            'colab_setup': self._operation_run_full_setup,  # Map colab_setup button to full setup
            'run_full_setup': self._operation_run_full_setup,
            'init_environment': self._operation_init_environment,
            'mount_drive': self._operation_mount_drive,
            'create_symlinks': self._operation_create_symlinks,
            'create_folders': self._operation_create_folders,
            'sync_config': self._operation_sync_config,
            'setup_environment': self._operation_setup_environment,
            'verify_setup': self._operation_verify_setup,
            'detect_environment': self._operation_detect_environment,
            'reset_setup': self._operation_reset_setup
        }
        
        handlers.update(colab_handlers)
        return handlers
    
    def _operation_run_full_setup(self, button=None) -> Dict[str, Any]:
        """Execute complete Colab setup sequence."""
        return self._execute_operation_with_wrapper(
            operation_name="Full Setup",
            operation_func=lambda: self._execute_full_setup_operation(),
            button=button,
            validation_func=lambda: self._validate_colab_environment(),
            success_message="Full setup completed successfully",
            error_message="Full setup failed"
        )
    
    def _operation_init_environment(self, button=None) -> Dict[str, Any]:
        """Initialize Colab environment."""
        return self._execute_single_operation('init', button)
    
    def _operation_mount_drive(self, button=None) -> Dict[str, Any]:
        """Mount Google Drive."""
        return self._execute_single_operation('drive', button)
    
    def _operation_create_symlinks(self, button=None) -> Dict[str, Any]:
        """Create symbolic links."""
        return self._execute_single_operation('symlink', button)
    
    def _operation_create_folders(self, button=None) -> Dict[str, Any]:
        """Create necessary folders."""
        return self._execute_single_operation('folders', button)
    
    def _operation_sync_config(self, button=None) -> Dict[str, Any]:
        """Sync configuration."""
        return self._execute_single_operation('config', button)
    
    def _operation_setup_environment(self, button=None) -> Dict[str, Any]:
        """Setup Python environment."""
        return self._execute_single_operation('env', button)
    
    def _operation_verify_setup(self, button=None) -> Dict[str, Any]:
        """Verify setup completion."""
        return self._execute_single_operation('verify', button)
    
    def _operation_detect_environment(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Detect current environment."""
        try:
            env_info = detect_environment()
            self._update_environment_info(env_info)
            return {'success': True, 'message': 'Environment detected', 'env_info': env_info}
        except Exception as e:
            return {'success': False, 'message': f'Detection failed: {e}'}
    
    def _operation_reset_setup(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Reset setup status."""
        try:
            self._operation_status = {op: 'pending' for op in self._operation_sequence}
            self._current_operation_index = 0
            self._update_progress_display()
            self.log_info("Setup status reset")
            return {'success': True, 'message': 'Setup status reset'}
        except Exception as e:
            return {'success': False, 'message': f'Reset failed: {e}'}
    
    def _execute_single_operation(self, operation_type: str, button=None) -> Dict[str, Any]:
        """Execute a single colab operation."""
        operation_names = {
            'init': 'Environment Initialization', 'drive': 'Drive Mount', 'symlink': 'Symlink Creation',
            'folders': 'Folder Creation', 'config': 'Config Sync', 'env': 'Environment Setup', 'verify': 'Setup Verification'
        }
        
        return self._execute_operation_with_wrapper(
            operation_name=operation_names.get(operation_type, operation_type.title()),
            operation_func=lambda: self._execute_colab_operation(operation_type),
            button=button,
            validation_func=lambda: self._validate_operation_prerequisites(operation_type),
            success_message=f"{operation_names.get(operation_type, operation_type)} completed",
            error_message=f"{operation_names.get(operation_type, operation_type)} failed"
        )
    
    def _execute_full_setup_operation(self) -> Dict[str, Any]:
        """Execute complete setup sequence."""
        try:
            self.log_info("🚀 Starting full Colab setup sequence...")
            config = self.get_current_config()
            
            result = execute_full_setup(
                config=config,
                operation_container=self.get_component('operation_container'),
                progress_callback=self._handle_progress_update
            )
            
            if result.get('success'):
                self._operation_status = {op: 'completed' for op in self._operation_sequence}
                self._current_operation_index = len(self._operation_sequence)
                self._update_progress_display()
                
            return result
        except Exception as e:
            return {'success': False, 'message': f'Full setup failed: {e}'}
    
    def _execute_colab_operation(self, operation_type: str) -> Dict[str, Any]:
        """Execute specific colab operation."""
        try:
            config = self.get_current_config()
            operation_container = self.get_component('operation_container')
            
            operation_functions = {
                'init': lambda: init_environment(config, operation_container),
                'drive': lambda: mount_drive(config, operation_container),
                'symlink': lambda: create_symlinks(config, operation_container),
                'folders': lambda: create_folders(config, operation_container),
                'config': lambda: sync_config(config, operation_container),
                'env': lambda: setup_environment(config, operation_container),
                'verify': lambda: verify_setup(config, operation_container)
            }
            
            if operation_type in operation_functions:
                result = operation_functions[operation_type]()
                
                if result.get('success'):
                    self._operation_status[operation_type] = 'completed'
                    if operation_type in self._operation_sequence:
                        index = self._operation_sequence.index(operation_type)
                        self._current_operation_index = max(self._current_operation_index, index + 1)
                    self._update_progress_display()
                
                return result
            else:
                return {'success': False, 'message': f'Unknown operation: {operation_type}'}
                
        except Exception as e:
            return {'success': False, 'message': f'Operation {operation_type} failed: {e}'}
    
    def _validate_colab_environment(self) -> Dict[str, Any]:
        """Validate Colab environment prerequisites."""
        if not self._is_colab_environment:
            return {'valid': False, 'message': 'Not running in Google Colab environment'}
        return {'valid': True}
    
    def _validate_operation_prerequisites(self, operation_type: str) -> Dict[str, Any]:
        """Validate prerequisites for specific operation."""
        try:
            # Check environment
            if not self._is_colab_environment and operation_type == 'drive':
                return {'valid': False, 'message': 'Drive mount only available in Colab'}
            
            # Check sequence dependencies
            if operation_type in self._operation_sequence:
                index = self._operation_sequence.index(operation_type)
                if index > 0:
                    prev_op = self._operation_sequence[index - 1]
                    if self._operation_status.get(prev_op) != 'completed':
                        return {'valid': False, 'message': f'Previous operation {prev_op} must be completed first'}
            
            return {'valid': True}
        except Exception:
            return {'valid': False, 'message': 'Prerequisites validation failed'}
    
    def _handle_progress_update(self, progress: int, message: str = "") -> None:
        """Handle progress updates from operations."""
        try:
            if hasattr(self, 'update_progress'):
                self.update_progress(progress=progress, message=message)
            self._update_progress_display()
        except Exception as e:
            self.log_warning(f"Progress update failed: {e}")
    
    def _update_progress_display(self) -> None:
        """Update progress display in UI."""
        try:
            progress_percent = (self._current_operation_index / len(self._operation_sequence)) * 100
            completed_ops = [op for op, status in self._operation_status.items() if status == 'completed']
            
            # Update footer with progress info
            footer_container = self.get_component('footer_container')
            if footer_container and hasattr(footer_container, 'update_progress'):
                footer_container.update_progress({
                    'progress': progress_percent,
                    'completed_operations': len(completed_ops),
                    'total_operations': len(self._operation_sequence),
                    'current_status': self._operation_status
                })
        except Exception as e:
            self.log_error(f"Progress display update failed: {e}")
    
    def _update_environment_info(self, env_info: Dict[str, Any]) -> None:
        """Update environment information in UI."""
        try:
            # Update environment info panel if available
            env_panel = self.get_component('env_info_panel')
            if env_panel and hasattr(env_panel, 'update_info'):
                env_panel.update_info(env_info)
        except Exception as e:
            self.log_error(f"Environment info update failed: {e}")
    
    def get_setup_status(self) -> Dict[str, Any]:
        """Get current setup status."""
        completed_ops = [op for op, status in self._operation_status.items() if status == 'completed']
        return {
            'is_colab': self._is_colab_environment,
            'operations_completed': len(completed_ops),
            'total_operations': len(self._operation_sequence),
            'progress_percent': (len(completed_ops) / len(self._operation_sequence)) * 100,
            'operation_status': self._operation_status.copy(),
            'next_operation': self._operation_sequence[self._current_operation_index] if self._current_operation_index < len(self._operation_sequence) else None
        }
    
    def is_setup_complete(self) -> bool:
        """Check if setup is complete."""
        return all(status == 'completed' for status in self._operation_status.values())

    def cleanup(self) -> None:
        """Widget lifecycle cleanup - optimization.md compliance."""
        try:
            # Cleanup operation status tracking
            if hasattr(self, '_operation_status'):
                self._operation_status.clear()
            
            # Cleanup environment manager
            if hasattr(self, 'environment_manager') and self.environment_manager:
                if hasattr(self.environment_manager, 'cleanup'):
                    self.environment_manager.cleanup()
            
            # Cleanup UI components if they have cleanup methods
            if hasattr(self, '_ui_components') and self._ui_components:
                # Call component-specific cleanup if available
                if hasattr(self._ui_components, '_cleanup'):
                    self._ui_components._cleanup()
                
                # Close individual widgets
                for component_name, component in self._ui_components.items():
                    if hasattr(component, 'close'):
                        try:
                            component.close()
                        except Exception:
                            pass  # Ignore cleanup errors
            
            # Call parent cleanup
            if hasattr(super(), 'cleanup'):
                super().cleanup()
            
            # Minimal logging for cleanup completion
            if hasattr(self, 'logger'):
                self.logger.info("Colab module cleanup completed")
                
        except Exception as e:
            # Critical errors always logged
            if hasattr(self, 'logger'):
                self.logger.error(f"Colab module cleanup failed: {e}")

    def __del__(self):
        """Memory management - ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during deletion


def get_colab_uimodule(auto_initialize: bool = True) -> ColabUIModule:
    """Factory function to get ColabUIModule instance."""
    module = ColabUIModule()
    if auto_initialize:
        module.initialize()
    return module