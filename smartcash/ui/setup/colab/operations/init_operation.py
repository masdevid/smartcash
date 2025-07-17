"""
File: smartcash/ui/setup/colab/operations/init_operation.py
Description: Initialize environment setup with detection and validation
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.components.operation_container import OperationContainer
from .base_colab_operation import BaseColabOperation


class InitOperation(BaseColabOperation):
    """Initialize environment setup with detection and validation."""
    
    def __init__(self, operation_name: str, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None, **kwargs):
        """Initialize init operation.
        
        Args:
            operation_name: Name of the operation
            config: Configuration dictionary
            operation_container: Optional operation container for UI integration
            **kwargs: Additional arguments
        """
        super().__init__(operation_name, config, operation_container, **kwargs)
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'init': self.execute_init
        }
    
    def execute_init(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Initialize environment setup with detection and validation.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with operation results
        """
        def _execute_init_internal() -> Dict[str, Any]:
            # Get progress steps using base class method
            steps = self.get_progress_steps('init')
            
            # Step 1: Detect environment
            self.update_progress_safe(progress_callback, steps[0]['progress'], steps[0]['message'])
            
            env_info = self.detect_environment_enhanced()
            env_type = env_info.get('runtime', {}).get('type', 'local')
            
            # Update config with detected environment
            if 'environment' not in self.config:
                self.config['environment'] = {}
            self.config['environment']['type'] = env_type
            
            self.log(f"Environment detected: {env_type}", 'info')
            
            # Step 2: Environment detected
            self.update_progress_safe(progress_callback, steps[1]['progress'], 
                                    f"✅ Environment: {env_type}")
            
            # Step 3: Check system requirements
            self.update_progress_safe(progress_callback, steps[2]['progress'], steps[2]['message'])
            
            system_info = self.format_system_info(env_info)
            self.log(f"System: {system_info['os_display']}", 'info')
            self.log(f"RAM: {system_info['ram_gb']:.1f}GB available", 'info')
            
            # Step 4: Validate configuration
            self.update_progress_safe(progress_callback, steps[3]['progress'], steps[3]['message'])
            
            validation_result = self.validate_colab_environment(self.config)
            if not validation_result['valid']:
                return self.create_error_result(
                    f"Configuration validation failed: {validation_result['issues']}"
                )
            
            # Step 5: Complete
            self.update_progress_safe(progress_callback, steps[4]['progress'], steps[4]['message'])
            
            return self.create_success_result(
                f'Environment initialized as {env_type}',
                environment=env_type,
                system_info=system_info,
                env_info=env_info,
                validation=validation_result
            )
        
        return self.execute_with_error_handling(_execute_init_internal)
    
