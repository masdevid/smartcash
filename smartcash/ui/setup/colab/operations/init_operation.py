"""
File: smartcash/ui/setup/colab/operations/init_operation.py
Description: Initialize environment setup with detection and validation
"""

import os
from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from ..utils.env_detector import detect_environment_info


class InitOperation(OperationHandler):
    """Initialize environment setup with detection and validation."""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """Initialize init operation.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments
        """
        super().__init__(
            module_name='init_operation',
            parent_module='colab',
            **kwargs
        )
        self.config = config
    
    def initialize(self) -> None:
        """Initialize the init operation."""
        self.logger.info("🚀 Initializing init operation")
        # No specific initialization needed for init operation
        self.logger.info("✅ Init operation initialization complete")
    
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
        try:
            if progress_callback:
                progress_callback(10, "🔍 Detecting runtime environment...")
            
            # Detect environment using enhanced env_detector
            env_info = detect_environment_info()
            env_type = env_info.get('runtime', {}).get('type', 'local')
            
            # Update config with detected environment
            if 'environment' not in self.config:
                self.config['environment'] = {}
            self.config['environment']['type'] = env_type
            
            self.log(f"Environment detected: {env_type}", 'info')
            
            if progress_callback:
                progress_callback(30, f"✅ Environment: {env_type}")
            
            # Validate system requirements
            if progress_callback:
                progress_callback(50, "🔧 Checking system requirements...")
            
            system_info = self._get_system_info(env_info)
            self.log(f"System: {system_info['os_display']}", 'info')
            self.log(f"RAM: {system_info['ram_gb']:.1f}GB available", 'info')
            
            if progress_callback:
                progress_callback(80, "🔍 Validating configuration...")
            
            # Validate config
            validation_result = self._validate_config()
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': f"Configuration validation failed: {validation_result['issues']}"
                }
            
            if progress_callback:
                progress_callback(100, "✅ Initialization complete")
            
            return {
                'success': True,
                'environment': env_type,
                'system_info': system_info,
                'env_info': env_info,
                'validation': validation_result,
                'message': f'Environment initialized as {env_type}'
            }
            
        except Exception as e:
            self.log(f"Initialization failed: {str(e)}", 'error')
            return {
                'success': False,
                'error': f'Initialization failed: {str(e)}'
            }
    
    def _get_system_info(self, env_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get enhanced system information from env_detector data.
        
        Args:
            env_info: Environment information from detect_environment_info()
            
        Returns:
            Dictionary with system information
        """
        os_info = env_info.get('os', {})
        ram_bytes = env_info.get('total_ram', 0)
        
        return {
            'os': os_info.get('system', 'Unknown'),
            'release': os_info.get('release', 'Unknown'),
            'machine': os_info.get('machine', 'Unknown'),
            'os_display': f"{os_info.get('system', 'Unknown')} {os_info.get('release', '')}".strip(),
            'ram_gb': ram_bytes / (1024**3) if ram_bytes > 0 else 0,
            'cpu_cores': env_info.get('cpu_cores', 'Unknown'),
            'gpu_available': env_info.get('gpu', 'No GPU available') != 'No GPU available',
            'gpu_name': env_info.get('gpu', 'None'),
            'is_colab': env_info.get('is_colab', False),
            'drive_mounted': env_info.get('drive_mounted', False),
            'drive_mount_path': env_info.get('drive_mount_path', '')
        }
    
    def _validate_config(self) -> Dict[str, Any]:
        """Validate the current configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation = {'valid': True, 'issues': []}
        
        # Validate environment config
        env_config = self.config.get('environment', {})
        if not env_config:
            validation['issues'].append('Missing environment configuration')
            validation['valid'] = False
        
        if 'type' not in env_config:
            validation['issues'].append('Environment type not specified')
            validation['valid'] = False
        
        # Validate environment-specific requirements
        env_type = env_config.get('type', 'local')
        if env_type == 'colab':
            # Check if we're actually in Colab
            try:
                import google.colab
            except ImportError:
                validation['issues'].append('Configuration set to Colab but not running in Colab environment')
                validation['valid'] = False
        
        return validation