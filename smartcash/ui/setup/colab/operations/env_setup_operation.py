"""
File: smartcash/ui/setup/colab/operations/env_setup_operation.py
Description: Set up environment variables and Python path
"""

import os
from typing import Dict, Any, Optional, Callable
from smartcash.ui.components.operation_container import OperationContainer
from .base_colab_operation import BaseColabOperation
from smartcash.common.constants.paths import COLAB_DATA_ROOT


class EnvSetupOperation(BaseColabOperation):
    """Set up environment variables and Python path."""
    
    def __init__(self, operation_name: str, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None, **kwargs):
        """Initialize environment setup operation.
        
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
            'setup_environment': self.execute_setup_environment
        }
    
    def execute_setup_environment(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Set up environment variables and Python path.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with operation results
        """
        def execute_operation():
            progress_steps = self.get_progress_steps('env_setup')
            
            # Step 1: Check environment configuration
            self.update_progress_safe(
                progress_callback, 
                progress_steps[0]['progress'], 
                progress_steps[0]['message'],
                progress_steps[0].get('phase_progress', 0)
            )
            
            env_vars_set = []
            env_config = self.config.get('environment', {})
            
            # Step 2: Set environment variables based on type
            self.update_progress_safe(
                progress_callback, 
                progress_steps[1]['progress'], 
                progress_steps[1]['message'],
                progress_steps[1].get('phase_progress', 0)
            )
            
            if env_config.get('type') == 'colab':
                # Set project root
                smartcash_root = '/content/smartcash'
                if smartcash_root not in os.environ.get('PYTHONPATH', ''):
                    current_path = os.environ.get('PYTHONPATH', '')
                    new_path = f"{smartcash_root}:{current_path}" if current_path else smartcash_root
                    os.environ['PYTHONPATH'] = new_path
                    env_vars_set.append('PYTHONPATH')
                    self.log(f"✅ PYTHONPATH updated: {smartcash_root}", 'info')
                
                os.environ['SMARTCASH_ROOT'] = smartcash_root
                os.environ['SMARTCASH_ENV'] = 'colab'
                os.environ['SMARTCASH_DATA_ROOT'] = COLAB_DATA_ROOT
                env_vars_set.extend(['SMARTCASH_ROOT', 'SMARTCASH_ENV', 'SMARTCASH_DATA_ROOT'])
                
                # GPU setup if enabled
                gpu_result = self._setup_gpu(env_config)
                if gpu_result['gpu_configured']:
                    env_vars_set.extend(gpu_result['env_vars'])
                
                # Set additional environment variables for Colab
                additional_vars = self._set_additional_colab_vars()
                env_vars_set.extend(additional_vars)
            
            # Step 3: Verify environment setup
            self.update_progress_safe(
                progress_callback, 
                progress_steps[2]['progress'], 
                progress_steps[2]['message'],
                progress_steps[2].get('phase_progress', 0)
            )
            
            verification = self.verify_environment()
            
            # Step 4: Complete
            self.update_progress_safe(
                progress_callback, 
                progress_steps[3]['progress'], 
                progress_steps[3]['message'],
                progress_steps[3].get('phase_progress', 0)
            )
            
            return self.create_success_result(
                f'Set {len(env_vars_set)} environment variables',
                env_vars_set=env_vars_set,
                env_vars_detail=self._get_env_vars_detail(env_vars_set),
                verification=verification
            )
            
        return self.execute_with_error_handling(execute_operation)
    
    def _setup_gpu(self, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up GPU configuration if enabled.
        
        Args:
            env_config: Environment configuration
            
        Returns:
            Dictionary with GPU setup results
        """
        gpu_result = {'gpu_configured': False, 'env_vars': [], 'gpu_info': None}
        
        if env_config.get('gpu_enabled', False):
            try:
                import torch
                if torch.cuda.is_available():
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    gpu_result['env_vars'].append('CUDA_VISIBLE_DEVICES')
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_result['gpu_info'] = {
                        'name': gpu_name,
                        'count': torch.cuda.device_count(),
                        'memory': torch.cuda.get_device_properties(0).total_memory
                    }
                    gpu_result['gpu_configured'] = True
                    self.log(f"✅ GPU enabled: {gpu_name}", 'info')
                else:
                    self.log("⚠️ GPU requested but not available", 'warning')
            except ImportError:
                self.log("⚠️ PyTorch not available for GPU setup", 'warning')
        
        return gpu_result
    
    def _set_additional_colab_vars(self) -> list:
        """Set additional environment variables for Colab.
        
        Returns:
            List of environment variable names that were set
        """
        additional_vars = []
        
        # Set locale variables
        if 'LC_ALL' not in os.environ:
            os.environ['LC_ALL'] = 'C.UTF-8'
            additional_vars.append('LC_ALL')
        
        if 'LANG' not in os.environ:
            os.environ['LANG'] = 'C.UTF-8'
            additional_vars.append('LANG')
        
        # Set optimization flags
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        additional_vars.extend(['OMP_NUM_THREADS', 'MKL_NUM_THREADS'])
        
        # Set display for matplotlib
        os.environ['MPLBACKEND'] = 'Agg'
        additional_vars.append('MPLBACKEND')
        
        return additional_vars
    
    def _get_env_vars_detail(self, env_vars: list) -> Dict[str, str]:
        """Get detailed information about set environment variables.
        
        Args:
            env_vars: List of environment variable names
            
        Returns:
            Dictionary mapping variable names to their values
        """
        detail = {}
        for var in env_vars:
            detail[var] = os.environ.get(var, 'Not set')
        return detail
    
    def verify_environment(self) -> Dict[str, Any]:
        """Verify that environment variables are properly set.
        
        Returns:
            Dictionary with verification results
        """
        required_vars = ['SMARTCASH_ROOT', 'SMARTCASH_ENV', 'SMARTCASH_DATA_ROOT']
        return self.verify_environment_variables(required_vars)