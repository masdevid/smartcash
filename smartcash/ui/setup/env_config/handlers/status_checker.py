"""
Status checker for environment configuration.

Status Checker Module.

This module provides the StatusChecker class which verifies the environment
status and requirements for the SmartCash application.
"""

import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional, Tuple, TypedDict

from smartcash.common.environment import get_environment_manager
from smartcash.ui.handlers.base_handler import BaseHandler
from smartcash.ui.setup.env_config.handlers.base_config_mixin import BaseConfigMixin
from smartcash.ui.setup.env_config.utils.env_detector import detect_environment_info

class EnvironmentInfo(TypedDict, total=False):
    """Type definition for environment information.
    
    Attributes:
        os: Operating system name (e.g., 'Linux', 'Windows', 'Darwin')
        os_info: Detailed OS information from platform module
        python_version: Python version string (e.g., '3.9.0')
        gpu_available: Boolean indicating if GPU is available
        gpu_info: Dictionary with detailed GPU information if available
        environment_type: Type of environment ('colab' or 'local')
        runtime_info: Detailed runtime information from env_detector
        runtime_display: Formatted display string for the runtime
        is_valid: Boolean indicating if environment is valid
        total_memory_gb: Total available memory in GB
        free_disk_gb: Free disk space in GB
    """
    os: str
    os_info: Dict[str, Any]
    python_version: str
    gpu_available: bool
    gpu_info: Dict[str, Any]
    environment_type: str
    runtime_info: Dict[str, Any]
    runtime_display: str
    is_valid: bool
    total_memory_gb: float
    free_disk_gb: float

class RequirementStatus(TypedDict, total=False):
    """Type definition for requirement check status."""
    status: bool
    message: str
    required: Any
    found: Any

class StatusCheckResult(TypedDict, total=False):
    """Type definition for status check results."""
    env_info: EnvironmentInfo
    requirements_status: Dict[str, RequirementStatus]
    status_message: str
    status_type: Literal['success', 'warning', 'error', 'info']
    status: bool

class StatusChecker(BaseHandler, BaseConfigMixin):
    """Handler for checking environment status and requirements."""
    
    # Default configuration for the handler
    DEFAULT_CONFIG = {
        'required_checks': [
            'python_version', 
            'disk_space', 
            'memory',
            'gpu_available',
            'internet_connection'
        ],
        'warning_thresholds': {
            'disk_space_gb': 5.0,
            'memory_gb': 4.0,
            'python_version': '3.8.0',
            'gpu_memory_gb': 4.0
        },
        'enable_logging': True,
        'check_interval_seconds': 300
    }
    
    def __init__(self, config_handler=None, **kwargs):
        """Initialize the StatusChecker with configuration.
        
        Args:
            config_handler: Instance of ConfigHandler for configuration
            **kwargs: Additional keyword arguments for BaseHandler
        """
        super().__init__(
            module_name='status',
            parent_module='env_config',
            **kwargs
        )
        
        # Initialize BaseConfigMixin with the provided config handler
        BaseConfigMixin.__init__(self, config_handler=config_handler, **kwargs)
        
        # Initialize environment manager
        self.env_manager = get_environment_manager(logger=self.logger)
        
        # Initialize configuration with defaults
        self.required_checks = self.get_config_value('required_checks', [
            'python_version', 
            'disk_space', 
            'memory'
        ])
        
        self.warning_thresholds = self.get_config_value('warning_thresholds', {
            'disk_space_gb': 5.0,
            'memory_gb': 4.0,
            'python_version': '3.8.0',
            'gpu_memory_gb': 4.0
        })
        
        # Log initialization
        self.logger.debug(
            f"Initialized with required_checks={self.required_checks}"
        )
        
        # Store config handler reference
        self.config_handler = config_handler
        
        # Get configuration
        self.config = self.config_handler.get_handler_config('status') if config_handler else {}
        
        # Initialize last check result
        self._last_check_result: Optional[StatusCheckResult] = None
        
        self.logger.debug("Initialized StatusChecker")
    
    async def check_environment(self) -> StatusCheckResult:
        """Check the current environment status.
        
        Returns:
            StatusCheckResult with environment information and requirements status
        """
        self.set_stage("status_check", "Checking environment status")
        
        result: StatusCheckResult = {
            'env_info': self._get_environment_info(),
            'requirements_status': {},
            'status_message': 'Environment check completed',
            'status_type': 'success',
            'status': True
        }
        
        try:
            # Check system requirements
            result['requirements_status'] = await self._check_requirements()
            
            # Update overall status based on requirements
            all_passed = all(
                req.get('status', False) 
                for req in result['requirements_status'].values()
            )
            
            if not all_passed:
                result['status_type'] = 'warning'
                result['status'] = False
                result['status_message'] = 'Some requirements are not met'
            
            self._last_check_result = result
            
        except Exception as e:
            error_msg = f"Error checking environment: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            result.update({
                'status_type': 'error',
                'status': False,
                'status_message': error_msg
            })
        
        return result
    
    def _get_environment_info(self) -> EnvironmentInfo:
        """Get information about the current environment using env_detector.
        
        Returns:
            EnvironmentInfo dictionary with environment details including:
            - os: Operating system information
            - python_version: Python version
            - gpu_available: Boolean indicating if GPU is available
            - gpu_info: Detailed GPU information if available
            - environment_type: Type of environment (colab/local)
            - runtime_info: Detailed runtime information
            - is_valid: Boolean indicating if environment is valid
            - total_memory_gb: Total available memory in GB
            - free_disk_gb: Free disk space in GB
        """
        # Get environment info from env_detector
        detector_info = detect_environment_info()
        
        # Get runtime information
        runtime_info = detector_info.get('runtime', {})
        
        # Map detector info to our EnvironmentInfo format
        info: EnvironmentInfo = {
            'os': platform.system(),
            'python_version': platform.python_version(),
            'gpu_available': runtime_info.get('gpu') == 'available',
            'environment_type': runtime_info.get('type', 'local'),
            'runtime_info': runtime_info,
            'is_valid': True,
            'runtime_display': detector_info.get('runtime_display', 'Local Environment')
        }
        
        # Add GPU info if available
        gpu_info = detector_info.get('gpu')
        if gpu_info and gpu_info != 'Unknown':
            info['gpu_info'] = gpu_info if isinstance(gpu_info, dict) else {'info': str(gpu_info)}
        
        # Add memory and disk info if available
        storage_info = detector_info.get('storage_info', {})
        if isinstance(storage_info, dict):
            # Convert storage from bytes to GB if needed
            total = storage_info.get('total')
            free = storage_info.get('free')
            
            if total is not None and free is not None:
                # If values are in bytes, convert to GB
                if total > 1024:  # Assume it's in bytes if > 1KB
                    total = round(total / (1024 ** 3), 2)
                    free = round(free / (1024 ** 3), 2)
                
                info.update({
                    'total_memory_gb': total,
                    'free_disk_gb': free
                })
        
        # Add OS-specific information
        if 'os' in detector_info:
            info['os_info'] = detector_info['os']
        
        # Add Python version from detector if available
        if 'python_version' in detector_info:
            info['python_version'] = detector_info['python_version']
        
        return info
    
    async def _check_requirements(self) -> Dict[str, RequirementStatus]:
        """Check system and package requirements.
        
        Returns:
            Dictionary of requirement checks with their status
        """
        requirements = self.config.get('requirements', {})
        results: Dict[str, RequirementStatus] = {}
        
        # Check Python version
        if 'python_version' in requirements:
            required_version = requirements['python_version']
            current_version = platform.python_version()
            
            results['python_version'] = {
                'status': self._version_compare(current_version, required_version) >= 0,
                'message': f"Python {required_version} or higher required",
                'required': required_version,
                'found': current_version
            }
        
        # Check installed packages
        if 'packages' in requirements:
            for pkg, required_version in requirements['packages'].items():
                try:
                    import importlib.metadata
                    installed_version = importlib.metadata.version(pkg)
                    
                    results[f"pkg_{pkg}"] = {
                        'status': self._version_compare(installed_version, required_version) >= 0,
                        'message': f"{pkg} {required_version} or higher required",
                        'required': required_version,
                        'found': installed_version
                    }
                except importlib.metadata.PackageNotFoundError:
                    results[f"pkg_{pkg}"] = {
                        'status': False,
                        'message': f"{pkg} is not installed",
                        'required': required_version,
                        'found': None
                    }
        
        # Check environment variables
        if 'env_vars' in requirements:
            for var_name, required_value in requirements['env_vars'].items():
                current_value = os.environ.get(var_name, '')
                
                results[f"env_{var_name}"] = {
                    'status': bool(current_value) and (required_value is True or current_value == required_value),
                    'message': f"Environment variable {var_name} is required",
                    'required': required_value if required_value is not True else "(set)",
                    'found': current_value or "(not set)"
                }
        
        # Check file system permissions
        if 'required_paths' in requirements:
            for path, required_perms in requirements['required_paths'].items():
                path_exists = os.path.exists(path)
                path_perms = self._check_path_permissions(path)
                
                results[f"path_{path}"] = {
                    'status': all(path_perms[perm] for perm in required_perms if perm in path_perms),
                    'message': f"Path {path} must be {', '.join(required_perms)}",
                    'required': required_perms,
                    'found': {
                        'exists': path_exists,
                        'permissions': path_perms
                    }
                }
        
        return results
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available using env_detector.
        
        Returns:
            bool: True if GPU is available, False otherwise
        """
        detector_info = detect_environment_info()
        return bool(detector_info.get('gpu_info') and detector_info['gpu_info'] != 'Unknown')
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get information about available GPUs using env_detector.
        
        Returns:
            Dictionary with GPU information
        """
        detector_info = detect_environment_info()
        gpu_info = detector_info.get('gpu_info', {})
        
        # Return empty dict if no GPU info or GPU not available
        if not gpu_info or gpu_info == 'Unknown':
            return {'count': 0, 'devices': []}
            
        # Convert env_detector format to our format if needed
        if isinstance(gpu_info, dict):
            # If it's already in the right format, return as is
            if 'count' in gpu_info and 'devices' in gpu_info:
                return cast(Dict[str, Any], gpu_info)
            
            # Otherwise, try to convert from env_detector's format
            return {
                'count': 1 if gpu_info.get('name') else 0,
                'devices': [
                    {
                        'name': gpu_info.get('name', 'Unknown'),
                        'memory': gpu_info.get('memory', 0),
                        'driver_version': gpu_info.get('driver_version', 'unknown')
                    }
                ] if gpu_info.get('name') else []
            }
            
        return {'count': 0, 'devices': []}
    
    def _get_environment_type(self) -> str:
        """Detect the type of environment using environment manager.
        
        Returns:
            str: Environment type ('colab' or 'local')
        """
        env_manager = get_environment_manager()
        return 'colab' if env_manager.is_colab else 'local'
    
    def _version_compare(self, v1: str, v2: str) -> int:
        """Compare two version strings.
        
        Args:
            v1: First version string
            v2: Second version string
            
        Returns:
            -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
        """
        from distutils.version import LooseVersion
        return (LooseVersion(v1) > LooseVersion(v2)) - (LooseVersion(v1) < LooseVersion(v2))
    
    def _check_path_permissions(self, path: str) -> Dict[str, bool]:
        """Check permissions for a path.
        
        Args:
            path: Path to check
            
        Returns:
            Dictionary with permission checks
        """
        result = {
            'exists': False,
            'readable': False,
            'writable': False,
            'executable': False
        }
        
        if not os.path.exists(path):
            return result
        
        result['exists'] = True
        
        try:
            if os.access(path, os.R_OK):
                result['readable'] = True
            if os.access(path, os.W_OK):
                result['writable'] = True
            if os.access(path, os.X_OK):
                result['executable'] = True
        except Exception:
            pass
        
        return result
    
    def get_last_check_result(self) -> Optional[StatusCheckResult]:
        """Get the result of the last environment check.
        
        Returns:
            The last status check result, or None if no check has been performed
        """
        return self._last_check_result
