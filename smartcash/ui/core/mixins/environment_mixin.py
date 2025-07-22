"""
File: smartcash/ui/core/mixins/environment_mixin.py
Description: Environment management mixin for UI modules.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass

from smartcash.common.environment import get_environment_manager, EnvironmentManager
from smartcash.common.constants.paths import get_paths_for_environment


@dataclass
class EnvironmentPaths:
    """Data class to store environment paths."""
    data_root: str = 'data'
    config: str = './smartcash/configs'
    logs: str = 'logs'
    cache: str = '.cache'


class EnvironmentMixin:
    """
    Mixin for environment management in UI modules.
    
    Provides standardized environment detection and path management.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize environment mixin.
        
        Note: This mixin doesn't call super().__init__() to avoid MRO issues.
        The parent class should be properly initialized before this mixin.
        """
        # Initialize attributes without calling super() to avoid MRO issues
        self._environment_manager: Optional[EnvironmentManager] = None
        self._environment_paths = EnvironmentPaths()
        self._is_colab: Optional[bool] = None
        self._is_drive_mounted: Optional[bool] = None
        
        # Automatically set up the environment
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """
        Setup environment management.
        
        This method should be called during initialization of the parent class.
        """
        try:
            # Use standardized environment manager
            self._environment_manager = get_environment_manager(
                logger=getattr(self, 'logger', None)
            )
            
            # Set environment flags
            self._is_colab = getattr(self._environment_manager, 'is_colab', False)
            self._is_drive_mounted = (
                getattr(self._environment_manager, 'is_drive_mounted', False)
                if self._is_colab else False
            )
            
            # Get appropriate paths for current environment
            paths = get_paths_for_environment(
                is_colab=self._is_colab,
                is_drive_mounted=self._is_drive_mounted
            )
            
            # Update environment paths with any additional paths from the environment
            for key, value in paths.items():
                if hasattr(self._environment_paths, key):
                    setattr(self._environment_paths, key, value)
            
            # Log environment information
            self._log_environment_info()
            
        except Exception as e:
            logger = getattr(self, 'logger', None)
            if logger:
                logger.error(f"Failed to setup environment: {e}")
            # Use default paths from EnvironmentPaths
            
    def _log_environment_info(self) -> None:
        """Log environment information."""
        if not hasattr(self, 'logger'):
            return
            
        env_type = "Google Colab" if self._is_colab else "Lokal/Jupyter"
        self.logger.debug(f"✅ Environment detected: {env_type}")
        
        # Log to UI if possible
        if hasattr(self, 'log'):
            self.log(f"🌍 Lingkungan terdeteksi: {env_type}", 'info')
            self.log(
                f"📁 Direktori kerja: {self._environment_paths.data_root}", 
                'info'
            )
    
    @property
    def environment_paths(self) -> EnvironmentPaths:
        """Get environment paths."""
        if not hasattr(self, '_environment_paths'):
            self._setup_environment()
        return self._environment_paths
    
    @property
    def is_colab(self) -> bool:
        """Check if running in Google Colab."""
        # Force re-check of Colab environment
        if hasattr(self, '_environment_manager'):
            self._is_colab = getattr(self._environment_manager, 'is_colab', False)
        if self._is_colab is None:
            self._setup_environment()
        return self._is_colab or False
    
    @property
    def is_drive_mounted(self) -> bool:
        """Check if Google Drive is mounted (only relevant in Colab)."""
        if self._is_colab is None:
            self._setup_environment()
        return self._is_drive_mounted or False
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get environment information.
        
        Returns:
            Dict containing environment information
        """
        return {
            'environment_type': 'colab' if self.is_colab else 'local',
            'drive_mounted': self.is_drive_mounted,
            'paths': self._environment_paths.__dict__
        }
