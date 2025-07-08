"""
File: smartcash/ui/setup/colab/configs/colab_config_handler.py
Description: Configuration handler for colab module following dependency pattern
"""

from typing import Dict, Any, Optional, List
import copy
import os

try:
    from smartcash.common.logger import SmartCashLogger
except ImportError:
    # Fallback for testing
    class SmartCashLogger:
        def __init__(self, name=None): pass
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def success(self, msg): print(f"SUCCESS: {msg}")

from .colab_defaults import (
    get_default_colab_config,
    get_available_environments,
    get_setup_stages_config,
    get_gpu_configurations
)


class ColabConfigHandler:
    """Configuration handler for colab module following dependency pattern."""
    
    def __init__(self, logger: Optional[SmartCashLogger] = None):
        """Initialize configuration handler.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or SmartCashLogger('ColabConfigHandler')
        self._config = get_default_colab_config()
        self._available_environments = get_available_environments()
        self._setup_stages = get_setup_stages_config()
        self._gpu_configurations = get_gpu_configurations()
        
        # Detect current environment
        self._detect_environment()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return copy.deepcopy(self._config)
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """Update configuration.
        
        Args:
            config: New configuration to apply
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Validate configuration
            if not self.validate_config(config):
                return False
            
            # Deep merge configuration
            self._config = self._deep_merge(self._config, config)
            self.logger.info("✅ Configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update configuration: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            env_config = config.get('environment', {})
            
            # Validate environment type
            env_type = env_config.get('type')
            if env_type and env_type not in self._available_environments:
                self.logger.warning(f"⚠️ Unknown environment type: {env_type}")
                return False
            
            # Validate setup stages
            setup_config = config.get('setup', {})
            stages = setup_config.get('stages', [])
            for stage in stages:
                if stage not in self._setup_stages:
                    self.logger.warning(f"⚠️ Unknown setup stage: {stage}")
                    return False
            
            # Validate paths
            paths_config = config.get('paths', {})
            base_path = paths_config.get('colab_base')
            if base_path and not self._is_valid_path(base_path):
                self.logger.warning(f"⚠️ Invalid base path: {base_path}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self._config = get_default_colab_config()
        self._detect_environment()
        self.logger.info("🔄 Configuration reset to defaults")
    
    def get_available_environments(self) -> Dict[str, Dict[str, Any]]:
        """Get available environments.
        
        Returns:
            Dictionary of available environments
        """
        return copy.deepcopy(self._available_environments)
    
    def get_setup_stages_config(self) -> Dict[str, Dict[str, Any]]:
        """Get setup stages configuration.
        
        Returns:
            Dictionary of setup stages configuration
        """
        return copy.deepcopy(self._setup_stages)
    
    def get_gpu_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get GPU configurations.
        
        Returns:
            Dictionary of GPU configurations
        """
        return copy.deepcopy(self._gpu_configurations)
    
    def set_environment_type(self, env_type: str) -> bool:
        """Set environment type.
        
        Args:
            env_type: Environment type to set
            
        Returns:
            True if successful, False otherwise
        """
        if env_type not in self._available_environments:
            self.logger.warning(f"⚠️ Unknown environment type: {env_type}")
            return False
        
        self._config['environment']['type'] = env_type
        
        # Update paths based on environment
        env_info = self._available_environments[env_type]
        self._config['environment']['base_path'] = env_info['base_path']
        
        # Update mount requirement
        if env_info.get('mount_required', False):
            self._config['environment']['auto_mount_drive'] = True
        
        self.logger.info(f"✅ Environment type set to: {env_type}")
        return True
    
    def set_setup_stages(self, stages: List[str]) -> bool:
        """Set setup stages.
        
        Args:
            stages: List of setup stages
            
        Returns:
            True if successful, False otherwise
        """
        # Validate stages
        for stage in stages:
            if stage not in self._setup_stages:
                self.logger.warning(f"⚠️ Unknown setup stage: {stage}")
                return False
        
        self._config['setup']['stages'] = stages
        self.logger.info(f"✅ Setup stages set to: {stages}")
        return True
    
    def set_gpu_enabled(self, enabled: bool, gpu_type: Optional[str] = None) -> bool:
        """Set GPU configuration.
        
        Args:
            enabled: Whether to enable GPU
            gpu_type: Type of GPU (optional)
            
        Returns:
            True if successful, False otherwise
        """
        self._config['environment']['gpu_enabled'] = enabled
        
        if enabled and gpu_type:
            if gpu_type not in self._gpu_configurations:
                self.logger.warning(f"⚠️ Unknown GPU type: {gpu_type}")
                return False
            self._config['environment']['gpu_type'] = gpu_type
        
        self.logger.info(f"✅ GPU enabled: {enabled}")
        return True
    
    def set_auto_mount_drive(self, auto_mount: bool) -> bool:
        """Set auto mount drive setting.
        
        Args:
            auto_mount: Whether to auto mount drive
            
        Returns:
            True if successful, False otherwise
        """
        self._config['environment']['auto_mount_drive'] = auto_mount
        self.logger.info(f"✅ Auto mount drive set to: {auto_mount}")
        return True
    
    def set_project_name(self, project_name: str) -> bool:
        """Set project name.
        
        Args:
            project_name: Project name to set
            
        Returns:
            True if successful, False otherwise
        """
        if not project_name or not project_name.strip():
            self.logger.warning("⚠️ Project name cannot be empty")
            return False
        
        self._config['environment']['project_name'] = project_name.strip()
        
        # Update paths with new project name
        self._update_project_paths(project_name.strip())
        
        self.logger.info(f"✅ Project name set to: {project_name}")
        return True
    
    def get_current_environment(self) -> str:
        """Get current environment type.
        
        Returns:
            Current environment type
        """
        return self._config['environment']['type']
    
    def get_current_gpu_config(self) -> Dict[str, Any]:
        """Get current GPU configuration.
        
        Returns:
            Current GPU configuration
        """
        env_config = self._config['environment']
        gpu_enabled = env_config.get('gpu_enabled', False)
        gpu_type = env_config.get('gpu_type', 'none')
        
        if gpu_enabled and gpu_type in self._gpu_configurations:
            return self._gpu_configurations[gpu_type]
        
        return self._gpu_configurations['none']
    
    def _detect_environment(self) -> None:
        """Detect current environment and update configuration."""
        try:
            if self._is_colab():
                self.set_environment_type('colab')
            elif self._is_kaggle():
                self.set_environment_type('kaggle')
            else:
                self.set_environment_type('local')
                
        except Exception as e:
            self.logger.warning(f"⚠️ Environment detection failed: {e}")
    
    def _is_colab(self) -> bool:
        """Check if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _is_kaggle(self) -> bool:
        """Check if running in Kaggle."""
        return os.path.exists('/kaggle')
    
    def _is_valid_path(self, path: str) -> bool:
        """Validate if path is valid.
        
        Args:
            path: Path to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if path is absolute
            if not os.path.isabs(path):
                return False
            
            # Check if parent directory exists or can be created
            parent = os.path.dirname(path)
            return os.path.exists(parent) or parent == '/'
            
        except Exception:
            return False
    
    def _update_project_paths(self, project_name: str) -> None:
        """Update project paths with new project name."""
        paths = self._config['paths']
        base_path = self._config['environment']['base_path']
        
        if self._config['environment']['type'] == 'colab':
            paths['drive_base'] = f"/content/drive/MyDrive/{project_name}"
            paths['colab_base'] = f"/content/{project_name}"
        else:
            paths['colab_base'] = f"{base_path}/{project_name}"
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary
            
        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(dict1)
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result