"""
Colab Config Handler (Optimized) - Pure Delegation Pattern
Configuration handler for Google Colab environment setup.
"""

from typing import Dict, Any, Optional, List
import os
from smartcash.ui.logger import get_module_logger
from .colab_defaults import get_default_colab_config, get_available_environments, get_setup_stages_config, get_gpu_configurations


class ColabConfigHandler:
    """Optimized configuration handler using pure delegation pattern."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None):
        self.logger = logger or get_module_logger('smartcash.ui.setup.colab.configs')
        self.module_name, self.parent_module = 'colab', 'setup'
        self._config = config or get_default_colab_config()
        self.config = self._config  # Backwards compatibility
        self._available_environments = get_available_environments()
        self._setup_stages = get_setup_stages_config()
        self._gpu_configurations = get_gpu_configurations()
        self._detect_environment()
        self.logger.info("✅ ColabConfigHandler initialized")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self._config.copy()
    
    def get_default_config(self) -> Dict[str, Any]: 
        return get_default_colab_config()
    
    def create_config_handler(self, config: Dict[str, Any]) -> 'ColabConfigHandler': 
        return ColabConfigHandler(config, self.logger)
    
    def extract_config_from_ui(self) -> Dict[str, Any]: 
        return self._config.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        self._deep_merge(self._config, config)
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        try:
            env_config = config.get('environment', {})
            if env_config.get('type') and env_config['type'] not in self._available_environments:
                self.logger.warning(f"Unknown environment type: {env_config['type']}")
                return False
            
            setup_config = config.get('setup', {})
            for stage in setup_config.get('stages', []):
                if stage not in self._setup_stages:
                    self.logger.warning(f"Unknown setup stage: {stage}")
                    return False
            
            paths_config = config.get('paths', {})
            base_path = paths_config.get('colab_base')
            if base_path and not self._is_valid_path(base_path):
                self.logger.warning(f"Invalid base path: {base_path}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def save_config(self) -> Dict[str, Any]:
        """Save the current configuration."""
        try:
            current_config = self.get_current_config()
            self.logger.info(f"✅ Colab configuration saved with {len(current_config)} keys")
            return {
                'success': True,
                'message': f"Colab configuration saved with {len(current_config)} keys",
                'config': current_config
            }
        except Exception as e:
            error_msg = f"Failed to save Colab configuration: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg, 'config': getattr(self, '_config', {})}
    
    def reset_config(self) -> Dict[str, Any]:
        """Reset configuration to defaults."""
        try:
            default_config = self.get_default_config()
            self._config = default_config.copy()
            self.config = self._config  # Backwards compatibility
            self._detect_environment()
            self.logger.info("✅ Colab configuration reset to defaults")
            return {
                'success': True,
                'message': "Colab configuration reset to defaults",
                'config': self._config
            }
        except Exception as e:
            error_msg = f"Failed to reset Colab configuration: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg, 'config': getattr(self, '_config', {})}
    
    # Environment Management
    def get_available_environments(self) -> Dict[str, Dict[str, Any]]: 
        return self._available_environments.copy()
    
    def get_setup_stages_config(self) -> Dict[str, Dict[str, Any]]: 
        return self._setup_stages.copy()
    
    def get_gpu_configurations(self) -> Dict[str, Dict[str, Any]]: 
        return self._gpu_configurations.copy()
    
    def set_environment_type(self, env_type: str) -> bool:
        if env_type not in self._available_environments:
            self.logger.warning(f"Unknown environment type: {env_type}")
            return False
        
        self.config['environment']['type'] = env_type
        env_info = self._available_environments[env_type]
        self.config['environment']['base_path'] = env_info['base_path']
        
        if env_info.get('mount_required', False):
            self.config['environment']['auto_mount_drive'] = True
        
        self.logger.info(f"Environment type set to: {env_type}")
        return True
    
    def set_setup_stages(self, stages: List[str]) -> bool:
        for stage in stages:
            if stage not in self._setup_stages:
                self.logger.warning(f"Unknown setup stage: {stage}")
                return False
        
        self.config['setup']['stages'] = stages
        self.logger.info(f"Setup stages set to: {stages}")
        return True
    
    def set_gpu_enabled(self, enabled: bool, gpu_type: Optional[str] = None) -> bool:
        self.config['environment']['gpu_enabled'] = enabled
        
        if enabled and gpu_type:
            if gpu_type not in self._gpu_configurations:
                self.logger.warning(f"Unknown GPU type: {gpu_type}")
                return False
            self.config['environment']['gpu_type'] = gpu_type
        
        self.logger.info(f"GPU enabled: {enabled}")
        return True
    
    def set_auto_mount_drive(self, auto_mount: bool) -> bool:
        self.config['environment']['auto_mount_drive'] = auto_mount
        self.logger.info(f"Auto mount drive set to: {auto_mount}")
        return True
    
    def set_project_name(self, project_name: str) -> bool:
        if not project_name or not project_name.strip():
            self.logger.warning("Project name cannot be empty")
            return False
        
        self.config['environment']['project_name'] = project_name.strip()
        self._update_project_paths(project_name.strip())
        self.logger.info(f"Project name set to: {project_name}")
        return True
    
    def get_current_environment(self) -> str: 
        return self.config['environment']['type']
    
    def get_current_gpu_config(self) -> Dict[str, Any]:
        env_config = self.config['environment']
        gpu_enabled = env_config.get('gpu_enabled', False)
        gpu_type = env_config.get('gpu_type', 'none')
        
        if gpu_enabled and gpu_type in self._gpu_configurations:
            return self._gpu_configurations[gpu_type]
        return self._gpu_configurations['none']
    
    # Private Helper Methods
    def _detect_environment(self) -> None:
        try:
            if self._is_colab():
                self.set_environment_type('colab')
            elif self._is_kaggle():
                self.set_environment_type('kaggle')
            else:
                self.set_environment_type('local')
        except Exception as e:
            self.logger.warning(f"Environment detection failed: {e}")
    
    def _is_colab(self) -> bool:
        try:
            import google.colab  # noqa: F401
            return True
        except ImportError:
            return 'COLAB_GPU' in os.environ or '/content' in os.getcwd()
    
    def _is_kaggle(self) -> bool: 
        return os.path.exists('/kaggle')
    
    def _is_valid_path(self, path: str) -> bool:
        try:
            if not os.path.isabs(path):
                return False
            parent = os.path.dirname(path)
            return os.path.exists(parent) or parent == '/'
        except Exception:
            return False
    
    def _update_project_paths(self, project_name: str) -> None:
        paths = self.config['paths']
        base_path = self.config['environment']['base_path']
        
        if self.config['environment']['type'] == 'colab':
            paths['drive_base'] = f"/content/drive/MyDrive/{project_name}"
            paths['colab_base'] = f"/content/{project_name}"
        else:
            paths['colab_base'] = f"{base_path}/{project_name}"
    
    # Enhanced Colab-Specific Methods
    def get_colab_setup_status(self) -> Dict[str, Any]:
        """Get comprehensive Colab setup status."""
        return {
            'environment_type': self.get_current_environment(),
            'gpu_config': self.get_current_gpu_config(),
            'mount_required': self.config['environment'].get('auto_mount_drive', False),
            'project_name': self.config['environment'].get('project_name', 'smartcash'),
            'setup_stages': self.config['setup'].get('stages', []),
            'paths_configured': bool(self.config.get('paths', {}))
        }
    
    def sync_with_environment(self) -> Dict[str, Any]:
        """Sync configuration with detected environment."""
        try:
            # Re-detect environment
            self._detect_environment()
            
            return {
                'success': True,
                'message': 'Configuration synced with environment',
                'environment': self.get_current_environment()
            }
        except Exception as e:
            return {'success': False, 'message': f'Environment sync failed: {e}'}