"""
File: smartcash/common/config/manager.py
Deskripsi: Extended SimpleConfigManager dengan sync_configs_to_drive method
"""

import os
import copy
import shutil
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from smartcash.common.constants.core import DEFAULT_CONFIG_DIR, APP_NAME
from smartcash.common.constants.paths import COLAB_PATH, DRIVE_PATH

class SimpleConfigManager:
    """Config manager dengan sync functionality"""
    
    def __init__(self, base_dir: Optional[str] = None, config_file: Optional[str] = None, auto_sync: bool = False):
        """Initialize config manager
        
        Args:
            base_dir: Base directory for configs (defaults to appropriate location based on environment)
            config_file: Name of the main config file (default: base_config.yaml)
            auto_sync: Whether to automatically sync configs on initialization (default: False)
        """
        # Setup logging
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)  # More verbose logging for debugging
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
        
        try:
            # Set directories
            if base_dir is None:
                self.base_dir = self._get_default_base_dir()
            else:
                self.base_dir = Path(base_dir)
                
            self.config_file = config_file or "base_config.yaml"
            self.config_cache = {}
            self.auto_sync = auto_sync
            
            # Config directories - try multiple possible locations
            self.config_dir = self.base_dir / DEFAULT_CONFIG_DIR
            
            # Try to find the repo config directory in multiple locations
            possible_repo_dirs = [
                Path('/content/smartcash/configs'),  # Colab default
                self.base_dir / 'smartcash' / 'configs',  # Local dev: /Users/masdevid/Projects/smartcash/smartcash/configs
                Path(__file__).parent.parent.parent.parent.parent / 'configs',  # Alternative local dev
                Path('/content/configs'),  # Alternative location
                self.base_dir / 'configs'  # Fallback
            ]
            
            # Use the first existing directory, or the first one if none exist
            self.repo_config_dir = None
            for dir_path in possible_repo_dirs:
                if dir_path.exists() and dir_path.is_dir():
                    self.repo_config_dir = dir_path
                    self._logger.info(f"Using config directory: {self.repo_config_dir}")
                    break
            
            if self.repo_config_dir is None:
                # Create the first possible directory
                self.repo_config_dir = possible_repo_dirs[0]
                self._logger.warning(f"No existing config directory found, will use: {self.repo_config_dir}")
            
            # Ensure the directory exists
            self.repo_config_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up drive config directory
            self.drive_config_dir = Path(DRIVE_PATH) / DEFAULT_CONFIG_DIR
            
            # Create a sample config if none exists
            self._create_sample_configs()
            
            # Setup structure
            self._ensure_config_directory()
            
            # Only auto-sync if explicitly enabled
            if self.auto_sync:
                self.sync_configs_to_drive(force_overwrite=False)
                
        except Exception as e:
            self._logger.error(f"Error initializing ConfigManager: {str(e)}", exc_info=True)
            raise
    
    def _get_default_base_dir(self) -> Path:
        """Get default base directory"""
        try:
            # Check for Colab environment
            if 'COLAB_GPU' in os.environ:
                self._logger.info("Detected Colab environment")
                return Path(COLAB_PATH)
            
            # Check for local development environment
            cwd = Path(os.getcwd())
            if (cwd / 'smartcash').exists():
                self._logger.info(f"Detected local development environment in {cwd}")
                return cwd
                
            # Fall back to current working directory
            self._logger.info(f"Using current working directory as base: {cwd}")
            return cwd
            
        except Exception as e:
            self._logger.error(f"Error determining base directory: {e}")
            return Path(os.getcwd())
    
    def _create_sample_configs(self):
        """Create sample config files if they don't exist"""
        try:
            sample_configs = {
                'base_config.yaml': """# Base Configuration\n# This is a sample configuration file\n\n# Application settings\napp:\n  name: "SmartCash"\n  version: "1.0.0"\n  debug: false\n\n# Data paths\npaths:\n  data: "/content/data"\n  models: "/content/models"\n  logs: "/content/logs"\n\n# Model settings\nmodel:\n  batch_size: 32\n  learning_rate: 0.001\n  epochs: 10\n\n# Logging\nlogging:\n  level: "INFO"\n  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"\n""",
                'colab_config.yaml': """# Colab-Specific Configuration\n# This config is specific to Google Colab environments\n\ncolab:\n  gpu: true\n  drive_mount: true\n  drive_path: "/content/drive/MyDrive"\n  \n  # Google Drive settings\n  drive:\n    sync_enabled: true\n    sync_interval: 300  # seconds\n    \n  # Display settings\n  display:\n    show_images: true\n    image_columns: 2\n    image_size: "(10, 8)"\n"""
            }
            
            # Create sample configs if they don't exist
            for filename, content in sample_configs.items():
                config_path = self.repo_config_dir / filename
                if not config_path.exists():
                    with open(config_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self._logger.info(f"Created sample config: {config_path}")
                    
        except Exception as e:
            self._logger.warning(f"Could not create sample configs: {str(e)}")
    
    def _ensure_config_directory(self):
        """Ensure config directory exists and is properly set up"""
        try:
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Ensured config directory exists: {self.config_dir}")
            
            # Create a README in the config directory
            readme_path = self.config_dir / 'README.md'
            if not readme_path.exists():
                readme_content = """# SmartCash Configuration Directory

This directory contains configuration files for the SmartCash application.

## Default Files

- `base_config.yaml`: Main configuration file with application settings
- `colab_config.yaml`: Configuration specific to Google Colab environments

## Adding New Configurations

1. Create a new `.yaml` file in this directory
2. Use the existing configs as templates
3. The application will automatically discover and load the new configuration

## Syncing with Google Drive

In Colab environments, configurations can be synced with Google Drive to persist between sessions.
"""
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                
        except Exception as e:
            self._logger.error(f"Could not ensure config directory: {str(e)}")
            raise
    
    def discover_repo_configs(self) -> List[str]:
        """🔍 Discover semua config files di repo"""
        if not self.repo_config_dir.exists():
            return []
        
        config_extensions = ['.yaml', '.yml', '.json', '.toml']
        discovered = []
        
        for ext in config_extensions:
            discovered.extend([
                f.name for f in self.repo_config_dir.glob(f'*{ext}')
                if f.is_file() and not f.name.startswith('.')
            ])
        
        return sorted(list(set(discovered)))  # Remove duplicates
    
    def sync_single_config(self, filename: str, force_overwrite: bool = False) -> Tuple[bool, str]:
        """📋 Sync single config file dari repo ke Drive"""
        source_file = self.repo_config_dir / filename
        dest_file = self.drive_config_dir / filename
        
        # Check source exists
        if not source_file.exists():
            return False, f"Source tidak ditemukan: {filename}"
        
        # Ensure destination directory
        self.drive_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if overwrite needed
        if dest_file.exists() and not force_overwrite:
            try:
                # Quick content comparison
                if source_file.read_text(encoding='utf-8') == dest_file.read_text(encoding='utf-8'):
                    return True, f"Identik, skip: {filename}"
            except Exception:
                pass  # Proceed with copy if comparison fails
        
        # Perform copy
        try:
            shutil.copy2(source_file, dest_file)
            return True, f"Sync berhasil: {filename}"
        except Exception as e:
            return False, f"Error sync {filename}: {str(e)}"
    
    def sync_configs_to_drive(self, force_overwrite: bool = False, 
                            target_configs: Optional[List[str]] = None) -> Dict[str, Any]:
        """🚀 Sync configs from repo to Drive with auto-discovery
        
        Args:
            force_overwrite: Whether to overwrite existing files (default: False)
            target_configs: List of specific config files to sync (default: None = sync all)
            
        Returns:
            Dictionary containing sync results
        """
        # Discover available configs
        available_configs = self.discover_repo_configs()
        
        if not available_configs:
            return {
                'success': False,
                'message': 'No configs found in repository',
                'synced_count': 0,
                'error_count': 1
            }
        
        # Determine configs to sync
        if target_configs:
            configs_to_sync = [cfg for cfg in target_configs if cfg in available_configs]
        else:
            configs_to_sync = available_configs
        
        # Sync each config
        synced_files = []
        skipped_files = []
        error_files = []
        
        for config in configs_to_sync:
            success, message = self.sync_single_config(config, force_overwrite)
            
            if success:
                if "skip" in message.lower():
                    skipped_files.append(config)
                else:
                    synced_files.append(config)
            else:
                error_files.append((config, message))
        
        # Generate result
        total_processed = len(synced_files) + len(skipped_files) + len(error_files)
        success_rate = (len(synced_files) + len(skipped_files)) / max(total_processed, 1)
        
        return {
            'success': success_rate >= 0.8,  # 80% success threshold
            'message': self._generate_sync_message(synced_files, skipped_files, error_files),
            'synced_count': len(synced_files),
            'skipped_count': len(skipped_files),
            'error_count': len(error_files),
            'synced_files': synced_files,
            'skipped_files': skipped_files,
            'error_files': error_files,
            'discovered_configs': available_configs,
            'success_rate': round(success_rate * 100, 1)
        }
    
    def _generate_sync_message(self, synced: List[str], skipped: List[str], 
                             errors: List[Tuple[str, str]]) -> str:
        """📊 Generate human-readable sync message"""
        if not errors:
            if synced:
                return f"✅ Sync {len(synced)} configs" + (f", skip {len(skipped)} identik" if skipped else "")
            else:
                return f"ℹ️ Semua {len(skipped)} configs up-to-date"
        else:
            return f"⚠️ Sync {len(synced)} berhasil, {len(errors)} error"
    
    # === EXISTING METHODS ===
    
    def get_config_path(self, config_name: str = None) -> Path:
        """Get path untuk config file"""
        filename = config_name or self.config_file
        return self.config_dir / filename
    
    def load_config(self, config_name: str = None) -> Dict[str, Any]:
        """Load config dari file"""
        config_path = self.get_config_path(config_name)
        
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                    
                cache_key = config_name or self.config_file
                self.config_cache[cache_key] = copy.deepcopy(config)
                return config
            else:
                return {}
                
        except Exception as e:
            self._logger.error(f"Error loading config: {str(e)}")
            return {}
    
    def save_config(self, config: Dict[str, Any], config_name: str = None) -> bool:
        """Save config file"""
        try:
            config_path = self.get_config_path(config_name)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            cache_key = config_name or self.config_file
            self.config_cache[cache_key] = copy.deepcopy(config)
            return True
            
        except Exception as e:
            self._logger.error(f"Error saving config: {str(e)}")
            return False
    
    def get_config(self, config_name: str = None, reload: bool = False) -> Dict[str, Any]:
        """Get config dengan caching"""
        cache_key = config_name or self.config_file
        
        if reload or cache_key not in self.config_cache:
            return self.load_config(config_name)
        
        return copy.deepcopy(self.config_cache.get(cache_key, {}))
    
    def is_symlink_active(self) -> bool:
        """Check apakah symlink config aktif"""
        return self.config_dir.is_symlink() and self.config_dir.exists()
    
    def get_environment_config_path(self, filename: str) -> str:
        """Get environment-aware config file path.
        
        Args:
            filename: Name of the config file
            
        Returns:
            Full path to the config file based on current environment
        """
        try:
            # Use the existing config_dir which is already environment-aware
            config_path = self.config_dir / filename
            
            # Ensure the directory exists
            os.makedirs(self.config_dir, exist_ok=True)
            
            return str(config_path)
            
        except Exception as e:
            self._logger.error(f"Error getting config path for {filename}: {e}")
            # Fallback to current directory
            return os.path.join('./configs', filename)

# Singleton instance
_INSTANCE = None

def get_config_manager(base_dir=None, config_file=None, auto_sync=False):
    """Get singleton config manager
    
    Args:
        base_dir: Base directory for configs
        config_file: Name of the main config file
        auto_sync: Whether to automatically sync configs on initialization (default: False)
    """
    global _INSTANCE
    
    if _INSTANCE is None:
        try:
            _INSTANCE = SimpleConfigManager(base_dir, config_file, auto_sync=auto_sync)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error creating config manager: {str(e)}")
            raise
    
    return _INSTANCE

# Compatibility
SimpleConfigManager.get_instance = staticmethod(get_config_manager)

def get_environment_config_path(filename: str) -> str:
    """Get environment-aware config file path using the singleton config manager.
    
    Args:
        filename: Name of the config file
        
    Returns:
        Full path to the config file based on current environment
    """
    try:
        config_manager = get_config_manager()
        return config_manager.get_environment_config_path(filename)
    except Exception as e:
        # Fallback to current directory
        return os.path.join('./configs', filename)