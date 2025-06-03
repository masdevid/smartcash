"""
File: smartcash/common/config/manager.py
Deskripsi: Config manager dengan tanggung jawab yang jelas dan logging minimal
"""

import os
import copy
import shutil
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

from smartcash.common.constants.core import DEFAULT_CONFIG_DIR, APP_NAME
from smartcash.common.constants.paths import COLAB_PATH, DRIVE_PATH

class SimpleConfigManager:
    """Config manager dengan fokus pada operasi config tanpa environment management"""
    
    def __init__(self, base_dir: Optional[str] = None, config_file: Optional[str] = None):
        """Inisialisasi config manager"""
        # Setup minimal logging
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.WARNING)  # Hanya warning dan error
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.WARNING)
        
        # Set directories
        if base_dir is None:
            self.base_dir = self._get_default_base_dir()
        else:
            self.base_dir = Path(base_dir)
            
        self.config_file = config_file or "base_config.yaml"
        self.config_cache = {}
        
        # Config directories
        self.config_dir = self.base_dir / DEFAULT_CONFIG_DIR
        self.repo_config_dir = Path('/content/smartcash/configs')
        self.drive_config_dir = Path(DRIVE_PATH) / DEFAULT_CONFIG_DIR
        
        # Setup structure hanya jika diperlukan
        self._ensure_config_directory()
    
    def _get_default_base_dir(self) -> Path:
        """Get default base directory"""
        try:
            import google.colab
            return Path(COLAB_PATH)
        except ImportError:
            return Path(__file__).resolve().parents[3]
    
    def _ensure_config_directory(self) -> None:
        """Ensure config directory exists"""
        try:
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False
            self.config_dir.mkdir(parents=True, exist_ok=True)
            return
        
        if not is_colab:
            return
            
        # Check jika symlink sudah ada dan valid
        if (self.config_dir.is_symlink() and 
            self.config_dir.exists() and 
            self.config_dir.resolve() == self.drive_config_dir.resolve()):
            return
        
        # Setup symlink jika drive tersedia
        if Path('/content/drive/MyDrive').exists():
            try:
                self.drive_config_dir.mkdir(parents=True, exist_ok=True)
                
                # Remove existing directory/symlink
                if self.config_dir.exists():
                    if self.config_dir.is_symlink():
                        self.config_dir.unlink()
                    else:
                        shutil.rmtree(self.config_dir)
                
                # Create symlink
                self.config_dir.symlink_to(self.drive_config_dir)
            except Exception:
                # Fallback to local directory
                self.config_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Local directory jika drive tidak tersedia
            self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def get_config_path(self, config_name: str = None) -> Path:
        """Get path ke config file"""
        if config_name is None:
            config_name = self.config_file
        
        if not (config_name.endswith('.yaml') or config_name.endswith('.yml')):
            if not config_name.endswith('_config'):
                config_name = f"{config_name}_config.yaml"
            else:
                config_name = f"{config_name}.yaml"
        
        return self.config_dir / config_name
    
    def load_config(self, config_name: str = None) -> Dict[str, Any]:
        """Load config file"""
        config_path = self.get_config_path(config_name)
        
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                
                cache_key = config_name or self.config_file
                self.config_cache[cache_key] = copy.deepcopy(config)
                return config
        except Exception as e:
            self._logger.error(f"Error loading config {config_path}: {str(e)}")
        
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

# Singleton instance
_INSTANCE = None

def get_config_manager(base_dir=None, config_file=None):
    """Get singleton config manager"""
    global _INSTANCE
    
    if _INSTANCE is None:
        try:
            _INSTANCE = SimpleConfigManager(base_dir, config_file)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error creating config manager: {str(e)}")
            raise
    
    return _INSTANCE

# Compatibility
SimpleConfigManager.get_instance = staticmethod(get_config_manager)