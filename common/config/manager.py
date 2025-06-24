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
    
    def __init__(self, base_dir: Optional[str] = None, config_file: Optional[str] = None):
        """Inisialisasi config manager"""
        # Setup minimal logging
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.WARNING)
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
        
        # Setup structure
        self._ensure_config_directory()
    
    def _get_default_base_dir(self) -> Path:
        """Get default base directory"""
        try:
            import google.colab
            return Path(COLAB_PATH)
        except ImportError:
            return Path(os.getcwd())
    
    def _ensure_config_directory(self):
        """Ensure config directory exists"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._logger.warning(f"Could not create config directory: {str(e)}")
    
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
        """🚀 Sync configs dari repo ke Drive dengan auto-discovery"""
        
        # Discover configs yang tersedia
        available_configs = self.discover_repo_configs()
        
        if not available_configs:
            return {
                'success': False,
                'message': 'Tidak ada config ditemukan di repo',
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