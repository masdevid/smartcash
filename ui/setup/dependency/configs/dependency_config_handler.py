"""
File: smartcash/ui/setup/dependency/configs/dependency_config_handler.py
Deskripsi: Config handler untuk dependency menggunakan proper inheritance
"""

from typing import Dict, Any, List
from pathlib import Path
import yaml

from smartcash.ui.core.handlers.config_handler import PersistentConfigHandler
from .dependency_defaults import get_default_dependency_config

class DependencyConfigHandler(PersistentConfigHandler):
    """Handler untuk dependency config dengan persistence ke YAML"""
    
    def __init__(self):
        config_path = Path("smartcash/configs/dependency_config.yaml")
        default_config = get_default_dependency_config()
        
        super().__init__(
            module_name='dependency',
            parent_module='setup',
            config_path=config_path,
            default_config=default_config
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate dependency configuration"""
        try:
            # Validate required keys
            required_keys = ['module_name', 'version', 'selected_packages', 'custom_packages']
            for key in required_keys:
                if key not in config:
                    self.logger.error(f"❌ Missing required key: {key}")
                    return False
            
            # Validate selected_packages adalah list
            if not isinstance(config['selected_packages'], list):
                self.logger.error("❌ selected_packages must be a list")
                return False
            
            # Validate custom_packages adalah string
            if not isinstance(config['custom_packages'], str):
                self.logger.error("❌ custom_packages must be a string")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating config: {e}")
            return False
    
    def add_selected_package(self, package_name: str) -> bool:
        """Add package ke selected_packages"""
        try:
            config = self.config.copy()
            selected_packages = config.get('selected_packages', [])
            
            if package_name not in selected_packages:
                selected_packages.append(package_name)
                config['selected_packages'] = selected_packages
                self.config = config  # Trigger save
                
                self.logger.info(f"✅ Package '{package_name}' ditambahkan ke selected")
                return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error adding package {package_name}: {e}")
            return False
    
    def remove_selected_package(self, package_name: str) -> bool:
        """Remove package dari selected_packages"""
        try:
            config = self.config.copy()
            selected_packages = config.get('selected_packages', [])
            
            if package_name in selected_packages:
                selected_packages.remove(package_name)
                config['selected_packages'] = selected_packages
                self.config = config  # Trigger save
                
                self.logger.info(f"✅ Package '{package_name}' dihapus dari selected")
                return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error removing package {package_name}: {e}")
            return False
    
    def update_custom_packages(self, custom_packages: str) -> bool:
        """Update custom packages string"""
        try:
            config = self.config.copy()
            config['custom_packages'] = custom_packages
            self.config = config  # Trigger save
            
            self.logger.info("✅ Custom packages berhasil diupdate")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error updating custom packages: {e}")
            return False
    
    def get_selected_packages(self) -> List[str]:
        """Get list selected packages"""
        return self.config.get('selected_packages', [])
    
    def get_custom_packages(self) -> str:
        """Get custom packages string"""
        return self.config.get('custom_packages', '')
    
    def get_install_options(self) -> Dict[str, Any]:
        """Get install options"""
        return self.config.get('install_options', {})
    
    def get_ui_settings(self) -> Dict[str, Any]:
        """Get UI settings"""
        return self.config.get('ui_settings', {})
    
    def get_package_count(self) -> int:
        """Get total package count"""
        selected_count = len(self.get_selected_packages())
        custom_count = len([line.strip() for line in self.get_custom_packages().split('\n') if line.strip()])
        return selected_count + custom_count