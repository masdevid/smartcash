"""
File: smartcash/ui/setup/dependency/handlers/dependency_ui_handler.py
Deskripsi: UI handler untuk dependency management
"""

from typing import Dict, Any
from smartcash.ui.core.handlers.ui_handler import ModuleUIHandler
from ..configs.dependency_config_handler import DependencyConfigHandler

class DependencyUIHandler(ModuleUIHandler):
    """Handler untuk dependency UI management"""
    
    def __init__(self, module_name: str = 'dependency', parent_module: str = 'setup', **kwargs):
        super().__init__(module_name, parent_module, **kwargs)
        self.config_handler = DependencyConfigHandler()
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Extract configuration dari UI components"""
        try:
            config = self.config_handler.get_config().copy()
            
            # Extract dari dependency tabs
            if 'dependency_tabs' in self._ui_components:
                tabs = self._ui_components['dependency_tabs']
                
                # Extract selected packages dari package categories tab
                selected_packages = self._extract_selected_packages(tabs)
                config['selected_packages'] = selected_packages
                
                # Extract custom packages dari custom tab
                custom_packages = self._extract_custom_packages(tabs)
                config['custom_packages'] = custom_packages
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting config dari UI: {e}")
            return self.config_handler.get_config()
    
    def _extract_selected_packages(self, tabs) -> list:
        """Extract selected packages dari categories tab"""
        selected_packages = []
        
        try:
            # Get first tab (package categories)
            if hasattr(tabs, 'children') and len(tabs.children) > 0:
                categories_tab = tabs.children[0]
                
                # Find package widgets dalam categories
                for widget in self._find_package_widgets(categories_tab):
                    if hasattr(widget, 'is_selected') and widget.is_selected:
                        if hasattr(widget, 'package_info'):
                            selected_packages.append(widget.package_info['name'])
        except Exception as e:
            self.logger.error(f"❌ Error extracting selected packages: {e}")
        
        return selected_packages
    
    def _extract_custom_packages(self, tabs) -> str:
        """Extract custom packages dari custom tab"""
        try:
            # Get second tab (custom packages)
            if hasattr(tabs, 'children') and len(tabs.children) > 1:
                custom_tab = tabs.children[1]
                
                if hasattr(custom_tab, 'packages_textarea'):
                    return custom_tab.packages_textarea.value.strip()
        except Exception as e:
            self.logger.error(f"❌ Error extracting custom packages: {e}")
        
        return ""
    
    def _find_package_widgets(self, container, widgets_list=None):
        """Recursively find package widgets dalam container"""
        if widgets_list is None:
            widgets_list = []
        
        if hasattr(container, 'package_info'):
            widgets_list.append(container)
        
        if hasattr(container, 'children'):
            for child in container.children:
                self._find_package_widgets(child, widgets_list)
        
        return widgets_list
    
    def update_ui_from_config(self, config: Dict[str, Any]) -> None:
        """Update UI components dari configuration"""
        try:
            if 'dependency_tabs' in self._ui_components:
                tabs = self._ui_components['dependency_tabs']
                
                # Update selected packages
                selected_packages = config.get('selected_packages', [])
                self._update_selected_packages(tabs, selected_packages)
                
                # Update custom packages
                custom_packages = config.get('custom_packages', '')
                self._update_custom_packages(tabs, custom_packages)
                
            self.logger.info("✅ UI berhasil diupdate dari config")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating UI dari config: {e}")
    
    def _update_selected_packages(self, tabs, selected_packages: list) -> None:
        """Update selected packages dalam UI"""
        try:
            if hasattr(tabs, 'children') and len(tabs.children) > 0:
                categories_tab = tabs.children[0]
                
                for widget in self._find_package_widgets(categories_tab):
                    if hasattr(widget, 'package_info'):
                        is_selected = widget.package_info['name'] in selected_packages
                        widget.is_selected = is_selected
                        
                        # Update visual state
                        if hasattr(widget, 'status_button'):
                            widget.status_button.description = '✅' if is_selected else '⭕'
                            widget.status_button.button_style = 'success' if is_selected else ''
                        
                        # Update container styling
                        if hasattr(widget, 'layout'):
                            border_color = '#4CAF50' if is_selected else '#ddd'
                            bg_color = '#fafafa' if is_selected else 'white'
                            widget.layout.border = f'1px solid {border_color}'
                            widget.layout.background_color = bg_color
                            
        except Exception as e:
            self.logger.error(f"❌ Error updating selected packages: {e}")
    
    def _update_custom_packages(self, tabs, custom_packages: str) -> None:
        """Update custom packages dalam UI"""
        try:
            if hasattr(tabs, 'children') and len(tabs.children) > 1:
                custom_tab = tabs.children[1]
                
                if hasattr(custom_tab, 'packages_textarea'):
                    custom_tab.packages_textarea.value = custom_packages
                    
        except Exception as e:
            self.logger.error(f"❌ Error updating custom packages: {e}")
    
    def sync_config_with_ui(self) -> None:
        """Sync configuration dengan UI state"""
        try:
            # Extract current config dari UI
            current_config = self.extract_config_from_ui()
            
            # Update config handler
            self.config_handler.update_config(current_config)
            
            self.logger.info("✅ Config berhasil di-sync dengan UI")
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing config dengan UI: {e}")
    
    def sync_ui_with_config(self) -> None:
        """Sync UI dengan configuration"""
        try:
            # Get current config
            current_config = self.config_handler.get_config()
            
            # Update UI
            self.update_ui_from_config(current_config)
            
            self.logger.info("✅ UI berhasil di-sync dengan config")
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing UI dengan config: {e}")
    
    def get_selected_packages(self) -> list:
        """Get list selected packages"""
        config = self.extract_config_from_ui()
        return config.get('selected_packages', [])
    
    def get_custom_packages(self) -> str:
        """Get custom packages string"""
        config = self.extract_config_from_ui()
        return config.get('custom_packages', '')
    
    def add_selected_package(self, package_name: str) -> bool:
        """Add package ke selected list"""
        try:
            return self.config_handler.add_selected_package(package_name)
        except Exception as e:
            self.logger.error(f"❌ Error adding package {package_name}: {e}")
            return False
    
    def remove_selected_package(self, package_name: str) -> bool:
        """Remove package dari selected list"""
        try:
            return self.config_handler.remove_selected_package(package_name)
        except Exception as e:
            self.logger.error(f"❌ Error removing package {package_name}: {e}")
            return False
    
    def update_custom_packages(self, custom_packages: str) -> bool:
        """Update custom packages"""
        try:
            return self.config_handler.update_custom_packages(custom_packages)
        except Exception as e:
            self.logger.error(f"❌ Error updating custom packages: {e}")
            return False