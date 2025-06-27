# =============================================================================
# File: smartcash/ui/setup/dependency/handlers/selection_handlers.py - COMPLETE
# Deskripsi: Handlers untuk package selection dan custom package management
# =============================================================================

import ipywidgets as widgets
from typing import Dict, Any, List
from .base_handler import BaseDependencyHandler
from .defaults import get_package_by_key

class SelectionHandler(BaseDependencyHandler):
    """Handler untuk package selection dan custom packages"""
    
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup selection handlers"""
        handlers = {}
        
        # Setup package checkbox handlers
        for key, component in self.ui_components.items():
            if key.startswith('pkg_') and hasattr(component, 'observe'):
                component.observe(self.on_package_selection_change, names='value')
        
        # Setup custom package button handler
        add_custom_btn = self.ui_components.get('add_custom_btn')
        if add_custom_btn:
            add_custom_btn.on_click(self.add_custom_package)
            handlers['add_custom'] = self.add_custom_package
        
        handlers['package_selection'] = self.on_package_selection_change
        return handlers
    
    def on_package_selection_change(self, change):
        """Handle package checkbox changes"""
        try:
            checkbox = change['owner']
            is_selected = change['new']
            
            # Extract package key dari checkbox description atau identifier
            package_key = None
            for key, component in self.ui_components.items():
                if component is checkbox and key.startswith('pkg_'):
                    package_key = key.replace('pkg_', '')
                    break
            
            if package_key:
                package = get_package_by_key(package_key)
                if package:
                    status = "dipilih ✅" if is_selected else "batal dipilih ❌"
                    self.log_info(f"📝 {package['name']} {status}")
                    
                    # Update selected packages counter
                    self._update_selection_counter()
                else:
                    self.log_warning(f"⚠️ Package {package_key} tidak ditemukan")
                    
        except Exception as e:
            self.log_error(f"❌ Error package selection: {str(e)}")
    
    def add_custom_package(self, *args):
        """Add custom package dari input"""
        try:
            custom_input = self.ui_components.get('custom_packages_input')
            custom_list = self.ui_components.get('custom_packages_list')
            
            if not custom_input or not custom_input.value.strip():
                self.log_warning("⚠️ Input custom package kosong")
                return
            
            custom_packages = [pkg.strip() for pkg in custom_input.value.split(',') if pkg.strip()]
            
            if not custom_packages:
                self.log_warning("⚠️ Format custom package tidak valid")
                return
            
            # Validate package names
            valid_packages = []
            for pkg in custom_packages:
                if self._validate_package_name(pkg):
                    valid_packages.append(pkg)
                else:
                    self.log_warning(f"⚠️ Package name tidak valid: {pkg}")
            
            if valid_packages:
                # Update custom packages list display
                self._update_custom_packages_display(valid_packages, custom_list)
                
                # Clear input
                custom_input.value = ""
                
                # Log success
                self.log_success(f"✅ {len(valid_packages)} custom package ditambahkan")
            else:
                self.log_error("❌ Tidak ada package valid yang ditambahkan")
                
        except Exception as e:
            self.log_error(f"❌ Error add custom package: {str(e)}")
    
    def remove_custom_package(self, package_name: str):
        """Remove custom package dari list"""
        try:
            custom_list = self.ui_components.get('custom_packages_list')
            if custom_list:
                # Get current custom packages
                current_packages = self._get_current_custom_packages()
                if package_name in current_packages:
                    current_packages.remove(package_name)
                    self._update_custom_packages_display(current_packages, custom_list)
                    self.log_success(f"✅ {package_name} dihapus dari custom packages")
                else:
                    self.log_warning(f"⚠️ {package_name} tidak ditemukan dalam list")
        except Exception as e:
            self.log_error(f"❌ Error remove custom package: {str(e)}")
    
    def _validate_package_name(self, package_name: str) -> bool:
        """Validate package name format"""
        if not package_name or len(package_name.strip()) < 2:
            return False
        
        # Basic validation untuk pip package name
        import re
        pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$'
        base_name = package_name.split('==')[0].split('>=')[0].split('<=')[0].strip()
        
        return bool(re.match(pattern, base_name))
    
    def _update_custom_packages_display(self, packages: List[str], custom_list_widget):
        """Update custom packages display"""
        if not packages:
            custom_list_widget.value = "<div style='margin-top: 10px; color: #6c757d;'>Belum ada custom package</div>"
            return
        
        package_items = []
        for pkg in packages:
            package_items.append(f"""
            <div style='display: flex; justify-content: space-between; align-items: center; 
                        padding: 8px; margin: 4px 0; background: #e3f2fd; border-radius: 4px;'>
                <span style='font-family: monospace; color: #1976d2;'>{pkg}</span>
                <button onclick='remove_custom_package("{pkg}")' 
                        style='background: #f44336; color: white; border: none; border-radius: 3px; 
                               padding: 2px 8px; cursor: pointer; font-size: 0.8em;'>✕</button>
            </div>
            """)
        
        custom_list_widget.value = f"""
        <div style='margin-top: 10px;'>
            <h5 style='margin: 0 0 10px 0; color: #333;'>Custom Packages ({len(packages)})</h5>
            {''.join(package_items)}
        </div>
        """
    
    def _get_current_custom_packages(self) -> List[str]:
        """Get current custom packages dari display"""
        custom_input = self.ui_components.get('custom_packages_input')
        if custom_input and custom_input.value:
            return [pkg.strip() for pkg in custom_input.value.split(',') if pkg.strip()]
        return []
    
    def _update_selection_counter(self):
        """Update counter untuk selected packages"""
        selected_count = 0
        total_count = 0
        
        for key, component in self.ui_components.items():
            if key.startswith('pkg_') and hasattr(component, 'value'):
                total_count += 1
                if component.value:
                    selected_count += 1
        
        # Update status panel dengan counter
        status_panel = self.ui_components.get('status_panel')
        if status_panel:
            counter_msg = f"📦 {selected_count}/{total_count} packages dipilih"
            # Hanya update jika ada perubahan
            if counter_msg not in str(status_panel.value):
                from smartcash.ui.components.status_panel import update_status_panel
                update_status_panel(status_panel, counter_msg, 'info')

# Factory function
def setup_selection_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup selection handlers untuk dependency management"""
    handler = SelectionHandler(ui_components)
    return handler.setup_handlers()