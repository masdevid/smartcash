"""
File: smartcash/ui/setup/dependency/components/package_selector.py
Deskripsi: Utility functions untuk package selection dari UI components
"""

from typing import Dict, Any, List

def get_selected_packages(ui_components: Dict[str, Any]) -> List[str]:
    """Extract selected packages dari UI components"""
    selected_packages = []
    
    try:
        if 'dependency_tabs' not in ui_components:
            return selected_packages
        
        tabs = ui_components['dependency_tabs']
        
        if not hasattr(tabs, 'children') or len(tabs.children) == 0:
            return selected_packages
        
        # Get first tab (package categories)
        categories_tab = tabs.children[0]
        
        # Find all package widgets with selection state
        package_widgets = find_package_widgets(categories_tab)
        
        for widget in package_widgets:
            if hasattr(widget, 'selection_checkbox') and hasattr(widget, 'package_info'):
                if widget.selection_checkbox.value:  # Is selected
                    selected_packages.append(widget.package_info['name'])
        
    except Exception as e:
        print(f"❌ Error extracting selected packages: {e}")
    
    return selected_packages

def get_custom_packages_text(ui_components: Dict[str, Any]) -> str:
    """Get custom packages text dari UI"""
    try:
        if 'dependency_tabs' not in ui_components:
            return ""
        
        tabs = ui_components['dependency_tabs']
        
        if not hasattr(tabs, 'children') or len(tabs.children) < 2:
            return ""
        
        # Get second tab (custom packages)
        custom_tab = tabs.children[1]
        
        if hasattr(custom_tab, 'packages_textarea'):
            return custom_tab.packages_textarea.value.strip()
        
        return ""
        
    except Exception as e:
        print(f"❌ Error getting custom packages text: {e}")
        return ""

def find_package_widgets(container, widgets_list=None):
    """Recursively find package widgets dalam container"""
    if widgets_list is None:
        widgets_list = []
    
    if hasattr(container, 'package_info'):
        widgets_list.append(container)
    
    if hasattr(container, 'children'):
        for child in container.children:
            find_package_widgets(child, widgets_list)
    
    return widgets_list

def update_package_selection(ui_components: Dict[str, Any], package_name: str, selected: bool) -> bool:
    """Update selection state untuk specific package"""
    try:
        if 'dependency_tabs' not in ui_components:
            return False
        
        tabs = ui_components['dependency_tabs']
        
        if not hasattr(tabs, 'children') or len(tabs.children) == 0:
            return False
        
        categories_tab = tabs.children[0]
        package_widgets = find_package_widgets(categories_tab)
        
        for widget in package_widgets:
            if (hasattr(widget, 'package_info') and 
                hasattr(widget, 'selection_checkbox') and 
                widget.package_info['name'] == package_name):
                
                widget.selection_checkbox.value = selected
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error updating package selection for {package_name}: {e}")
        return False