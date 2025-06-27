
# =============================================================================
# File: smartcash/ui/setup/dependency/handlers/config_extractor.py
# Deskripsi: Extract konfigurasi dari UI components
# =============================================================================

from typing import Dict, Any, List

def extract_dependency_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari dependency UI components"""
    config = {'module_name': 'dependency', 'version': '1.0.0', 'selected_packages': [], 'custom_packages': ''}
    
    # Extract selected packages
    selected_packages = []
    for key, component in ui_components.items():
        if key.startswith('pkg_') and hasattr(component, 'value') and component.value:
            selected_packages.append(key.replace('pkg_', ''))
    config['selected_packages'] = selected_packages
    
    # Extract custom packages
    if 'custom_packages_input' in ui_components:
        config['custom_packages'] = ui_components['custom_packages_input'].value or ''
    
    return config