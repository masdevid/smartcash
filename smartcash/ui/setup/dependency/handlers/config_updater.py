
# =============================================================================
# File: smartcash/ui/setup/dependency/handlers/config_updater.py
# Deskripsi: Update UI dari konfigurasi yang dimuat
# =============================================================================

from typing import Dict, Any

def update_dependency_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update dependency UI dari config"""
    selected_packages = config.get('selected_packages', [])
    custom_packages = config.get('custom_packages', '')
    
    # Update checkboxes
    for key, component in ui_components.items():
        if key.startswith('pkg_') and hasattr(component, 'value'):
            package_key = key.replace('pkg_', '')
            component.value = package_key in selected_packages
    
    # Update custom input
    if 'custom_packages_input' in ui_components:
        ui_components['custom_packages_input'].value = custom_packages

def reset_dependency_ui(ui_components: Dict[str, Any]) -> None:
    """Reset dependency UI ke default"""
    from .defaults import get_default_dependency_config
    default_config = get_default_dependency_config()
    update_dependency_ui(ui_components, default_config)