"""
File: smartcash/ui/setup/dependency_installer/handlers/config_updater.py
Deskripsi: Config updater untuk dependency installer dengan CommonInitializer pattern (logger sudah built-in)
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.setup.dependency_installer.components.package_selector import reset_package_selections
from smartcash.ui.setup.dependency_installer.utils.ui_state_utils import log_to_ui_safe

def update_dependency_installer_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config dengan comprehensive approach - one-liner style"""
    
    try:
        # Update custom packages text dengan safe attribute setting
        custom_packages = config.get('custom_packages', '')
        ui_components.get('custom_packages') and setattr(ui_components['custom_packages'], 'value', custom_packages)
        
        # Update auto-analyze checkbox dengan safe attribute setting
        auto_analyze = config.get('auto_analyze', True)
        ui_components.get('auto_analyze_checkbox') and setattr(ui_components['auto_analyze_checkbox'], 'value', auto_analyze)
        
        # Update installation settings dengan safe fallback
        installation_config = config.get('installation', {})
        _update_installation_settings(ui_components, installation_config)
        
        # Update analysis settings dengan safe fallback
        analysis_config = config.get('analysis', {})
        _update_analysis_settings(ui_components, analysis_config)
        
        # Update package selections menggunakan package selector
        selected_packages = config.get('selected_packages', [])
        _update_package_selections(ui_components, selected_packages)
        
        # Log update success menggunakan built-in logger dari CommonInitializer
        log_to_ui_safe(ui_components, f"📋 UI updated from config: {_get_update_summary(config)}")
        
    except Exception as e:
        # Log error menggunakan built-in logger
        log_to_ui_safe(ui_components, f"⚠️ Error updating UI from config: {str(e)}", "error")

def _update_installation_settings(ui_components: Dict[str, Any], installation_config: Dict[str, Any]) -> None:
    """Update installation settings dengan safe widget updates - one-liner approach"""
    
    settings_mapping = {
        'parallel_workers_slider': 'parallel_workers',
        'timeout_slider': 'timeout',
        'force_reinstall_checkbox': 'force_reinstall',
        'use_cache_checkbox': 'use_cache'
    }
    
    # Update widgets dengan safe attribute setting - one-liner
    [setattr(ui_components[widget_key], 'value', installation_config.get(config_key, 
                                                   _get_installation_default(config_key)))
     for widget_key, config_key in settings_mapping.items() 
     if widget_key in ui_components and hasattr(ui_components[widget_key], 'value')]

def _update_analysis_settings(ui_components: Dict[str, Any], analysis_config: Dict[str, Any]) -> None:
    """Update analysis settings dengan safe widget updates - one-liner approach"""
    
    settings_mapping = {
        'check_compatibility_checkbox': 'check_compatibility',
        'include_dev_deps_checkbox': 'include_dev_deps'
    }
    
    # Update widgets dengan safe attribute setting - one-liner
    [setattr(ui_components[widget_key], 'value', analysis_config.get(config_key, 
                                                 _get_analysis_default(config_key)))
     for widget_key, config_key in settings_mapping.items() 
     if widget_key in ui_components and hasattr(ui_components[widget_key], 'value')]

def _update_package_selections(ui_components: Dict[str, Any], selected_packages: list) -> None:
    """Update package selections menggunakan package selector utils dengan error handling"""
    
    try:
        # Reset package selections terlebih dahulu
        reset_package_selections(ui_components)
        
        # Set selected packages menggunakan package selector jika tersedia
        if 'set_selected_packages' in ui_components and callable(ui_components['set_selected_packages']):
            ui_components['set_selected_packages'](selected_packages)
        else:
            # Fallback: manual update package checkboxes jika ada
            _manual_update_package_checkboxes(ui_components, selected_packages)
            
    except Exception as e:
        log_to_ui_safe(ui_components, f"⚠️ Error updating package selections: {str(e)}", "error")

def _manual_update_package_checkboxes(ui_components: Dict[str, Any], selected_packages: list) -> None:
    """Manual update package checkboxes sebagai fallback - one-liner approach"""
    
    # Get package keys yang perlu diselect
    from smartcash.ui.setup.dependency_installer.components.package_selector import get_package_categories
    
    try:
        package_categories = get_package_categories()
        selected_package_keys = []
        
        # Map selected packages ke package keys
        for category in package_categories:
            for package in category['packages']:
                if package['pip_name'] in selected_packages:
                    selected_package_keys.append(package['key'])
        
        # Update checkboxes dengan one-liner
        [setattr(ui_components[key], 'value', key in selected_package_keys)
         for key in ui_components.keys() 
         if key.endswith('_checkbox') and hasattr(ui_components.get(key), 'value')]
        
    except Exception as e:
        log_to_ui_safe(ui_components, f"⚠️ Manual package checkbox update failed: {str(e)}", "error")

def reset_dependency_installer_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke default state dengan comprehensive approach - one-liner style"""
    
    try:
        # Reset package selections menggunakan utils
        reset_package_selections(ui_components)
        
        # Reset other UI components dengan safe attribute setting - one-liner
        ui_components.get('custom_packages') and setattr(ui_components['custom_packages'], 'value', '')
        ui_components.get('auto_analyze_checkbox') and setattr(ui_components['auto_analyze_checkbox'], 'value', True)
        
        # Reset installation settings ke default
        _reset_installation_settings(ui_components)
        
        # Reset analysis settings ke default
        _reset_analysis_settings(ui_components)
        
        # Clear status dan outputs
        _clear_ui_state(ui_components)
        
        # Log success menggunakan built-in logger
        log_to_ui_safe(ui_components, "🔄 UI berhasil direset ke default state")
        
    except Exception as e:
        # Log error menggunakan built-in logger
        log_to_ui_safe(ui_components, f"⚠️ Error resetting UI: {str(e)}", "error")

def _reset_installation_settings(ui_components: Dict[str, Any]) -> None:
    """Reset installation settings ke default - one-liner approach"""
    
    default_settings = {
        'parallel_workers_slider': 3,
        'timeout_slider': 300,
        'force_reinstall_checkbox': False,
        'use_cache_checkbox': True
    }
    
    # Reset dengan safe attribute setting - one-liner
    [setattr(ui_components[widget_key], 'value', default_value)
     for widget_key, default_value in default_settings.items() 
     if widget_key in ui_components and hasattr(ui_components[widget_key], 'value')]

def _reset_analysis_settings(ui_components: Dict[str, Any]) -> None:
    """Reset analysis settings ke default - one-liner approach"""
    
    default_settings = {
        'check_compatibility_checkbox': True,
        'include_dev_deps_checkbox': False
    }
    
    # Reset dengan safe attribute setting - one-liner
    [setattr(ui_components[widget_key], 'value', default_value)
     for widget_key, default_value in default_settings.items() 
     if widget_key in ui_components and hasattr(ui_components[widget_key], 'value')]

def _clear_ui_state(ui_components: Dict[str, Any]) -> None:
    """Clear UI state seperti outputs dan progress bars - one-liner approach"""
    
    # Clear outputs dengan safe method calls - one-liner
    [widget.clear_output(wait=True) for key in ['log_output', 'status', 'confirmation_area'] 
     if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]
    
    # Hide progress containers dengan safe attribute setting - one-liner
    [setattr(ui_components[key].layout, 'visibility', 'hidden')
     for key in ['progress_container', 'progress_bar', 'current_progress'] 
     if key in ui_components and hasattr(ui_components[key], 'layout')]

def _get_installation_default(config_key: str) -> Any:
    """Get installation default value - one-liner mapping"""
    defaults = {
        'parallel_workers': 3,
        'timeout': 300,
        'force_reinstall': False,
        'use_cache': True
    }
    return defaults.get(config_key)

def _get_analysis_default(config_key: str) -> Any:
    """Get analysis default value - one-liner mapping"""
    defaults = {
        'check_compatibility': True,
        'include_dev_deps': False
    }
    return defaults.get(config_key)

def _get_update_summary(config: Dict[str, Any]) -> str:
    """Get update summary untuk logging - one-liner"""
    selected_count = len(config.get('selected_packages', []))
    custom_packages = config.get('custom_packages', '').strip()
    custom_count = len([pkg for pkg in custom_packages.split('\n') if pkg.strip()]) if custom_packages else 0
    total_packages = selected_count + custom_count
    
    return f"{total_packages} packages ({selected_count} selected, {custom_count} custom)"

def apply_config_to_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Public interface untuk apply config ke UI - alias untuk update_dependency_installer_ui"""
    update_dependency_installer_ui(ui_components, config)

def get_ui_state_summary(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get current UI state summary untuk debugging - one-liner approach"""
    
    return {
        'custom_packages_count': len([pkg for pkg in getattr(ui_components.get('custom_packages'), 'value', '').split('\n') if pkg.strip()]),
        'auto_analyze': getattr(ui_components.get('auto_analyze_checkbox'), 'value', False),
        'has_log_output': 'log_output' in ui_components,
        'has_status_panel': 'status_panel' in ui_components,
        'has_progress': 'progress_container' in ui_components
    }