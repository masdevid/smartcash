"""
File: smartcash/ui/setup/dependency/handlers/config_updater.py
Deskripsi: Helper untuk update UI components dari config values
"""

from typing import Dict, Any, List
from smartcash.ui.setup.dependency.utils import reset_package_selections

def update_dependency_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config dengan error handling"""
    try:
        _update_package_selections(ui_components, config)
        _update_custom_packages(ui_components, config)
        _update_installation_settings(ui_components, config)
        _update_analysis_settings(ui_components, config)
        _update_ui_settings(ui_components, config)
        _update_advanced_settings(ui_components, config)
    except Exception as e:
        from smartcash.ui.utils.ui_logger import get_default_logger
        logger = get_default_logger()
        logger.warning(f"⚠️ UI update error: {str(e)}")

def reset_dependency_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke default values"""
    from .defaults import get_default_dependency_config
    default_config = get_default_dependency_config()
    update_dependency_ui(ui_components, default_config)

def apply_config_to_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Apply config to UI (alias untuk update_dependency_ui)"""
    update_dependency_ui(ui_components, config)

def _update_package_selections(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update package selector dari config"""
    try:
        selected_packages = config.get('selected_packages', [])
        package_selector = ui_components.get('package_selector')
        
        if package_selector and selected_packages:
            # Reset dulu semua selections
            reset_package_selections(package_selector)
            
            # Set selected packages
            from smartcash.ui.setup.dependency.utils import update_package_status
            for package in selected_packages:
                update_package_status(package_selector, package, True)
    except Exception as e:
        pass  # Silent fail untuk package selection

def _update_custom_packages(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update custom packages textarea"""
    _safe_update_widget(ui_components, 'custom_packages', config.get('custom_packages', ''))

def _update_installation_settings(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update installation settings widgets"""
    installation = config.get('installation', {})
    
    _safe_update_widget(ui_components, 'parallel_workers', installation.get('parallel_workers', 3))
    _safe_update_widget(ui_components, 'force_reinstall', installation.get('force_reinstall', False))
    _safe_update_widget(ui_components, 'use_cache', installation.get('use_cache', True))
    _safe_update_widget(ui_components, 'timeout', installation.get('timeout', 300))
    _safe_update_widget(ui_components, 'max_retries', installation.get('max_retries', 2))
    _safe_update_widget(ui_components, 'retry_delay', installation.get('retry_delay', 1.0))

def _update_analysis_settings(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update analysis settings widgets"""
    analysis = config.get('analysis', {})
    
    _safe_update_widget(ui_components, 'check_compatibility', analysis.get('check_compatibility', True))
    _safe_update_widget(ui_components, 'include_dev_deps', analysis.get('include_dev_deps', False))
    _safe_update_widget(ui_components, 'batch_size', analysis.get('batch_size', 10))
    _safe_update_widget(ui_components, 'detailed_info', analysis.get('detailed_info', True))

def _update_ui_settings(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI settings widgets"""
    ui_settings = config.get('ui_settings', {})
    
    _safe_update_widget(ui_components, 'auto_analyze_on_render', ui_settings.get('auto_analyze_on_render', True))
    _safe_update_widget(ui_components, 'show_progress', ui_settings.get('show_progress', True))
    _safe_update_widget(ui_components, 'log_level', ui_settings.get('log_level', 'info'))
    _safe_update_widget(ui_components, 'compact_view', ui_settings.get('compact_view', False))

def _update_advanced_settings(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update advanced settings widgets"""
    advanced = config.get('advanced', {})
    
    # Convert list ke string untuk textarea
    pip_args = '\n'.join(advanced.get('pip_extra_args', []))
    _safe_update_widget(ui_components, 'pip_extra_args', pip_args)
    
    pre_commands = '\n'.join(advanced.get('pre_install_commands', []))
    _safe_update_widget(ui_components, 'pre_install_commands', pre_commands)
    
    post_commands = '\n'.join(advanced.get('post_install_commands', []))
    _safe_update_widget(ui_components, 'post_install_commands', post_commands)

def _safe_update_widget(ui_components: Dict[str, Any], key: str, value: Any) -> None:
    """Safely update widget value dengan error handling"""
    try:
        widget = ui_components.get(key)
        if widget and hasattr(widget, 'value'):
            widget.value = value
    except Exception:
        pass  # Silent fail untuk individual widget update

def get_ui_update_summary(config: Dict[str, Any]) -> str:
    """Get summary of UI updates yang akan dilakukan"""
    try:
        updates = []
        
        if config.get('selected_packages'):
            updates.append(f"📦 {len(config['selected_packages'])} packages")
            
        if config.get('custom_packages'):
            custom_count = len([x for x in config['custom_packages'].split('\n') if x.strip()])
            updates.append(f"➕ {custom_count} custom packages")
            
        if config.get('installation'):
            inst = config['installation']
            updates.append(f"⚙️ Workers: {inst.get('parallel_workers', 3)}")
            
        return ', '.join(updates) if updates else "No updates"
    except:
        return "Update summary unavailable"