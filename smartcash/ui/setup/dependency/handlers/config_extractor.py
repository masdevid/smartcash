"""
File: smartcash/ui/setup/dependency/handlers/config_extractor.py

Configuration Extractor for Dependency Management UI.

This module provides functionality to extract and validate configuration
from UI components in the dependency management interface.
"""

# Standard library imports
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, overload, TypedDict
from functools import partial
import copy

# Absolute imports
from smartcash.ui.setup.dependency.utils.ui.utils import get_selected_packages
from smartcash.common.logger import get_logger

# Type aliases
ConfigDict = Dict[str, Any]
UIComponents = Dict[str, Any]

# Logger
logger = get_logger(__name__)

# Type variable for generic type hinting
T = TypeVar('T', int, float, bool, str, list, dict)

def extract_dependency_config(ui_components: UIComponents) -> ConfigDict:
    """Extract and validate configuration from UI components.
    
    This is the main entry point for extracting configuration from the UI.
    It handles all the extraction logic and ensures the returned config
    is properly structured and validated.
    
    Args:
        ui_components: Dictionary of UI components to extract values from
        
    Returns:
        ConfigDict containing the extracted and validated configuration
    """
    if not isinstance(ui_components, dict):
        logger.warning("Invalid UI components provided, using empty dict")
        ui_components = {}
    
    try:
        # Define all config sections to extract
        config_sections = {
            'module_name': 'dependency',
            'selected_packages': _extract_selected_packages(ui_components),
            'custom_packages': _extract_custom_packages(ui_components),
            'auto_analyze': _extract_auto_analyze(ui_components),
            'installation': _extract_installation_settings(ui_components),
            'analysis': _extract_analysis_settings(ui_components),
            'ui_settings': _extract_ui_settings(ui_components),
            'advanced': _extract_advanced_settings(ui_components)
        }
        
        return validate_extracted_config(config_sections)
        
    except Exception as e:
        logger.error(f"Error extracting config: {str(e)}", exc_info=True)
        return _get_fallback_config()

def _get_fallback_config() -> ConfigDict:
    """Return a minimal valid configuration as fallback."""
    return {
        'module_name': 'dependency',
        'selected_packages': [],
        'custom_packages': '',
        'auto_analyze': False,
        'installation': _extract_installation_settings({}),
        'analysis': _extract_analysis_settings({}),
        'ui_settings': _extract_ui_settings({}),
        'advanced': _extract_advanced_settings({})
    }

def _extract_widget_value(widget: Any, default: Any = None) -> Any:
    """Safely extract value from a widget with a default fallback.
    
    Args:
        widget: The widget to extract value from
        default: Default value if extraction fails
        
    Returns:
        The widget value or default if extraction fails
    """
    try:
        return widget.value if hasattr(widget, 'value') else default
    except Exception:
        return default

def _extract_selected_packages(ui_components: Dict[str, Any]) -> List[str]:
    """Extract selected packages from package selector.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        List of selected package names
    """
    try:
        return get_selected_packages(ui_components.get('package_selector'))
    except Exception:
        return []

def _extract_custom_packages(ui_components: Dict[str, Any]) -> str:
    """Extract custom packages from textarea.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        String containing custom packages
    """
    return _extract_widget_value(ui_components.get('custom_packages'), '')

def _extract_auto_analyze(ui_components: Dict[str, Any]) -> bool:
    """Extract auto analyze setting.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        Boolean indicating if auto analyze is enabled
    """
    return _extract_widget_value(ui_components.get('auto_analyze_checkbox'), True)

def _create_extractor(
    settings_map: Dict[str, tuple[type, Any]]
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a settings extractor function from a settings mapping.
    
    Args:
        settings_map: Dictionary mapping setting names to (type, default) tuples
        
    Returns:
        A function that extracts settings from UI components
    """
    def extractor(ui_components: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, (value_type, default) in settings_map.items():
            if value_type == int:
                result[key] = _safe_extract(ui_components, key, default, int)
            elif value_type == bool:
                result[key] = _safe_extract(ui_components, key, default, bool)
            elif value_type == float:
                result[key] = _safe_extract(ui_components, key, default, float)
            elif value_type == str:
                result[key] = _safe_extract(ui_components, key, default, str)
            else:
                result[key] = _extract_widget_value(ui_components.get(key), default)
        return result
    return extractor

# Define settings extractors using the factory function
_extract_installation_settings = _create_extractor({
    'parallel_workers': (int, 3),
    'force_reinstall': (bool, False),
    'use_cache': (bool, True),
    'timeout': (int, 300),
    'max_retries': (int, 2),
    'retry_delay': (float, 1.0)
})

_extract_analysis_settings = _create_extractor({
    'check_compatibility': (bool, True),
    'include_dev_deps': (bool, False),
    'batch_size': (int, 10),
    'detailed_info': (bool, True)
})

_extract_ui_settings = _create_extractor({
    'auto_analyze_on_render': (bool, True),
    'show_progress': (bool, True),
    'log_level': (str, 'info'),
    'compact_view': (bool, False)
})

def _extract_advanced_settings(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract advanced settings from UI components.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        Dictionary with advanced settings
    """
    return _create_extractor({
        'debug_mode': (bool, False),
        'verbose_logging': (bool, False),
        'experimental_features': (bool, False),
        'post_install_commands': (list, [])
    })(ui_components)

def _safe_extract(
    ui_components: Dict[str, Any], 
    key: str, 
    default: T, 
    value_type: type = None
) -> Union[int, float, bool, str, list, dict, T]:
    """Safely extract and convert a value from UI components.
    
    Args:
        ui_components: Dictionary containing UI components
        key: Key of the component to extract
        default: Default value if extraction fails
        value_type: Type to convert the extracted value to
        
    Returns:
        Extracted and converted value or default if extraction fails
    """
    try:
        widget = ui_components.get(key)
        if not widget or not hasattr(widget, 'value'):
            return default
            
        value = widget.value
        if value is None:
            return default
            
        if value_type is None:
            return value
            
        if value_type == bool:
            return bool(value)
        elif value_type == int:
            return int(float(value))  # Handle both int and float strings
        elif value_type == float:
            return float(value)
        elif value_type == str:
            return str(value)
        elif value_type == list:
            return list(value) if hasattr(value, '__iter__') and not isinstance(value, str) else [value]
        elif value_type == dict:
            return dict(value) if hasattr(value, 'items') else {}
            
        return value
    except (ValueError, AttributeError, TypeError):
        return default

# Type-specific extractors for backward compatibility
_safe_extract_int = partial(_safe_extract, value_type=int)
_safe_extract_bool = partial(_safe_extract, value_type=bool)
_safe_extract_float = partial(_safe_extract, value_type=float)
_safe_extract_str = partial(_safe_extract, value_type=str)
_safe_extract_list = partial(_safe_extract, value_type=list)
_safe_extract_dict = partial(_safe_extract, value_type=dict)

def validate_extracted_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the extracted configuration.
    
    This function ensures that the extracted configuration has all required fields
    and that their values are within expected ranges. It also applies default values
    for any missing fields.
    
    Args:
        config: The raw extracted configuration
        
    Returns:
        Validated and normalized configuration dictionary
    """
    if not isinstance(config, dict):
        return {}
    
    # Define default configuration structure
    default_config = {
        'module_name': 'dependency',
        'selected_packages': [],
        'custom_packages': '',
        'auto_analyze': True,
        'installation': {
            'parallel_workers': 3,
            'force_reinstall': False,
            'use_cache': True,
            'timeout': 300,
            'max_retries': 2,
            'retry_delay': 1.0
        },
        'analysis': {
            'check_compatibility': True,
            'include_dev_deps': False,
            'batch_size': 10,
            'detailed_info': True
        },
        'ui_settings': {
            'auto_analyze_on_render': True,
            'show_progress': True,
            'log_level': 'info',
            'compact_view': False
        },
        'advanced': {
            'debug_mode': False,
            'verbose_logging': False,
            'experimental_features': False,
            'post_install_commands': []
        }
    }
    
    # Helper function to recursively merge dictionaries
    def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    # Merge the provided config with defaults
    return deep_merge(default_config, config)

def get_config_summary(config: Dict[str, Any]) -> str:
    """Generate a human-readable summary of the configuration.
    
    This function creates a formatted string that provides a clear overview
    of all the configuration settings in a user-friendly way.
    
    Args:
        config: The configuration dictionary to summarize
        
    Returns:
        Formatted string containing the configuration summary
    """
    try:
        # Validate config first to ensure all required fields exist
        valid_config = validate_extracted_config(config)
        
        # Helper function to format boolean values consistently
        def fmt_bool(value: bool) -> str:
            """Format boolean value with emoji for better readability."""
            return "✅ Yes" if value else "❌ No"
        
        # Start building the summary
        summary = ["📋 Configuration Summary", "=" * 40]
        
        # Basic package information
        selected_count = len(valid_config.get('selected_packages', []))
        custom_pkgs = valid_config.get('custom_packages', '')
        custom_count = len([pkg for pkg in custom_pkgs.split('\n') if pkg.strip()])
        
        summary.append(f"\n🔹 Selected Packages: {selected_count}")
        if custom_count > 0:
            summary.append(f"🔹 Custom Packages: {custom_count}")
        
        # Installation settings
        install = valid_config.get('installation', {})
        summary.extend([
            "\n⚙️ Installation Settings",
            f"  • Parallel Workers: {install.get('parallel_workers', 3)}",
            f"  • Force Reinstall: {fmt_bool(install.get('force_reinstall', False))}",
            f"  • Use Cache: {fmt_bool(install.get('use_cache', True))}",
            f"  • Timeout: {install.get('timeout', 300)}s",
            f"  • Max Retries: {install.get('max_retries', 2)}",
            f"  • Retry Delay: {install.get('retry_delay', 1.0)}s"
        ])
        
        # Analysis settings
        analysis = valid_config.get('analysis', {})
        summary.extend([
            "\n🔍 Analysis Settings",
            f"  • Check Compatibility: {fmt_bool(analysis.get('check_compatibility', True))}",
            f"  • Include Dev Dependencies: {fmt_bool(analysis.get('include_dev_deps', False))}",
            f"  • Batch Size: {analysis.get('batch_size', 10)}",
            f"  • Detailed Info: {fmt_bool(analysis.get('detailed_info', True))}"
        ])
        
        # UI Settings
        ui = valid_config.get('ui_settings', {})
        summary.extend([
            "\n🖥️ UI Settings",
            f"  • Auto-analyze on Render: {fmt_bool(ui.get('auto_analyze_on_render', True))}",
            f"  • Show Progress: {fmt_bool(ui.get('show_progress', True))}",
            f"  • Log Level: {ui.get('log_level', 'info').upper()}",
            f"  • Compact View: {fmt_bool(ui.get('compact_view', False))}"
        ])
        
        # Advanced settings (only show non-default values)
        advanced = valid_config.get('advanced', {})
        advanced_settings = []
        
        for key, value in advanced.items():
            if isinstance(value, bool) and value:
                advanced_settings.append(f"  • {key.replace('_', ' ').title()}: {fmt_bool(value)}")
            elif isinstance(value, list) and value:
                advanced_settings.append(f"  • {key.replace('_', ' ').title()}: {len(value)} items")
            elif value:  # Any other non-empty, non-default value
                advanced_settings.append(f"  • {key.replace('_', ' ').title()}: {value}")
        
        if advanced_settings:
            summary.append("\n⚡ Advanced Settings")
            summary.extend(advanced_settings)
        
        return "\n".join(summary)
        
    except Exception as e:
        logger = get_default_logger()
        logger.error(f"Error generating config summary: {str(e)}", exc_info=True)
        return "⚠️ Error generating configuration summary. Please check the logs for details."