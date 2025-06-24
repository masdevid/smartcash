"""
File: smartcash/ui/setup/dependency/handlers/config_extractor.py
Deskripsi: Helper untuk ekstrak nilai dari UI components ke struktur config
"""

from typing import Dict, Any, List
from smartcash.ui.setup.dependency.utils import get_selected_packages

def extract_dependency_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI components dengan error handling"""
    try:
        config = {
            'module_name': 'dependency',
            'selected_packages': _extract_selected_packages(ui_components),
            'custom_packages': _extract_custom_packages(ui_components),
            'auto_analyze': _extract_auto_analyze(ui_components),
            'installation': _extract_installation_settings(ui_components),
            'analysis': _extract_analysis_settings(ui_components),
            'ui_settings': _extract_ui_settings(ui_components),
            'advanced': _extract_advanced_settings(ui_components)
        }
        return config
    except Exception as e:
        from smartcash.ui.utils.ui_logger import get_default_logger
        logger = get_default_logger()
        logger.warning(f"⚠️ Config extraction error: {str(e)}")
        return {}

def _extract_selected_packages(ui_components: Dict[str, Any]) -> List[str]:
    """Extract selected packages dari package selector"""
    try:
        return get_selected_packages(ui_components.get('package_selector'))
    except:
        return []

def _extract_custom_packages(ui_components: Dict[str, Any]) -> str:
    """Extract custom packages dari textarea"""
    try:
        widget = ui_components.get('custom_packages')
        return widget.value if widget else ''
    except:
        return ''

def _extract_auto_analyze(ui_components: Dict[str, Any]) -> bool:
    """Extract auto analyze setting"""
    try:
        widget = ui_components.get('auto_analyze_checkbox')
        return widget.value if widget else True
    except:
        return True

def _extract_installation_settings(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract installation settings dari UI"""
    return {
        'parallel_workers': _safe_extract_int(ui_components, 'parallel_workers', 3),
        'force_reinstall': _safe_extract_bool(ui_components, 'force_reinstall', False),
        'use_cache': _safe_extract_bool(ui_components, 'use_cache', True),
        'timeout': _safe_extract_int(ui_components, 'timeout', 300),
        'max_retries': _safe_extract_int(ui_components, 'max_retries', 2),
        'retry_delay': _safe_extract_float(ui_components, 'retry_delay', 1.0)
    }

def _extract_analysis_settings(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract analysis settings dari UI"""
    return {
        'check_compatibility': _safe_extract_bool(ui_components, 'check_compatibility', True),
        'include_dev_deps': _safe_extract_bool(ui_components, 'include_dev_deps', False),
        'batch_size': _safe_extract_int(ui_components, 'batch_size', 10),
        'detailed_info': _safe_extract_bool(ui_components, 'detailed_info', True)
    }

def _extract_ui_settings(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract UI settings"""
    return {
        'auto_analyze_on_render': _safe_extract_bool(ui_components, 'auto_analyze_on_render', True),
        'show_progress': _safe_extract_bool(ui_components, 'show_progress', True),
        'log_level': _safe_extract_str(ui_components, 'log_level', 'info'),
        'compact_view': _safe_extract_bool(ui_components, 'compact_view', False)
    }

def _extract_advanced_settings(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract advanced settings"""
    return {
        'pip_extra_args': _safe_extract_list(ui_components, 'pip_extra_args', []),
        'environment_variables': _safe_extract_dict(ui_components, 'environment_variables', {}),
        'pre_install_commands': _safe_extract_list(ui_components, 'pre_install_commands', []),
        'post_install_commands': _safe_extract_list(ui_components, 'post_install_commands', [])
    }

def _safe_extract_int(ui_components: Dict[str, Any], key: str, default: int) -> int:
    """Safe extract integer value"""
    try:
        widget = ui_components.get(key)
        return int(widget.value) if widget else default
    except:
        return default

def _safe_extract_bool(ui_components: Dict[str, Any], key: str, default: bool) -> bool:
    """Safe extract boolean value"""
    try:
        widget = ui_components.get(key)
        return bool(widget.value) if widget else default
    except:
        return default

def _safe_extract_float(ui_components: Dict[str, Any], key: str, default: float) -> float:
    """Safe extract float value"""
    try:
        widget = ui_components.get(key)
        return float(widget.value) if widget else default
    except:
        return default

def _safe_extract_str(ui_components: Dict[str, Any], key: str, default: str) -> str:
    """Safe extract string value"""
    try:
        widget = ui_components.get(key)
        return str(widget.value) if widget else default
    except:
        return default

def _safe_extract_list(ui_components: Dict[str, Any], key: str, default: List) -> List:
    """Safe extract list value"""
    try:
        widget = ui_components.get(key)
        if widget and hasattr(widget, 'value'):
            if isinstance(widget.value, str):
                return [x.strip() for x in widget.value.split('\n') if x.strip()]
            return list(widget.value) if widget.value else default
        return default
    except:
        return default

def _safe_extract_dict(ui_components: Dict[str, Any], key: str, default: Dict) -> Dict:
    """Safe extract dict value"""
    try:
        widget = ui_components.get(key)
        return dict(widget.value) if widget and widget.value else default
    except:
        return default

def validate_extracted_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate extracted config structure"""
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required keys
    required_keys = ['module_name', 'selected_packages', 'installation', 'analysis']
    for key in required_keys:
        if key not in config:
            validation_result['errors'].append(f"Missing required key: {key}")
            validation_result['valid'] = False
    
    # Validate installation settings
    if 'installation' in config:
        inst = config['installation']
        if inst.get('parallel_workers', 0) < 1:
            validation_result['warnings'].append("parallel_workers should be >= 1")
        if inst.get('timeout', 0) < 60:
            validation_result['warnings'].append("timeout should be >= 60 seconds")
    
    return validation_result

def get_config_summary(config: Dict[str, Any]) -> str:
    """Get human-readable config summary"""
    try:
        selected_count = len(config.get('selected_packages', []))
        custom_packages = config.get('custom_packages', '').strip()
        custom_count = len([x for x in custom_packages.split('\n') if x.strip()]) if custom_packages else 0
        
        return f"📦 {selected_count} packages selected, {custom_count} custom packages"
    except:
        return "📦 Config summary unavailable"