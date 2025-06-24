"""
File: smartcash/ui/setup/dependency/handlers/config_extractor.py
Deskripsi: Config extractor untuk dependency installer dengan logging terstandarisasi
"""

from typing import Dict, Any, List
from smartcash.ui.setup.dependency.utils import with_logging, LogLevel, get_selected_packages

@with_logging("Extract Dependency Config", LogLevel.DEBUG)
def extract_dependency_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract konfigurasi dependency installer dari UI components"""
    # Pastikan ui_components adalah dictionary
    if not isinstance(ui_components, dict):
        ui_components = {}
        
    # Get selected packages menggunakan package selector utils
    selected_packages = get_selected_packages(ui_components)
    
    # Extract UI settings dengan safe attribute access - one-liner
    custom_packages_text = ''
    if 'custom_packages' in ui_components and hasattr(ui_components['custom_packages'], 'value'):
        custom_packages_text = str(ui_components['custom_packages'].value).strip()
        
    auto_analyze_enabled = True
    if 'auto_analyze_checkbox' in ui_components and hasattr(ui_components['auto_analyze_checkbox'], 'value'):
        auto_analyze_enabled = bool(ui_components['auto_analyze_checkbox'].value)
    
    # Extract installation settings dengan safe fallback - one-liner
    config = ui_components.get('config', {}) if isinstance(ui_components, dict) else {}
    installation_config = config.get('installation', _get_default_installation_config()) if isinstance(config, dict) else _get_default_installation_config()
    analysis_config = config.get('analysis', _get_default_analysis_config()) if isinstance(config, dict) else _get_default_analysis_config()
    
    return {
        'selected_packages': selected_packages,
        'custom_packages': custom_packages_text,
        'auto_analyze': auto_analyze_enabled,
        'installation': installation_config,
        'analysis': analysis_config,
        'module_name': 'dependency',
        'version': '1.0.0',
        'extracted_at': _get_current_timestamp()
    }
        

@with_logging("Get Default Installation Config", LogLevel.DEBUG, with_status=False)
def _get_default_installation_config() -> Dict[str, Any]:
    """Get default installation config"""
    return {
        'parallel_workers': 3,
        'force_reinstall': False,
        'use_cache': True,
        'timeout': 300
    }

@with_logging("Get Default Analysis Config", LogLevel.DEBUG, with_status=False)
def _get_default_analysis_config() -> Dict[str, Any]:
    """Get default analysis config"""
    return {
        'check_compatibility': True,
        'include_dev_deps': False
    }

@with_logging(level=LogLevel.DEBUG, with_status=False)
def _get_current_timestamp() -> str:
    """Get current timestamp"""
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

@with_logging("Validate Extracted Config", LogLevel.DEBUG)
def validate_extracted_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate extracted config dengan comprehensive check"""
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate selected packages - one-liner check
    selected_packages = config.get('selected_packages', [])
    if not selected_packages and not config.get('custom_packages', '').strip():
        validation_results['warnings'].append("Tidak ada packages yang dipilih untuk instalasi")
    
    # Validate installation config - one-liner validation
    installation_config = config.get('installation', {})
    parallel_workers = installation_config.get('parallel_workers', 3)
    timeout = installation_config.get('timeout', 300)
    
    if not isinstance(parallel_workers, int) or parallel_workers < 1 or parallel_workers > 10:
        validation_results['errors'].append("parallel_workers harus antara 1-10")
        validation_results['valid'] = False
    
    if not isinstance(timeout, int) or timeout < 60 or timeout > 1800:
        validation_results['errors'].append("timeout harus antara 60-1800 detik")
        validation_results['valid'] = False
    
    # Validate custom packages format - one-liner check
    custom_packages = config.get('custom_packages', '').strip()
    if custom_packages:
        invalid_packages = [pkg for pkg in custom_packages.split('\n') 
                           if pkg.strip() and not _is_valid_package_name(pkg.strip())]
        if invalid_packages:
            validation_results['warnings'].extend([f"Format package tidak valid: {pkg}" for pkg in invalid_packages])
    
    return validation_results

@with_logging(level=LogLevel.DEBUG, with_status=False)
def _is_valid_package_name(package_name: str) -> bool:
    """Check apakah package name valid"""
    import re
    return bool(re.match(r'^[a-zA-Z0-9_\-\.]+([<>=!~]+.+)?$', package_name.strip()))

@with_logging("Get Config Summary", LogLevel.DEBUG, with_status=False)
def get_config_summary(config: Dict[str, Any]) -> str:
    """Get config summary untuk display"""
    selected = len(config.get('selected_packages', []))
    custom = len([p for p in config.get('custom_packages', '').split('\n') if p.strip()])
    total_packages = selected + custom
    auto_analyze = config.get('auto_analyze', True)
    
    return f"📊 Config: {total_packages} packages, auto-analyze: {'✅' if auto_analyze else '❌'}"

@with_logging("Extract Package List Only", LogLevel.DEBUG)
def extract_package_list_only(ui_components: Dict[str, Any]) -> List[str]:
    """Extract package list saja untuk quick operations"""
    config = extract_dependency_config(ui_components)
    selected = get_selected_packages(ui_components)
    custom = [pkg.strip() for pkg in getattr(ui_components.get('custom_packages'), 'value', '').split('\n') if pkg.strip()]
    return selected + custom