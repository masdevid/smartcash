"""
File: smartcash/ui/setup/dependency/handlers/config_extractor.py
Deskripsi: Config extractor untuk dependency installer dengan CommonInitializer pattern (logger sudah built-in)
"""

from typing import Dict, Any
from smartcash.ui.setup.dependency.components.package_selector import get_selected_packages
from smartcash.ui.setup.dependency.utils.ui_state_utils import log_to_ui_safe

def extract_dependency_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract konfigurasi dependency installer dari UI components dengan one-liner approach"""
    
    log_to_ui_safe(ui_components, "🔍 Mengekstrak konfigurasi dari UI...")
    
    try:
        # Get selected packages menggunakan package selector utils
        selected_packages = get_selected_packages(ui_components)
        
        # Extract UI settings dengan safe attribute access - one-liner
        custom_packages_text = getattr(ui_components.get('custom_packages'), 'value', '').strip()
        auto_analyze_enabled = getattr(ui_components.get('auto_analyze_checkbox'), 'value', True)
        
        # Extract installation settings dengan safe fallback - one-liner
        installation_config = ui_components.get('config', {}).get('installation', _get_default_installation_config())
        analysis_config = ui_components.get('config', {}).get('analysis', _get_default_analysis_config())
        
        log_to_ui_safe(ui_components, "✅ Ekstraksi konfigurasi selesai")
        
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
    except Exception as e:
        log_to_ui_safe(ui_components, f"❌ Gagal mengekstrak konfigurasi: {str(e)}", "error")
        return {}
        

def _get_default_installation_config() -> Dict[str, Any]:
    """Get default installation config - one-liner"""
    return {
        'parallel_workers': 3,
        'force_reinstall': False,
        'use_cache': True,
        'timeout': 300
    }

def _get_default_analysis_config() -> Dict[str, Any]:
    """Get default analysis config - one-liner"""
    return {
        'check_compatibility': True,
        'include_dev_deps': False
    }

def _get_current_timestamp() -> str:
    """Get current timestamp - one-liner"""
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def validate_extracted_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate extracted config dengan comprehensive check - one-liner approach"""
    
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

def _is_valid_package_name(package_name: str) -> bool:
    """Check apakah package name valid - one-liner regex"""
    import re
    return bool(re.match(r'^[a-zA-Z0-9_\-\.]+([<>=!~]+.+)?$', package_name.strip()))

def get_config_summary(config: Dict[str, Any]) -> str:
    """Get config summary untuk display - one-liner"""
    selected_count = len(config.get('selected_packages', []))
    custom_count = len([pkg for pkg in config.get('custom_packages', '').split('\n') if pkg.strip()])
    total_packages = selected_count + custom_count
    auto_analyze = config.get('auto_analyze', True)
    
    return f"📊 Config: {total_packages} packages, auto-analyze: {'✅' if auto_analyze else '❌'}"

def extract_package_list_only(ui_components: Dict[str, Any]) -> list:
    """Extract package list saja untuk quick operations - one-liner"""
    selected = get_selected_packages(ui_components)
    custom = [pkg.strip() for pkg in getattr(ui_components.get('custom_packages'), 'value', '').split('\n') if pkg.strip()]
    return selected + custom