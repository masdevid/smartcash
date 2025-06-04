"""
File: smartcash/ui/setup/dependency_installer/handlers/status_check_handler.py
Deskripsi: Fixed status check handler tanpa log_message_safe dependency dengan one-liner style
"""

from typing import Dict, Any
from IPython.display import display, HTML

from smartcash.ui.setup.dependency_installer.utils.system_info_utils import (
    get_comprehensive_system_info, check_system_requirements
)
from smartcash.ui.setup.dependency_installer.utils.package_utils import (
    get_installed_packages_dict, get_package_detailed_info
)
from smartcash.ui.setup.dependency_installer.utils.ui_state_utils import (
    create_operation_context, ProgressSteps, update_package_status_by_name,
    update_status_panel, log_to_ui_safe
)
from smartcash.ui.setup.dependency_installer.utils.report_generator_utils import (
    generate_comprehensive_status_report, generate_system_compatibility_report
)
from smartcash.ui.setup.dependency_installer.components.package_selector import (
    get_package_categories, update_package_status
)

def setup_status_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup status check handler tanpa log_message_safe dependency - one-liner style"""
    
    def execute_status_check(button=None):
        """Execute comprehensive status check dengan operation context"""
        with create_operation_context(ui_components, 'status_check') as ctx:
            _execute_status_check_with_utils(ui_components, config, ctx)
    
    ui_components['check_button'].on_click(execute_status_check)

def _execute_status_check_with_utils(ui_components: Dict[str, Any], config: Dict[str, Any], ctx):
    """Execute status check menggunakan built-in logger dari CommonInitializer - one-liner style"""
    
    try:
        # Step 1: Initialize check
        ctx.stepped_progress('STATUS_INIT', "Memulai status check...")
        log_to_ui_safe(ui_components, "🔍 Memeriksa status dependensi...")
        
        # Step 2: Get system information - one-liner
        ctx.stepped_progress('STATUS_SYSTEM_INFO', "Mengumpulkan informasi sistem...")
        system_info, system_requirements = get_comprehensive_system_info(), check_system_requirements()
        log_to_ui_safe(ui_components, "💻 System information collected")
        
        # Step 3: Get comprehensive package status
        ctx.stepped_progress('STATUS_PACKAGE_CHECK', "Checking package status...")
        package_status = _get_comprehensive_package_status_with_utils(ui_components, ctx)
        
        # Step 4: Generate detailed report
        ctx.stepped_progress('STATUS_REPORT', "Generating report...")
        _display_comprehensive_report_with_utils(ui_components, system_info, system_requirements, package_status, ctx)
        
        # Step 5: Update UI status
        ctx.stepped_progress('STATUS_UI_UPDATE', "Updating UI status...")
        _update_ui_status_from_check_with_utils(ui_components, package_status)
        
        # Summary - one-liner calculations
        total_packages, installed_packages = len(package_status), sum(1 for status in package_status.values() if status['installed'])
        summary_msg = f"📊 Status Check: {installed_packages}/{total_packages} packages terinstall"
        log_to_ui_safe(ui_components, f"✅ {summary_msg}")
        update_status_panel(ui_components, summary_msg, "success")
        
        log_to_ui_safe(ui_components, "✅ Pemeriksaan status dependensi selesai")
        ctx.stepped_progress('STATUS_COMPLETE', "Status check selesai", "overall")
        ctx.stepped_progress('STATUS_COMPLETE', "Complete", "step")
        
    except Exception as e:
        log_to_ui_safe(ui_components, f"❌ Gagal memeriksa status dependensi: {str(e)}", "error")
        raise

def _get_comprehensive_package_status_with_utils(ui_components: Dict[str, Any], ctx) -> Dict[str, Dict[str, Any]]:
    """Get comprehensive status menggunakan built-in logger - one-liner approach"""
    
    package_status = {}
    package_categories = get_package_categories()
    installed_packages = get_installed_packages_dict()
    
    # Process all packages - one-liner nested comprehension
    [package_status.update({
        package['key']: {
            'name': package['name'], 'pip_name': package['pip_name'], 'category': category['name'],
            'package_name': (package_name := package['pip_name'].split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()),
            **_get_detailed_package_info_with_utils(package_name, installed_packages)
        }
    }) for category in package_categories 
     for package in (logger and logger.info(f"🔍 Checking {category['name']} category...") or category['packages'])]
    
    return package_status

def _get_detailed_package_info_with_utils(package_name: str, installed_packages: Dict[str, str]) -> Dict[str, Any]:
    """Get detailed info menggunakan package utils - one-liner approach"""
    
    # Import utils - one-liner
    is_installed_func = __import__('smartcash.ui.setup.dependency_installer.utils.package_utils', fromlist=['is_package_installed']).is_package_installed
    get_version_func = __import__('smartcash.ui.setup.dependency_installer.utils.package_utils', fromlist=['get_package_version']).get_package_version
    
    # Check installation - one-liner conditional
    if not is_installed_func(package_name, installed_packages):
        return {'installed': False}
    
    # Get detailed info - one-liner
    version = get_version_func(package_name, installed_packages)
    detailed_info = get_package_detailed_info(package_name)
    
    return {
        'installed': True, 'version': version,
        'location': detailed_info.get('location'), 'dependencies': detailed_info.get('requires', []),
        'summary': detailed_info.get('summary', ''), 'author': detailed_info.get('author', ''),
        'home_page': detailed_info.get('home-page', '')
    }

def _display_comprehensive_report_with_utils(ui_components: Dict[str, Any], system_info: Dict[str, Any], 
                                           system_requirements: Dict[str, Any], package_status: Dict[str, Dict[str, Any]], ctx, logger):
    """Display comprehensive report menggunakan report generator utils - one-liner approach"""
    
    # Early return if no log output - one-liner
    if not (log_output := ui_components.get('log_output')):
        return
    
    # Generate combined report - one-liner
    combined_report = f"{generate_comprehensive_status_report(system_info, package_status)}{generate_system_compatibility_report(system_info)}"
    
    # Display report - one-liner
    with log_output: display(HTML(combined_report))
    
    # Log summary - one-liner
    _log_status_summary_with_utils(ui_components, package_status, system_requirements, ctx, logger)

def _log_status_summary_with_utils(ui_components: Dict[str, Any], package_status: Dict[str, Dict[str, Any]], 
                                 system_requirements: Dict[str, Any], ctx, logger):
    """Log summary menggunakan built-in logger - one-liner approach"""
    
    # Category summary - one-liner comprehension
    category_summary = {}
    [category_summary.setdefault(pkg_info['category'], {'total': 0, 'installed': 0}).update({
        'total': category_summary[pkg_info['category']]['total'] + 1,
        'installed': category_summary[pkg_info['category']]['installed'] + (1 if pkg_info['installed'] else 0)
    }) for pkg_info in package_status.values()]
    
    # Log category summaries - one-liner
    [logger and logger.info(f"📋 {category}: {stats['installed']}/{stats['total']} ({(stats['installed']/stats['total']*100):.1f}%)")
     for category, stats in category_summary.items()]
    
    # Log system requirements warnings - one-liner
    if not system_requirements.get('all_requirements_met', True) and logger:
        logger.warning("⚠️ Some system requirements not met")
        [logger.warning(f"   • {msg}") for check, msg in [
            ('python_version_ok', 'Python version < 3.7'),
            ('memory_sufficient', 'Memory < 2GB'),
            ('platform_supported', 'Platform not officially supported')
        ] if not system_requirements.get(check)]

def _update_ui_status_from_check_with_utils(ui_components: Dict[str, Any], package_status: Dict[str, Dict[str, Any]]):
    """Update UI package status menggunakan batch utils - one-liner"""
    
    # Create status mapping dan batch update - one-liner
    status_mapping = {pkg_info['package_name']: 'installed' if pkg_info['installed'] else 'missing' 
                     for pkg_info in package_status.values()}
    
    __import__('smartcash.ui.setup.dependency_installer.utils.ui_state_utils', fromlist=['batch_update_package_status']).batch_update_package_status(ui_components, status_mapping)