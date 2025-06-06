"""
File: smartcash/ui/setup/dependency/handlers/status_check_handler.py
Deskripsi: Fixed status check handler dengan proper logger reference
"""

from typing import Dict, Any
from IPython.display import display, HTML

from smartcash.ui.setup.dependency.utils.system_info_utils import (
    get_comprehensive_system_info, check_system_requirements
)
from smartcash.ui.setup.dependency.utils.package_utils import (
    get_installed_packages_dict, get_package_detailed_info
)
from smartcash.ui.setup.dependency.utils.ui_state_utils import (
    create_operation_context, ProgressSteps, update_package_status_by_name,
    update_status_panel, log_to_ui_safe
)
from smartcash.ui.setup.dependency.utils.report_generator_utils import (
    generate_comprehensive_status_report, generate_system_compatibility_report
)
from smartcash.ui.setup.dependency.components.package_selector import (
    get_package_categories, update_package_status
)

def setup_status_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup status check handler dengan fixed logger reference"""
    
    def execute_status_check(button=None):
        """Execute comprehensive status check dengan operation context"""
        with create_operation_context(ui_components, 'status_check') as ctx:
            _execute_status_check_with_utils(ui_components, config, ctx)
    
    ui_components['check_button'].on_click(execute_status_check)

def _execute_status_check_with_utils(ui_components: Dict[str, Any], config: Dict[str, Any], ctx):
    """Execute status check dengan fixed logger reference"""
    
    logger = ui_components.get('logger')  # Get logger from ui_components
    
    try:
        # Step 1: Initialize check
        ctx.stepped_progress('STATUS_INIT', "Memulai status check...")
        log_to_ui_safe(ui_components, "🔍 Memeriksa status dependensi...")
        
        # Step 2: Get system information
        ctx.stepped_progress('STATUS_SYSTEM_INFO', "Mengumpulkan informasi sistem...")
        system_info, system_requirements = get_comprehensive_system_info(), check_system_requirements()
        log_to_ui_safe(ui_components, "💻 System information collected")
        
        # Step 3: Get comprehensive package status
        ctx.stepped_progress('STATUS_PACKAGE_CHECK', "Checking package status...")
        package_status = _get_comprehensive_package_status_with_utils(ui_components, ctx, logger)
        
        # Step 4: Generate detailed report
        ctx.stepped_progress('STATUS_REPORT', "Generating report...")
        _display_comprehensive_report_with_utils(ui_components, system_info, system_requirements, package_status, ctx, logger)
        
        # Step 5: Update UI status
        ctx.stepped_progress('STATUS_UI_UPDATE', "Updating UI status...")
        _update_ui_status_from_check_with_utils(ui_components, package_status)
        
        # Summary
        total_packages, installed_packages = len(package_status), sum(1 for status in package_status.values() if status['installed'])
        summary_msg = f"📊 Status Check: {installed_packages}/{total_packages} packages terinstall"
        log_to_ui_safe(ui_components, f"✅ {summary_msg}")
        update_status_panel(ui_components, summary_msg, "success")
        
        log_to_ui_safe(ui_components, "✅ Pemeriksaan status dependensi selesai")
        ctx.stepped_progress('STATUS_COMPLETE', "Status check selesai", "overall")
        ctx.stepped_progress('STATUS_COMPLETE', "Complete", "step")
        
    except Exception as e:
        log_to_ui_safe(ui_components, f"❌ Gagal memeriksa status dependensi: {str(e)}", "error")
        if logger:
            logger.error(f"💥 Status check error: {str(e)}")
        raise

def _get_comprehensive_package_status_with_utils(ui_components: Dict[str, Any], ctx, logger) -> Dict[str, Dict[str, Any]]:
    """Get comprehensive status dengan fixed logger reference"""
    
    package_status = {}
    package_categories = get_package_categories()
    installed_packages = get_installed_packages_dict()
    
    if logger:
        logger.info("🔍 Analyzing package categories...")
    
    # Process all packages
    for category in package_categories:
        if logger:
            logger.info(f"🔍 Checking {category['name']} category...")
        for package in category['packages']:
            package_key = package['key']
            package_name = package['pip_name'].split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
            
            package_status[package_key] = {
                'name': package['name'], 
                'pip_name': package['pip_name'], 
                'category': category['name'],
                'package_name': package_name,
                **_get_detailed_package_info_with_utils(package_name, installed_packages, logger)
            }
    
    return package_status

def _get_detailed_package_info_with_utils(package_name: str, installed_packages: Dict[str, str], logger) -> Dict[str, Any]:
    """Get detailed info dengan fixed logger reference"""
    
    from smartcash.ui.setup.dependency.utils.package_utils import is_package_installed, get_package_version
    
    # Check installation
    if not is_package_installed(package_name, installed_packages):
        return {'installed': False}
    
    # Get detailed info
    version = get_package_version(package_name, installed_packages)
    detailed_info = get_package_detailed_info(package_name)
    
    return {
        'installed': True, 'version': version,
        'location': detailed_info.get('location'), 'dependencies': detailed_info.get('requires', []),
        'summary': detailed_info.get('summary', ''), 'author': detailed_info.get('author', ''),
        'home_page': detailed_info.get('home-page', '')
    }

def _display_comprehensive_report_with_utils(ui_components: Dict[str, Any], system_info: Dict[str, Any], 
                                           system_requirements: Dict[str, Any], package_status: Dict[str, Dict[str, Any]], ctx, logger):
    """Display comprehensive report dengan fixed logger reference"""
    
    log_output = ui_components.get('log_output')
    if not log_output:
        return
    
    # Generate combined report
    combined_report = f"{generate_comprehensive_status_report(system_info, package_status)}{generate_system_compatibility_report(system_info)}"
    
    # Display report
    with log_output: 
        display(HTML(combined_report))
    
    # Log summary
    _log_status_summary_with_utils(ui_components, package_status, system_requirements, ctx, logger)

def _log_status_summary_with_utils(ui_components: Dict[str, Any], package_status: Dict[str, Dict[str, Any]], 
                                 system_requirements: Dict[str, Any], ctx, logger):
    """Log summary dengan fixed logger reference"""
    
    # Category summary
    category_summary = {}
    for pkg_info in package_status.values():
        category = pkg_info['category']
        if category not in category_summary:
            category_summary[category] = {'total': 0, 'installed': 0}
        category_summary[category]['total'] += 1
        if pkg_info['installed']:
            category_summary[category]['installed'] += 1
    
    # Log category summaries
    for category, stats in category_summary.items():
        percentage = (stats['installed']/stats['total']*100) if stats['total'] > 0 else 0
        if logger:
            logger.info(f"📋 {category}: {stats['installed']}/{stats['total']} ({percentage:.1f}%)")
    
    # Log system requirements warnings
    if not system_requirements.get('all_requirements_met', True) and logger:
        logger.warning("⚠️ Some system requirements not met")
        requirement_checks = [
            ('python_version_ok', 'Python version < 3.7'),
            ('memory_sufficient', 'Memory < 2GB'),
            ('platform_supported', 'Platform not officially supported')
        ]
        
        for check, msg in requirement_checks:
            if not system_requirements.get(check):
                logger.warning(f"   • {msg}")

def _update_ui_status_from_check_with_utils(ui_components: Dict[str, Any], package_status: Dict[str, Dict[str, Any]]):
    """Update UI package status dengan batch utils"""
    
    # Create status mapping dan batch update
    status_mapping = {pkg_info['package_name']: 'installed' if pkg_info['installed'] else 'missing' 
                     for pkg_info in package_status.values()}
    
    from smartcash.ui.setup.dependency.utils.ui_state_utils import batch_update_package_status
    batch_update_package_status(ui_components, status_mapping)