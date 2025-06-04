"""
File: smartcash/ui/setup/dependency_installer/handlers/status_check_handler.py
Deskripsi: Fixed status check handler dengan simplified logging menggunakan built-in logger dari CommonInitializer
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
    update_status_panel  # removed log_message_safe
)
from smartcash.ui.setup.dependency_installer.utils.report_generator_utils import (
    generate_comprehensive_status_report, generate_system_compatibility_report
)
from smartcash.ui.setup.dependency_installer.components.package_selector import (
    get_package_categories, update_package_status
)

def setup_status_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup status check handler dengan simplified logging"""
    
    def execute_status_check(button=None):
        """Execute comprehensive status check dengan operation context"""
        
        with create_operation_context(ui_components, 'status_check') as ctx:
            _execute_status_check_with_utils(ui_components, config, ctx)
    
    ui_components['check_button'].on_click(execute_status_check)

def _execute_status_check_with_utils(ui_components: Dict[str, Any], config: Dict[str, Any], ctx):
    """Execute status check menggunakan built-in logger dari CommonInitializer"""
    
    # Get built-in logger dari CommonInitializer
    logger = ui_components.get('logger')
    
    try:
        # Step 1: Initialize check
        ctx.stepped_progress('STATUS_INIT', "Memulai status check...")
        logger and logger.info("🔍 Memulai comprehensive status check")
        
        # Step 2: Get system information
        ctx.stepped_progress('STATUS_SYSTEM_INFO', "Mengumpulkan informasi sistem...")
        system_info = get_comprehensive_system_info()
        system_requirements = check_system_requirements()
        logger and logger.info("💻 System information collected")
        
        # Step 3: Get comprehensive package status
        ctx.stepped_progress('STATUS_PACKAGE_CHECK', "Checking package status...")
        package_status = _get_comprehensive_package_status_with_utils(ui_components, ctx)
        
        # Step 4: Generate detailed report
        ctx.stepped_progress('STATUS_REPORT', "Generating report...")
        _display_comprehensive_report_with_utils(ui_components, system_info, system_requirements, package_status, ctx)
        
        # Step 5: Update UI status
        ctx.stepped_progress('STATUS_UI_UPDATE', "Updating UI status...")
        _update_ui_status_from_check_with_utils(ui_components, package_status)
        
        # Summary
        total_packages = len(package_status)
        installed_packages = sum(1 for status in package_status.values() if status['installed'])
        
        summary_msg = f"📊 Status Check: {installed_packages}/{total_packages} packages terinstall"
        logger and logger.success(f"✅ {summary_msg}")
        update_status_panel(ui_components, summary_msg, "success")
        
        ctx.stepped_progress('STATUS_COMPLETE', "Status check selesai", "overall")
        ctx.stepped_progress('STATUS_COMPLETE', "Complete", "step")
        
    except Exception as e:
        logger and logger.error(f"💥 Status check failed: {str(e)}")
        raise

def _get_comprehensive_package_status_with_utils(ui_components: Dict[str, Any], ctx) -> Dict[str, Dict[str, Any]]:
    """Get comprehensive status menggunakan simplified logging"""
    
    logger = ui_components.get('logger')
    package_status = {}
    package_categories = get_package_categories()
    
    # Get installed packages dengan detailed info
    installed_packages = get_installed_packages_dict()
    
    for category in package_categories:
        logger and logger.info(f"🔍 Checking {category['name']} category...")
        
        for package in category['packages']:
            package_key = package['key']
            pip_name = package['pip_name']
            package_name = pip_name.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
            
            # Check installation dan get detailed info
            package_info = _get_detailed_package_info_with_utils(package_name, installed_packages)
            
            package_status[package_key] = {
                'name': package['name'],
                'pip_name': pip_name,
                'package_name': package_name,
                'category': category['name'],
                'installed': package_info['installed'],
                'version': package_info.get('version'),
                'location': package_info.get('location'),
                'dependencies': package_info.get('dependencies', []),
                'summary': package_info.get('summary', package.get('description', ''))
            }
    
    return package_status

def _get_detailed_package_info_with_utils(package_name: str, installed_packages: Dict[str, str]) -> Dict[str, Any]:
    """Get detailed info menggunakan package utils"""
    
    from smartcash.ui.setup.dependency_installer.utils.package_utils import (
        _get_package_name_variants, is_package_installed, get_package_version
    )
    
    # Check if installed menggunakan utils
    if not is_package_installed(package_name, installed_packages):
        return {'installed': False}
    
    # Get version menggunakan utils
    version = get_package_version(package_name, installed_packages)
    
    # Get additional details
    detailed_info = get_package_detailed_info(package_name)
    
    return {
        'installed': True,
        'version': version,
        'location': detailed_info.get('location'),
        'dependencies': detailed_info.get('requires', []),
        'summary': detailed_info.get('summary', ''),
        'author': detailed_info.get('author', ''),
        'home_page': detailed_info.get('home-page', '')
    }

def _display_comprehensive_report_with_utils(ui_components: Dict[str, Any], system_info: Dict[str, Any], 
                                           system_requirements: Dict[str, Any], package_status: Dict[str, Dict[str, Any]], ctx):
    """Display comprehensive report menggunakan report generator utils"""
    
    if 'log_output' not in ui_components:
        return
    
    # Generate main report
    main_report = generate_comprehensive_status_report(system_info, package_status)
    
    # Generate compatibility report
    compatibility_report = generate_system_compatibility_report(system_info)
    
    # Combine reports
    combined_report = f"""
    {main_report}
    {compatibility_report}
    """
    
    # Display report
    with ui_components['log_output']:
        display(HTML(combined_report))
    
    # Log summary
    _log_status_summary_with_utils(ui_components, package_status, system_requirements, ctx)

def _log_status_summary_with_utils(ui_components: Dict[str, Any], package_status: Dict[str, Dict[str, Any]], 
                                 system_requirements: Dict[str, Any], ctx):
    """Log summary menggunakan built-in logger"""
    
    logger = ui_components.get('logger')
    
    # Package summary by category
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
        coverage = (stats['installed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        logger and logger.info(f"📋 {category}: {stats['installed']}/{stats['total']} ({coverage:.1f}%)")
    
    # Log system requirements
    if not system_requirements.get('all_requirements_met', True):
        logger and logger.warning("⚠️ Some system requirements not met")
        if not system_requirements.get('python_version_ok'):
            logger and logger.warning("   • Python version < 3.7")
        if not system_requirements.get('memory_sufficient'):
            logger and logger.warning("   • Memory < 2GB")
        if not system_requirements.get('platform_supported'):
            logger and logger.warning("   • Platform not officially supported")

def _update_ui_status_from_check_with_utils(ui_components: Dict[str, Any], package_status: Dict[str, Dict[str, Any]]):
    """Update UI package status menggunakan batch utils"""
    
    # Create status mapping
    status_mapping = {
        pkg_info['package_name']: 'installed' if pkg_info['installed'] else 'missing'
        for pkg_info in package_status.values()
    }
    
    # Batch update menggunakan utils
    from smartcash.ui.setup.dependency_installer.utils.ui_state_utils import batch_update_package_status
    batch_update_package_status(ui_components, status_mapping)