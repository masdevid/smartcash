"""
File: smartcash/ui/setup/dependency_installer/handlers/analysis_handler.py
Deskripsi: SRP handler untuk analisis packages dengan consolidated utils
"""

from typing import Dict, Any

from smartcash.ui.setup.dependency_installer.utils.package_utils import (
    get_installed_packages_dict, batch_check_packages_status, 
    parse_package_requirement, extract_package_name_from_requirement
)
from smartcash.ui.setup.dependency_installer.utils.ui_state_utils import (
    create_operation_context, ProgressSteps, update_package_status_by_name,
    update_status_panel, log_message_safe
)
from smartcash.ui.setup.dependency_installer.utils.report_generator_utils import (
    generate_analysis_summary_report
)
from smartcash.ui.setup.dependency_installer.components.package_selector import (
    get_package_categories, update_package_status
)

def setup_analysis_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup analysis handler dengan consolidated utils dan auto-trigger"""
    
    def execute_analysis(button=None):
        """Execute package analysis dengan operation context"""
        
        with create_operation_context(ui_components, 'analysis') as ctx:
            _execute_analysis_with_utils(ui_components, config, ctx)
    
    # Register handler
    ui_components['analyze_button'].on_click(execute_analysis)
    
    # Store trigger function untuk auto-analyze
    ui_components['trigger_analysis'] = lambda: execute_analysis()

def _execute_analysis_with_utils(ui_components: Dict[str, Any], config: Dict[str, Any], ctx):
    """Execute analysis menggunakan consolidated utils"""
    
    try:
        # Step 1: Initialize analysis
        ctx.stepped_progress('ANALYSIS_INIT', "Memulai analisis...")
        log_message_safe(ui_components, "🔍 Memulai analisis packages terinstall", "info")
        
        # Step 2: Get installed packages
        ctx.stepped_progress('ANALYSIS_GET_PACKAGES', "Mendapatkan daftar packages...")
        installed_packages = get_installed_packages_dict()
        log_message_safe(ui_components, f"📦 Found {len(installed_packages)} installed packages", "info")
        
        # Step 3: Get package categories dan reset status ke checking
        ctx.stepped_progress('ANALYSIS_CATEGORIES', "Menganalisis categories...")
        package_categories = get_package_categories()
        total_packages = sum(len(category['packages']) for category in package_categories)
        
        # Reset semua package status ke checking sebelum analysis
        _reset_all_package_status_to_checking(ui_components, package_categories)
        
        # Step 4: Analyze packages status
        ctx.stepped_progress('ANALYSIS_CHECK', "Checking package status...")
        analysis_results = _analyze_packages_with_utils(
            package_categories, installed_packages, ui_components, ctx
        )
        
        # Step 5: Update UI dan generate report
        ctx.stepped_progress('ANALYSIS_UPDATE_UI', "Updating UI...")
        _finalize_analysis_results(ui_components, analysis_results, ctx)
        
        ctx.stepped_progress('ANALYSIS_COMPLETE', "Analisis selesai")
        
    except Exception as e:
        log_message_safe(ui_components, f"💥 Analysis failed: {str(e)}", "error")
        raise

def _analyze_packages_with_utils(package_categories: list, installed_packages: Dict[str, str], 
                                ui_components: Dict[str, Any], ctx) -> Dict[str, Any]:
    """Analyze packages menggunakan batch utilities"""
    
    analysis_results = {
        'installed': [],
        'missing': [],
        'upgrade_needed': [],
        'package_details': {}
    }
    
    total_packages = sum(len(category['packages']) for category in package_categories)
    current_package = 0
    
    # Collect all packages untuk batch processing
    all_packages = []
    for category in package_categories:
        for package in category['packages']:
            all_packages.append((package, category['name']))
    
    # Batch check status
    package_requirements = [pkg['pip_name'] for pkg, _ in all_packages]
    batch_status = batch_check_packages_status(package_requirements)
    
    # Process results
    for (package, category_name), requirement in zip(all_packages, package_requirements):
        current_package += 1
        
        # Update progress
        progress = ProgressSteps.ANALYSIS_CHECK + int((current_package / total_packages) * 30)
        ctx.progress_tracker('overall', progress, f"Analyzing {package['name']}...")
        
        package_key = package['key']
        status_info = batch_status[requirement]
        
        # Store results
        analysis_results['package_details'][package_key] = {
            'name': package['name'],
            'pip_name': requirement,
            'package_name': extract_package_name_from_requirement(requirement),
            'required_version': status_info.get('required_version', ''),
            'status': status_info['status'],
            'installed_version': status_info.get('version'),
            'compatible': status_info.get('compatible', False),
            'category': category_name
        }
        
        # Categorize
        if status_info['status'] == 'installed':
            analysis_results['installed'].append(package_key)
        elif status_info['status'] == 'missing':
            analysis_results['missing'].append(package_key)
        elif status_info['status'] == 'upgrade':
            analysis_results['upgrade_needed'].append(package_key)
        
        # Update UI status immediately dengan proper logging dan debug
        ui_status = _map_status_to_ui(status_info['status'])
        update_package_status(ui_components, package_key, ui_status)
        
        # Debug logging untuk troubleshooting
        log_message_safe(ui_components, f"📋 {package['name']}: {status_info['status']} -> UI: {ui_status}", "debug")
    
    return analysis_results

def _reset_all_package_status_to_checking(ui_components: Dict[str, Any], package_categories: list):
    """Reset semua package status ke checking sebelum analysis"""
    for category in package_categories:
        for package in category['packages']:
            update_package_status(ui_components, package['key'], 'checking')

def _map_status_to_ui(status: str) -> str:
    """Map analysis status ke UI status - one-liner mapping dengan better mapping"""
    return {
        'installed': 'installed', 
        'missing': 'missing', 
        'upgrade': 'upgrade',
        'error': 'error'
    }.get(status, 'checking')

def _finalize_analysis_results(ui_components: Dict[str, Any], analysis_results: Dict[str, Any], ctx):
    """Finalize analysis results dengan comprehensive reporting"""
    
    installed_count = len(analysis_results['installed'])
    missing_count = len(analysis_results['missing'])
    upgrade_count = len(analysis_results['upgrade_needed'])
    total_count = installed_count + missing_count + upgrade_count
    
    # Generate dan display report
    report_html = generate_analysis_summary_report(analysis_results)
    
    if 'log_output' in ui_components:
        from IPython.display import display, HTML
        with ui_components['log_output']:
            display(HTML(report_html))
    
    # Log summary
    log_message_safe(ui_components, f"📊 Analysis Summary:", "info")
    log_message_safe(ui_components, f"   ✅ Installed: {installed_count}/{total_count}", "info")
    log_message_safe(ui_components, f"   ❌ Missing: {missing_count}/{total_count}", "info")
    log_message_safe(ui_components, f"   ⚠️ Need Upgrade: {upgrade_count}/{total_count}", "info")
    
    # Update status panel
    if missing_count == 0 and upgrade_count == 0:
        status_msg = f"✅ Semua {installed_count} packages sudah terinstall dengan benar"
        status_type = "success"
    else:
        status_msg = f"📊 Analysis: {installed_count} installed, {missing_count} missing, {upgrade_count} need upgrade"
        status_type = "info"
    
    update_status_panel(ui_components, status_msg, status_type)