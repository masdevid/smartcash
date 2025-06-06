"""
File: smartcash/ui/setup/dependency/handlers/analysis_handler.py
Deskripsi: Fixed analysis handler dengan proper logger reference
"""

from typing import Dict, Any

from smartcash.ui.setup.dependency.utils.package_utils import (
    get_installed_packages_dict, batch_check_packages_status, 
    parse_package_requirement, extract_package_name_from_requirement
)
from smartcash.ui.setup.dependency.utils.ui_state_utils import (
    create_operation_context, ProgressSteps, update_package_status_by_name,
    update_status_panel, log_to_ui_safe
)
from smartcash.ui.setup.dependency.utils.report_generator_utils import (
    generate_analysis_summary_report
)
from smartcash.ui.setup.dependency.components.package_selector import (
    get_package_categories, update_package_status
)

def setup_analysis_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup analysis handler dengan fixed logger reference"""
    
    def execute_analysis(button=None):
        """Execute package analysis dengan operation context"""
        with create_operation_context(ui_components, 'analysis') as ctx:
            _execute_analysis_with_utils(ui_components, config, ctx)
    
    ui_components['analyze_button'].on_click(execute_analysis)
    ui_components['trigger_analysis'] = lambda: execute_analysis()

def _execute_analysis_with_utils(ui_components: Dict[str, Any], config: Dict[str, Any], ctx):
    """Execute analysis dengan fixed logger reference"""
    
    logger = ui_components.get('logger')  # Get logger from ui_components
    
    # Get progress tracker jika tersedia
    progress_tracker = ui_components.get('progress_tracker')
    
    try:
        # Step 1: Initialize analysis
        if progress_tracker:
            progress_tracker.show("Analisis Dependensi", ["Persiapan", "Scanning", "Evaluasi", "Laporan"])
            progress_tracker.update_overall(10, "Memulai analisis...")
            progress_tracker.update_current(25, "Inisialisasi...")
        else:
            ctx.stepped_progress('ANALYSIS_INIT', "Memulai analisis...")
            
        log_to_ui_safe(ui_components, "🔍 Memulai analisis dependensi...")

        # Step 2: Get installed packages
        if progress_tracker:
            progress_tracker.update_overall(30, "Scanning packages...")
            progress_tracker.update_current(50, "Mendapatkan daftar packages...")
        else:
            ctx.stepped_progress('ANALYSIS_GET_PACKAGES', "Mendapatkan daftar packages...")
            
        installed_packages = get_installed_packages_dict()
        log_to_ui_safe(ui_components, f"📦 Found {len(installed_packages)} installed packages")

        # Step 3: Get package categories dan reset status
        if progress_tracker:
            progress_tracker.update_overall(50, "Evaluasi packages...")
            progress_tracker.update_current(75, "Menganalisis categories...")
        else:
            ctx.stepped_progress('ANALYSIS_CATEGORIES', "Menganalisis categories...")
            
        package_categories = get_package_categories()
        _reset_all_package_status_to_checking(ui_components, package_categories)
        
        # Step 4: Analyze packages status
        if progress_tracker:
            progress_tracker.update_overall(70, "Checking packages...")
            progress_tracker.update_current(85, "Memulai pengecekan")
        else:
            ctx.stepped_progress('ANALYSIS_CHECK', "Checking package status...")
            
        analysis_results = _analyze_packages_with_utils(
            package_categories, installed_packages, ui_components, ctx, logger
        )
        
        # Step 5: Update UI dan generate report
        if progress_tracker:
            progress_tracker.update_overall(90, "Generating report...")
            progress_tracker.update_current(95, "Updating UI...")
        else:
            ctx.stepped_progress('ANALYSIS_UPDATE_UI', "Updating UI...")
            
        _finalize_analysis_results(ui_components, analysis_results, ctx, logger)
        
        # Complete operation
        if progress_tracker:
            progress_tracker.update_overall(100, "Analisis selesai")
            progress_tracker.update_current(100, "Complete")
            progress_tracker.complete("✅ Analisis selesai")
        else:
            ctx.stepped_progress('ANALYSIS_COMPLETE', "Analisis selesai", "overall")
            ctx.stepped_progress('ANALYSIS_COMPLETE', "Complete", "step")
        
    except Exception as e:
        log_to_ui_safe(ui_components, f"❌ Gagal menganalisis dependensi: {str(e)}", "error")
        logger and logger.error(f"💥 Analysis error: {str(e)}")
        raise

def _analyze_packages_with_utils(package_categories: list, installed_packages: Dict[str, str], 
                                ui_components: Dict[str, Any], ctx, logger) -> Dict[str, Any]:
    """Analyze packages dengan fixed logger reference"""
    
    # Get progress tracker jika tersedia
    progress_tracker = ui_components.get('progress_tracker')
    
    analysis_results = {'installed': [], 'missing': [], 'upgrade_needed': [], 'package_details': {}}
    total_packages = sum(len(category['packages']) for category in package_categories)
    current_package = 0
    
    # Collect all packages untuk batch processing
    all_packages = [(package, category['name']) for category in package_categories for package in category['packages']]
    package_requirements = [pkg['pip_name'] for pkg, _ in all_packages]
    batch_status = batch_check_packages_status(package_requirements)
    
    # Process results
    for (package, category_name), requirement in zip(all_packages, package_requirements):
        current_package += 1
        progress = int((current_package / total_packages) * 100)
        
        # Update progress
        if progress_tracker:
            progress_tracker.update_overall(70 + int((current_package / total_packages) * 20), 
                                  f"Checking {current_package}/{total_packages}")
            progress_tracker.update_current(progress, f"Analyzing {package['name']}...")
        else:
            progress = ProgressSteps.ANALYSIS_CHECK + int((current_package / total_packages) * 30)
            # Gunakan metode yang benar untuk progress tracking
            if hasattr(ctx, 'update_progress'):
                ctx.update_progress(progress, f"Analyzing {package['name']}...")
            elif hasattr(ctx, 'progress_tracker') and callable(ctx.progress_tracker):
                # Fallback untuk kompatibilitas dengan context lama
                ctx.progress_tracker('overall', progress, f"Analyzing {package['name']}...")
        
        package_key, status_info = package['key'], batch_status[requirement]
        
        # Store results
        analysis_results['package_details'][package_key] = {
            'name': package['name'], 'pip_name': requirement, 'category': category_name,
            'package_name': extract_package_name_from_requirement(requirement),
            'required_version': status_info.get('required_version', ''),
            'status': status_info['status'], 'installed_version': status_info.get('version'),
            'compatible': status_info.get('compatible', False)
        }
        
        # Categorize dan update UI
        {'installed': analysis_results['installed'], 'missing': analysis_results['missing'], 
         'upgrade': analysis_results['upgrade_needed']}.get(status_info['status'], []).append(package_key)
        
        ui_status = _map_status_to_ui(status_info['status'])
        update_package_status(ui_components, package_key, ui_status)
        
        # Log progress
        status_emoji = "✅" if status_info['status'] == 'installed' else "❌" if status_info['status'] == 'missing' else "⚠️"
        logger and logger.debug(f"{status_emoji} {package['name']}: {status_info['status']} ({current_package}/{total_packages})")
    
    return analysis_results

def _reset_all_package_status_to_checking(ui_components: Dict[str, Any], package_categories: list):
    """Reset semua package status ke checking"""
    [update_package_status(ui_components, package['key'], 'checking') 
     for category in package_categories for package in category['packages']]

def _map_status_to_ui(status: str) -> str:
    """Map analysis status ke UI status"""
    return {'installed': 'installed', 'missing': 'missing', 'upgrade': 'upgrade', 'error': 'error'}.get(status, 'checking')

def _finalize_analysis_results(ui_components: Dict[str, Any], analysis_results: Dict[str, Any], ctx, logger):
    """Finalize analysis results dengan comprehensive reporting"""
    
    # Get progress tracker jika tersedia
    progress_tracker = ui_components.get('progress_tracker')
    
    installed_count, missing_count, upgrade_count = len(analysis_results['installed']), len(analysis_results['missing']), len(analysis_results['upgrade_needed'])
    total_count = installed_count + missing_count + upgrade_count
    
    # Generate dan display report
    report_html = generate_analysis_summary_report(analysis_results)
    log_output = ui_components.get('log_output')
    
    if log_output and hasattr(log_output, 'clear_output'):
        with log_output:
            from IPython.display import display, HTML
            display(HTML(report_html))
    
    # Log summary
    if logger:
        logger.info("📊 Analysis Summary:")
        logger.info(f"   ✅ Installed: {installed_count}/{total_count}")
        logger.info(f"   ❌ Missing: {missing_count}/{total_count}")
        logger.info(f"   ⚠️ Need Upgrade: {upgrade_count}/{total_count}")
    
    # Update status panel
    status_msg, status_type = (
        (f"✅ Semua {installed_count} packages sudah terinstall dengan benar", "success")
        if missing_count == 0 and upgrade_count == 0
        else (f"📊 Analysis: {installed_count} installed, {missing_count} missing, {upgrade_count} need upgrade", "info")
    )
    update_status_panel(ui_components, status_msg, status_type)