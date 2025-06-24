"""
File: smartcash/ui/setup/dependency/handlers/status_check_handler.py
Deskripsi: Handler untuk status check dan system report generation
"""

from typing import Dict, Any, Callable
from smartcash.ui.setup.dependency.utils import (
    update_status_panel, with_button_context,
    get_comprehensive_system_info, generate_system_compatibility_report,
    show_progress_tracker_safe, complete_operation_with_message
)

def setup_status_check_handler(ui_components: Dict[str, Any]) -> Dict[str, Callable]:
    """Setup status check handler untuk system report"""
    
    def handle_system_report():
        """Generate comprehensive system report"""
        logger = ui_components.get('logger')
        
        with with_button_context(ui_components, 'system_report_button'):
            try:
                update_status_panel(ui_components, "🔍 Mengumpulkan informasi sistem...", "info")
                show_progress_tracker_safe(ui_components, "System Analysis")
                
                if logger:
                    logger.info("🔍 Generating system compatibility report...")
                
                # Update progress - gathering system info
                from smartcash.ui.setup.dependency.utils import update_progress_step
                update_progress_step(ui_components, "overall", 20, "Collecting system info")
                
                # Collect system information
                system_info = get_comprehensive_system_info()
                
                update_progress_step(ui_components, "overall", 50, "Checking compatibility")
                
                # Generate compatibility report
                compatibility_report = generate_system_compatibility_report(system_info)
                
                update_progress_step(ui_components, "overall", 80, "Generating report")
                
                # Log detailed report
                _log_system_report(logger, system_info, compatibility_report)
                
                update_progress_step(ui_components, "overall", 100, "Report completed")
                
                # Summary message
                summary = _generate_report_summary(system_info, compatibility_report)
                complete_operation_with_message(ui_components, f"✅ {summary}")
                
                if logger:
                    logger.info(f"📊 System report completed: {summary}")
                
            except Exception as e:
                update_status_panel(ui_components, f"❌ System report error: {str(e)}", "error")
                if logger:
                    logger.error(f"❌ System report failed: {str(e)}")
    
    def handle_package_status_check():
        """Check status dari selected packages"""
        logger = ui_components.get('logger')
        
        with with_button_context(ui_components, 'check_button'):
            try:
                from smartcash.ui.setup.dependency.utils import get_selected_packages
                
                # Get selected packages
                selected_packages = get_selected_packages(ui_components.get('package_selector', {}))
                
                if not selected_packages:
                    update_status_panel(ui_components, "⚠️ Tidak ada packages yang dipilih", "warning")
                    return
                
                update_status_panel(ui_components, f"🔍 Checking status {len(selected_packages)} packages...", "info")
                show_progress_tracker_safe(ui_components, "Package Status Check")
                
                if logger:
                    logger.info(f"🔍 Checking status of {len(selected_packages)} packages...")
                
                # Check packages status
                results = _check_packages_status_batch(selected_packages, ui_components)
                
                # Generate status summary
                summary = _generate_status_summary(results)
                complete_operation_with_message(ui_components, f"✅ {summary}")
                
                if logger:
                    logger.info(f"📊 Status check completed: {summary}")
                
            except Exception as e:
                update_status_panel(ui_components, f"❌ Status check error: {str(e)}", "error")
                if logger:
                    logger.error(f"❌ Status check failed: {str(e)}")
    
    # Setup button handlers
    system_report_button = ui_components.get('system_report_button')
    if system_report_button:
        system_report_button.on_click(lambda b: handle_system_report())
    
    check_button = ui_components.get('check_button')
    if check_button:
        check_button.on_click(lambda b: handle_package_status_check())
    
    return {
        'handle_system_report': handle_system_report,
        'handle_package_status_check': handle_package_status_check
    }

def _check_packages_status_batch(packages: list, ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check status packages dalam batch"""
    from smartcash.ui.setup.dependency.utils import batch_check_packages_status, update_progress_step
    
    results = {
        'installed': [],
        'not_installed': [],
        'errors': []
    }
    
    batch_size = 10
    for i in range(0, len(packages), batch_size):
        batch = packages[i:i + batch_size]
        progress = int((i / len(packages)) * 100)
        
        update_progress_step(ui_components, "overall", progress, f"Checking batch {i//batch_size + 1}")
        
        batch_results = batch_check_packages_status(batch)
        
        for result in batch_results:
            if result['success']:
                if result['installed']:
                    results['installed'].append(result['package'])
                else:
                    results['not_installed'].append(result['package'])
            else:
                results['errors'].append({'package': result['package'], 'error': result.get('error', 'Unknown error')})
    
    return results

def _log_system_report(logger, system_info: Dict[str, Any], compatibility_report: Dict[str, Any]):
    """Log detailed system report"""
    if not logger:
        return
    
    try:
        logger.info("💻 System Information:")
        logger.info(f"   • Python: {system_info.get('python_version', 'Unknown')}")
        logger.info(f"   • Platform: {system_info.get('platform', 'Unknown')}")
        logger.info(f"   • Architecture: {system_info.get('architecture', 'Unknown')}")
        logger.info(f"   • Memory: {system_info.get('memory_info', {}).get('available_gb', 'Unknown')} GB available")
        
        # GPU info
        gpu_info = system_info.get('gpu_info', {})
        if gpu_info.get('cuda_available'):
            logger.info(f"   • CUDA: Available (version {gpu_info.get('cuda_version', 'Unknown')})")
        else:
            logger.info("   • CUDA: Not available")
        
        # Compatibility warnings
        warnings = compatibility_report.get('warnings', [])
        if warnings:
            logger.warning("⚠️ Compatibility Warnings:")
            for warning in warnings[:3]:  # Show first 3
                logger.warning(f"   • {warning}")
        
        # Recommendations
        recommendations = compatibility_report.get('recommendations', [])
        if recommendations:
            logger.info("💡 Recommendations:")
            for rec in recommendations[:3]:  # Show first 3
                logger.info(f"   • {rec}")
                
    except Exception:
        logger.info("📊 System report generated (details in compatibility report)")

def _generate_report_summary(system_info: Dict[str, Any], compatibility_report: Dict[str, Any]) -> str:
    """Generate concise report summary"""
    try:
        python_version = system_info.get('python_version', 'Unknown')
        warnings_count = len(compatibility_report.get('warnings', []))
        
        if warnings_count == 0:
            return f"System compatible (Python {python_version})"
        else:
            return f"System report complete, {warnings_count} warnings found"
    except:
        return "System report generated"

def _generate_status_summary(results: Dict[str, Any]) -> str:
    """Generate status check summary"""
    try:
        installed = len(results['installed'])
        not_installed = len(results['not_installed'])
        errors = len(results['errors'])
        total = installed + not_installed + errors
        
        return f"{installed}/{total} installed, {errors} errors"
    except:
        return "Status check completed"