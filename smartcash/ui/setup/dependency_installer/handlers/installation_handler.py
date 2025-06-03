"""
File: smartcash/ui/setup/dependency_installer/handlers/installation_handler.py
Deskripsi: SRP handler untuk proses instalasi packages dengan consolidated utils
"""

from typing import Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.ui.setup.dependency_installer.utils.package_utils import (
    filter_uninstalled_packages, install_single_package, get_installed_packages_dict
)
from smartcash.ui.setup.dependency_installer.utils.ui_state_utils import (
    create_operation_context, ProgressSteps, update_package_status_by_name,
    batch_update_package_status, update_status_panel, log_message_safe
)
from smartcash.ui.setup.dependency_installer.utils.report_generator_utils import (
    generate_installation_summary_report
)
from smartcash.ui.setup.dependency_installer.components.package_selector import get_selected_packages

def setup_installation_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup installation handler dengan consolidated utils"""
    
    def execute_installation(button=None):
        """Execute package installation dengan operation context"""
        
        with create_operation_context(ui_components, 'installation') as ctx:
            _execute_installation_with_utils(ui_components, config, ctx)
    
    ui_components['install_button'].on_click(execute_installation)

def _execute_installation_with_utils(ui_components: Dict[str, Any], config: Dict[str, Any], ctx):
    """Execute installation menggunakan consolidated utils"""
    import time
    start_time = time.time()
    
    try:
        # Step 1: Get selected packages
        ctx.stepped_progress('INSTALL_INIT', "Mempersiapkan instalasi...")
        log_message_safe(ui_components, "🚀 Memulai proses instalasi packages", "info")
        
        selected_packages = get_selected_packages(ui_components)
        if not selected_packages:
            update_status_panel(ui_components, "❌ Tidak ada packages yang dipilih", "error")
            return
        
        # Step 2: Filter uninstalled packages
        ctx.stepped_progress('INSTALL_ANALYSIS', "Menganalisis packages...")
        
        # Use safe logger function untuk filter
        def safe_logger_func(msg):
            log_message_safe(ui_components, msg, "info")
        
        packages_to_install = filter_uninstalled_packages(selected_packages, safe_logger_func)
        
        if not packages_to_install:
            log_message_safe(ui_components, "✅ Semua packages sudah terinstall dengan benar", "success")
            
            # Update progress ke 100% untuk semua packages sudah terinstall
            ui_components.get('update_progress', lambda *a: None)('overall', 100, "Semua packages sudah terinstall")
            ui_components.get('update_progress', lambda *a: None)('step', 100, "Complete")
            
            # Complete operation dengan proper completion
            ui_components.get('complete_operation', lambda x: None)("Semua packages sudah terinstall dengan benar")
            update_status_panel(ui_components, "✅ Semua packages sudah terinstall", "success")
            
            # Hide progress bars setelah complete dengan delay
            import threading
            import time
            def hide_progress_delayed():
                time.sleep(2)  # Wait 2 seconds
                ui_components.get('reset_all', lambda: None)()
            
            threading.Thread(target=hide_progress_delayed, daemon=True).start()
            return
        
        # Step 3: Install packages dengan parallel processing
        ctx.stepped_progress('INSTALL_START', f"Installing {len(packages_to_install)} packages...")
        log_message_safe(ui_components, f"📦 Installing {len(packages_to_install)} packages", "info")
        
        installation_results = _install_packages_parallel_with_utils(
            packages_to_install, ui_components, config, safe_logger_func
        )
        
        # Step 4: Finalize dan generate report
        ctx.stepped_progress('INSTALL_FINALIZE', "Finalisasi instalasi...")
        duration = time.time() - start_time
        
        # Update all package status
        _finalize_installation_results(ui_components, installation_results, duration)
        
        ctx.stepped_progress('INSTALL_FINALIZE', "Instalasi selesai", "overall")
        ctx.stepped_progress('INSTALL_FINALIZE', "Complete", "step")
        
    except Exception as e:
        log_message_safe(ui_components, f"💥 Installation failed: {str(e)}", "error")
        raise

def _install_packages_parallel_with_utils(packages: list, ui_components: Dict[str, Any], 
                                         config: Dict[str, Any], logger_func) -> Dict[str, bool]:
    """Install packages dengan parallel processing dan consolidated progress tracking"""
    
    results = {}
    total_packages = len(packages)
    completed_packages = 0
    
    def update_installation_progress(package: str, success: bool):
        nonlocal completed_packages
        completed_packages += 1
        
        # Calculate progress (20% start + 70% for installation)
        installation_progress = int((completed_packages / total_packages) * 70)
        overall_progress = ProgressSteps.INSTALL_START + installation_progress
        
        # Update progress using utils
        ctx.progress_tracker('overall', overall_progress, f"Installing package {completed_packages}/{total_packages}")
        ctx.progress_tracker('step', int((completed_packages / total_packages) * 100), f"Package {completed_packages}/{total_packages}")
        
        # Update package status using utils
        package_name = package.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
        status = 'installed' if success else 'error'
        update_package_status_by_name(ui_components, package_name, status)
        
        # Log progress
        status_emoji = "✅" if success else "❌"
        logger_func(f"{status_emoji} {package_name}: {'Success' if success else 'Failed'} ({completed_packages}/{total_packages})")
    
    # Parallel installation
    max_workers = min(len(packages), config.get('installation', {}).get('parallel_workers', 3))
    timeout = config.get('installation', {}).get('timeout', 300)
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_package = {
                executor.submit(install_single_package, package, timeout): package 
                for package in packages
            }
            
            # Process results
            for future in as_completed(future_to_package):
                package = future_to_package[future]
                try:
                    success, message = future.result()
                    results[package] = success
                    update_installation_progress(package, success)
                except Exception as e:
                    logger_func(f"💥 Error installing {package}: {str(e)}")
                    results[package] = False
                    update_installation_progress(package, False)
        
        return results
        
    except Exception as e:
        logger_func(f"💥 Installation process failed: {str(e)}")
        return {package: False for package in packages}

def _finalize_installation_results(ui_components: Dict[str, Any], installation_results: Dict[str, bool], 
                                  duration: float):
    """Finalize installation results dengan comprehensive reporting"""
    
    success_count = sum(1 for result in installation_results.values() if result)
    total_count = len(installation_results)
    
    # Generate dan display report
    report_html = generate_installation_summary_report(installation_results, duration)
    
    if 'log_output' in ui_components:
        from IPython.display import display, HTML
        with ui_components['log_output']:
            display(HTML(report_html))
    
    # Update status panel
    if success_count == total_count:
        status_msg = f"✅ Instalasi berhasil: {success_count}/{total_count} packages"
        log_message_safe(ui_components, f"🎉 Instalasi selesai: {success_count}/{total_count} packages berhasil", "success")
        update_status_panel(ui_components, status_msg, "success")
    else:
        failed_count = total_count - success_count
        status_msg = f"⚠️ Instalasi selesai: {success_count}/{total_count} berhasil, {failed_count} gagal"
        log_message_safe(ui_components, f"⚠️ Instalasi selesai dengan {failed_count} packages gagal", "warning")
        update_status_panel(ui_components, status_msg, "warning")
    
    # Update package status berdasarkan results
    status_mapping = {
        package.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip(): 
        'installed' if success else 'error'
        for package, success in installation_results.items()
    }
    batch_update_package_status(ui_components, status_mapping)