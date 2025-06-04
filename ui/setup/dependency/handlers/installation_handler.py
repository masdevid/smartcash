"""
File: smartcash/ui/setup/dependency/handlers/installation_handler.py
Deskripsi: Fixed installation handler dengan proper logger reference
"""

from typing import Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.ui.setup.dependency.utils.package_utils import (
    filter_uninstalled_packages, install_single_package, get_installed_packages_dict
)
from smartcash.ui.setup.dependency.utils.ui_state_utils import (
    create_operation_context, ProgressSteps, update_package_status_by_name,
    batch_update_package_status, update_status_panel, log_to_ui_safe
)
from smartcash.ui.setup.dependency.utils.report_generator_utils import (
    generate_installation_summary_report
)
from smartcash.ui.setup.dependency.components.package_selector import get_selected_packages

def setup_installation_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup installation handler dengan fixed logger reference"""
    
    def execute_installation(button=None):
        """Execute package installation dengan operation context"""
        
        with create_operation_context(ui_components, 'installation') as ctx:
            _execute_installation_with_utils(ui_components, config, ctx)
    
    ui_components['install_button'].on_click(execute_installation)

def _execute_installation_with_utils(ui_components: Dict[str, Any], config: Dict[str, Any], ctx):
    """Execute installation dengan fixed logger reference"""
    
    logger = ui_components.get('logger')  # Get logger from ui_components
    start_time = time.time()
    
    try:
        # Step 1: Get selected packages
        ctx.stepped_progress('INSTALL_INIT', "Mempersiapkan instalasi...")
        log_to_ui_safe(ui_components, "🚀 Memulai proses instalasi packages")
        
        selected_packages = get_selected_packages(ui_components)
        if not selected_packages:
            update_status_panel(ui_components, "❌ Tidak ada packages yang dipilih", "error")
            log_to_ui_safe(ui_components, "⚠️ Tidak ada packages yang dipilih untuk instalasi", "warning")
            return
        
        # Step 2: Filter uninstalled packages
        ctx.stepped_progress('INSTALL_ANALYSIS', "Menganalisis packages...")
        log_to_ui_safe(ui_components, f"📦 Menganalisis {len(selected_packages)} packages yang dipilih")
        
        def package_logger_func(msg):
            log_to_ui_safe(ui_components, msg)
        
        packages_to_install = filter_uninstalled_packages(selected_packages, package_logger_func)
        
        if not packages_to_install:
            log_to_ui_safe(ui_components, "✅ Semua packages sudah terinstall dengan benar")
            
            # Complete operation
            ui_components.get('update_progress', lambda *a: None)('overall', 100, "Semua packages sudah terinstall")
            ui_components.get('update_progress', lambda *a: None)('step', 100, "Complete")
            ui_components.get('complete_operation', lambda x: None)("Semua packages sudah terinstall dengan benar")
            update_status_panel(ui_components, "✅ Semua packages sudah terinstall", "success")
            
            # Hide progress bars setelah delay
            import threading
            def hide_progress_delayed():
                time.sleep(2)
                ui_components.get('reset_all', lambda: None)()
            
            threading.Thread(target=hide_progress_delayed, daemon=True).start()
            return
        
        # Step 3: Install packages dengan parallel processing
        ctx.stepped_progress('INSTALL_START', f"Installing {len(packages_to_install)} packages...")
        log_to_ui_safe(ui_components, f"📦 Installing {len(packages_to_install)} packages dengan parallel processing")
        
        installation_results = _install_packages_parallel_with_utils(
            packages_to_install, ui_components, config, package_logger_func, ctx, logger
        )
        
        # Step 4: Finalize dan generate report
        ctx.stepped_progress('INSTALL_FINALIZE', "Finalisasi instalasi...")
        duration = time.time() - start_time
        
        log_to_ui_safe(ui_components, f"⏱️ Installation selesai dalam {duration:.1f} detik")
        
        # Update all package status dan generate report
        _finalize_installation_results(ui_components, installation_results, duration, logger)
        
        ctx.stepped_progress('INSTALL_FINALIZE', "Instalasi selesai", "overall")
        ctx.stepped_progress('INSTALL_FINALIZE', "Complete", "step")
        
    except Exception as e:
        log_to_ui_safe(ui_components, f"❌ Gagal menginstal dependensi: {str(e)}", "error")
        logger and logger.error(f"💥 Installation error: {str(e)}")
        raise

def _install_packages_parallel_with_utils(packages: list, ui_components: Dict[str, Any], 
                                         config: Dict[str, Any], logger_func, ctx, logger) -> Dict[str, bool]:
    """Install packages dengan parallel processing dan detailed progress tracking"""
    
    results = {}
    total_packages = len(packages)
    completed_packages = 0
    
    def update_installation_progress(package: str, success: bool):
        nonlocal completed_packages
        completed_packages += 1
        
        # Calculate progress (20% start + 70% for installation)
        installation_progress = int((completed_packages / total_packages) * 70)
        overall_progress = ProgressSteps.INSTALL_START + installation_progress
        
        # Update progress
        ctx.progress_tracker('overall', overall_progress, f"Installing package {completed_packages}/{total_packages}")
        ctx.progress_tracker('step', int((completed_packages / total_packages) * 100), f"Package {completed_packages}/{total_packages}")
        
        # Update package status
        package_name = package.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
        status = 'installed' if success else 'error'
        update_package_status_by_name(ui_components, package_name, status)
        
        # Log progress
        status_emoji = "✅" if success else "❌"
        progress_msg = f"{status_emoji} {package_name}: {'Success' if success else 'Failed'} ({completed_packages}/{total_packages})"
        logger_func(progress_msg)
        
        # Update check/uncheck count display removed
        # Check/uncheck functionality removed
    
    # Get installation configuration
    max_workers = min(len(packages), config.get('installation', {}).get('parallel_workers', 3))
    timeout = config.get('installation', {}).get('timeout', 300)
    use_cache = config.get('installation', {}).get('use_cache', True)
    force_reinstall = config.get('installation', {}).get('force_reinstall', False)
    
    logger and logger.info(f"🔧 Installation config: {max_workers} workers, {timeout}s timeout, cache: {use_cache}, force: {force_reinstall}")
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_package = {
                executor.submit(install_single_package, package, timeout): package 
                for package in packages
            }
            
            logger and logger.info(f"🚀 Started parallel installation dengan {len(future_to_package)} tasks")
            
            # Process results
            for future in as_completed(future_to_package):
                package = future_to_package[future]
                try:
                    success, message = future.result()
                    results[package] = success
                    update_installation_progress(package, success)
                    
                    # Detailed logging
                    if success:
                        logger and logger.debug(f"✅ {package}: {message}")
                    else:
                        logger and logger.warning(f"⚠️ {package}: {message}")
                        
                except Exception as e:
                    error_msg = f"💥 Error installing {package}: {str(e)}"
                    logger_func(error_msg)
                    logger and logger.error(error_msg)
                    results[package] = False
                    update_installation_progress(package, False)
        
        return results
        
    except Exception as e:
        error_msg = f"💥 Installation process failed: {str(e)}"
        logger_func(error_msg)
        logger and logger.error(error_msg)
        return {package: False for package in packages}

def _finalize_installation_results(ui_components: Dict[str, Any], installation_results: Dict[str, bool], 
                                  duration: float, logger):
    """Finalize installation results dengan comprehensive reporting"""
    
    success_count = sum(1 for result in installation_results.values() if result)
    total_count = len(installation_results)
    failed_count = total_count - success_count
    
    # Log detailed summary
    if logger:
        logger.info("📊 Installation Summary:")
        logger.info(f"   ✅ Successful: {success_count}/{total_count}")
        logger.info(f"   ❌ Failed: {failed_count}/{total_count}")
        logger.info(f"   ⏱️ Duration: {duration:.1f} seconds")
        logger.info(f"   📈 Success Rate: {(success_count/total_count*100):.1f}%")
    
    # Log failed packages
    if failed_count > 0:
        failed_packages = [pkg for pkg, success in installation_results.items() if not success]
        logger and logger.warning(f"⚠️ Failed packages: {', '.join(failed_packages[:5])}" + 
                                 (f" and {len(failed_packages)-5} more" if len(failed_packages) > 5 else ""))
    
    # Generate dan display detailed report
    report_html = generate_installation_summary_report(installation_results, duration)
    
    log_output = ui_components.get('log_output')
    if log_output:
        from IPython.display import display, HTML
        with log_output:
            display(HTML(report_html))
    
    # Update status panel
    if success_count == total_count:
        status_msg = f"✅ Instalasi berhasil: {success_count}/{total_count} packages"
        logger and logger.success(f"🎉 Installation completed successfully: all {success_count} packages installed")
        update_status_panel(ui_components, status_msg, "success")
    elif success_count > 0:
        status_msg = f"⚠️ Instalasi partial: {success_count}/{total_count} berhasil, {failed_count} gagal"
        logger and logger.warning(f"⚠️ Partial installation: {failed_count} packages failed")
        update_status_panel(ui_components, status_msg, "warning")
    else:
        status_msg = f"❌ Instalasi gagal: {failed_count}/{total_count} packages gagal"
        logger and logger.error(f"💥 Installation failed: all {failed_count} packages failed")
        update_status_panel(ui_components, status_msg, "error")
    
    # Update package status
    status_mapping = {
        package.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip(): 
        'installed' if success else 'error'
        for package, success in installation_results.items()
    }
    
    batch_update_package_status(ui_components, status_mapping)
    logger and logger.info(f"🔄 Updated UI status untuk {len(status_mapping)} packages")