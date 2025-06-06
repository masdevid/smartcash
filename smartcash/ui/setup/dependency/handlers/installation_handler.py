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
    
    # Get progress tracker jika tersedia
    progress_tracker = ui_components.get('progress_tracker')
    
    try:
        # Step 1: Get selected packages
        if progress_tracker:
            progress_tracker.show("Instalasi Packages", ["Persiapan", "Analisis", "Instalasi", "Finalisasi"])
            progress_tracker.update_overall(10, "Mempersiapkan instalasi...")
            progress_tracker.update_current(25, "Mengumpulkan packages...")
        else:
            ctx.stepped_progress('INSTALL_INIT', "Mempersiapkan instalasi...")
        
        log_to_ui_safe(ui_components, "🚀 Memulai proses instalasi packages")
        
        selected_packages = get_selected_packages(ui_components)
        if not selected_packages:
            update_status_panel(ui_components, "❌ Tidak ada packages yang dipilih", "error")
            log_to_ui_safe(ui_components, "⚠️ Tidak ada packages yang dipilih untuk instalasi", "warning")
            return
        
        # Step 2: Filter uninstalled packages
        if progress_tracker:
            progress_tracker.update_overall(30, "Menganalisis packages...")
            progress_tracker.update_current(50, f"Menganalisis {len(selected_packages)} packages...")
        else:
            ctx.stepped_progress('INSTALL_ANALYSIS', "Menganalisis packages...")
        
        log_to_ui_safe(ui_components, f"📦 Menganalisis {len(selected_packages)} packages yang dipilih")
        
        def package_logger_func(msg):
            log_to_ui_safe(ui_components, msg)
        
        packages_to_install = filter_uninstalled_packages(selected_packages, package_logger_func)
        
        if not packages_to_install:
            log_to_ui_safe(ui_components, "✅ Semua packages sudah terinstall dengan benar")
            
            # Complete operation dengan progress tracker baru
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                progress_tracker.update_overall(100, "Semua packages sudah terinstall")
                progress_tracker.update_current(100, "Complete")
                progress_tracker.complete("✅ Semua packages sudah terinstall")
            else:
                # Fallback untuk progress tracking lama
                ui_components.get('update_progress', lambda *a: None)('overall', 100, "Semua packages sudah terinstall")
                ui_components.get('update_progress', lambda *a: None)('step', 100, "Complete")
                ui_components.get('complete_operation', lambda x: None)("Semua packages sudah terinstall dengan benar")
            
            update_status_panel(ui_components, "✅ Semua packages sudah terinstall", "success")
            
            # Hide progress bars setelah delay
            import threading
            def hide_progress_delayed():
                time.sleep(2)
                if progress_tracker:
                    progress_tracker.reset()
                else:
                    ui_components.get('reset_all', lambda: None)()
            
            threading.Thread(target=hide_progress_delayed, daemon=True).start()
            return
        
        # Step 3: Install packages dengan parallel processing
        if progress_tracker:
            progress_tracker.update_overall(50, f"Installing packages...")
            progress_tracker.update_current(75, f"Memulai instalasi {len(packages_to_install)} packages...")
        else:
            ctx.stepped_progress('INSTALL_START', f"Installing {len(packages_to_install)} packages...")
        
        log_to_ui_safe(ui_components, f"📦 Installing {len(packages_to_install)} packages dengan parallel processing")
        
        installation_results = _install_packages_parallel_with_utils(
            packages_to_install, ui_components, config, package_logger_func, ctx, logger
        )
        
        # Step 4: Finalize dan generate report
        if progress_tracker:
            progress_tracker.update_overall(80, "Finalizing installation...")
            progress_tracker.update_current(90, "Generating report...")
        else:
            ctx.stepped_progress('INSTALL_FINALIZE', "Finalizing installation...")
        
        log_to_ui_safe(ui_components, "📊 Generating installation report...")
        
        log_to_ui_safe(ui_components, f"⏱️ Installation selesai dalam {time.time() - start_time:.1f} detik")
        
        # Update all package status dan generate report
        _finalize_installation_results(ui_components, installation_results, time.time() - start_time, logger)
        
        # Complete operation
        if progress_tracker:
            progress_tracker.update_overall(100, "Installation complete")
            progress_tracker.update_current(100, "Complete")
            progress_tracker.complete("✅ Instalasi selesai")
        else:
            ctx.stepped_progress('INSTALL_COMPLETE', "Installation complete", "overall")
            ctx.stepped_progress('INSTALL_COMPLETE', "Complete", "step")
        
    except Exception as e:
        log_to_ui_safe(ui_components, f"❌ Gagal menginstal dependensi: {str(e)}", "error")
        if logger:
            logger.error(f"💥 Installation error: {str(e)}")
        raise

def _install_packages_parallel_with_utils(packages: list, ui_components: Dict[str, Any], 
                                         config: Dict[str, Any], logger_func, ctx, logger) -> Dict[str, bool]:
    """Install packages dengan parallel processing dan detailed progress tracking"""
    
    results = {}
    total_packages = len(packages)
    completed_count = 0
    
    # Update progress
    def update_installation_progress(package: str, success: bool):
        nonlocal completed_count
        completed_count += 1
        progress = int((completed_count / total_packages) * 100)
        
        # Get progress tracker jika tersedia
        progress_tracker = ui_components.get('progress_tracker')
        
        # Update progress bars
        if progress_tracker:
            # Update progress dengan metode yang benar
            progress_tracker.update_overall(50 + int((completed_count / total_packages) * 30), 
                                   f"Installing {completed_count}/{total_packages}")
            
            status_icon = "✅" if success else "❌"
            status_msg = "Success" if success else "Failed"
            color = "green" if success else "red"
            
            progress_tracker.update_current(progress, f"{status_icon} {package} {status_msg}", color)
        else:
            ui_components.get('update_progress', lambda *a: None)(
                'overall', 
                ProgressSteps.INSTALL_START + int((completed_count / total_packages) * 30),
                f"Installing packages: {completed_count}/{total_packages}"
            )
            
            ui_components.get('update_progress', lambda *a: None)(
                'step', 
                progress,
                f"{package}: {'✅ Success' if success else '❌ Failed'}",
                'green' if success else 'red'
            )
        
        # Update package status
        update_package_status_by_name(ui_components, package, 'installed' if success else 'error')
        
        # Log progress
        status_emoji = "✅" if success else "❌"
        logger_func(f"{status_emoji} {package}: {completed_count}/{total_packages} ({progress}%)")
    
    # Get installation configuration
    max_workers = min(len(packages), config.get('installation', {}).get('parallel_workers', 3))
    timeout = config.get('installation', {}).get('timeout', 300)
    use_cache = config.get('installation', {}).get('use_cache', True)
    force_reinstall = config.get('installation', {}).get('force_reinstall', False)
    
    if logger:
        logger.info(f"🔧 Installation config: {max_workers} workers, {timeout}s timeout, cache: {use_cache}, force: {force_reinstall}")
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_package = {
                executor.submit(install_single_package, package, timeout): package 
                for package in packages
            }
            
            if logger:
                logger.info(f"🚀 Started parallel installation dengan {len(future_to_package)} tasks")
            
            # Process results
            for future in as_completed(future_to_package):
                package = future_to_package[future]
                try:
                    success, message = future.result()
                    results[package] = success
                    update_installation_progress(package, success)
                    
                    # Detailed logging
                    if success:
                        if logger:
                            logger.debug(f"✅ {package}: {message}")
                    else:
                        if logger:
                            logger.warning(f"⚠️ {package}: {message}")
                        
                except Exception as e:
                    error_msg = f"💥 Error installing {package}: {str(e)}"
                    logger_func(error_msg)
                    if logger:
                        logger.error(error_msg)
                    results[package] = False
                    update_installation_progress(package, False)
        
        return results
        
    except Exception as e:
        error_msg = f"💥 Installation process failed: {str(e)}"
        logger_func(error_msg)
        if logger:
            logger.error(error_msg)
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
        if logger:
            logger.warning(f"⚠️ Failed packages: {', '.join(failed_packages[:5])}" + 
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
        if logger:
            logger.success(f"🎉 Installation completed successfully: all {success_count} packages installed")
        update_status_panel(ui_components, status_msg, "success")
    elif success_count > 0:
        status_msg = f"⚠️ Instalasi partial: {success_count}/{total_count} berhasil, {failed_count} gagal"
        if logger:
            logger.warning(f"⚠️ Partial installation: {failed_count} packages failed")
        update_status_panel(ui_components, status_msg, "warning")
    else:
        status_msg = f"❌ Instalasi gagal: {failed_count}/{total_count} packages gagal"
        if logger:
            logger.error(f"💥 Installation failed: all {failed_count} packages failed")
        update_status_panel(ui_components, status_msg, "error")
    
    # Update package status
    status_mapping = {
        package.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip(): 
        'installed' if success else 'error'
        for package, success in installation_results.items()
    }
    
    batch_update_package_status(ui_components, status_mapping)
    if logger:
        logger.info(f"🔄 Updated UI status untuk {len(status_mapping)} packages")