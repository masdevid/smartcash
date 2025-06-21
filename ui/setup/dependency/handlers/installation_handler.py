"""
File: smartcash/ui/setup/dependency/handlers/installation_handler.py
Deskripsi: Fixed installation handler dengan consistent return types dan proper error handling
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
    """Execute installation dengan fixed logger reference dan consistent error handling"""
    
    logger = ui_components.get('logger')
    start_time = time.time()
    progress_tracker = ui_components.get('progress_tracker')
    
    try:
        # Step 1: Initialize progress tracker
        _initialize_installation_progress(progress_tracker, logger)
        log_to_ui_safe(ui_components, "🚀 Memulai proses instalasi packages")
        
        # Step 2: Get dan validate selected packages
        selected_packages = get_selected_packages(ui_components)
        if not selected_packages:
            _handle_no_packages_selected(ui_components, progress_tracker)
            return
        
        # Step 3: Analysis phase
        _update_analysis_progress(progress_tracker, len(selected_packages), logger)
        log_to_ui_safe(ui_components, f"📦 Menganalisis {len(selected_packages)} packages yang dipilih")
        
        def package_logger_func(msg):
            log_to_ui_safe(ui_components, msg)
        
        packages_to_install = filter_uninstalled_packages(selected_packages, package_logger_func)
        
        if not packages_to_install:
            _handle_all_packages_installed(ui_components, progress_tracker, logger)
            return
        
        # Step 4: Installation phase
        _update_installation_start_progress(progress_tracker, len(packages_to_install), logger)
        log_to_ui_safe(ui_components, f"📦 Installing {len(packages_to_install)} packages dengan parallel processing")
        
        installation_results = _install_packages_parallel_with_utils(
            packages_to_install, ui_components, config, package_logger_func, ctx, logger
        )
        
        # Step 5: Finalization
        _finalize_installation_with_progress(ui_components, installation_results, time.time() - start_time, progress_tracker, logger)
        
    except Exception as e:
        _handle_installation_error(ui_components, progress_tracker, logger, str(e))
        raise

def _initialize_installation_progress(progress_tracker, logger):
    """Initialize installation progress dengan safe error handling"""
    try:
        if progress_tracker and hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(0, "🚀 Memulai proses instalasi...", "info")
            progress_tracker.update_current(0, "Menyiapkan...", "info")
    except Exception as e:
        if logger:
            logger.debug(f"🔄 Progress tracker error (non-critical): {str(e)}")

def _handle_no_packages_selected(ui_components: Dict[str, Any], progress_tracker):
    """Handle case ketika tidak ada packages yang dipilih"""
    update_status_panel(ui_components, "❌ Tidak ada packages yang dipilih", "error")
    log_to_ui_safe(ui_components, "⚠️ Tidak ada packages yang dipilih untuk instalasi", "warning")
    
    try:
        if progress_tracker and hasattr(progress_tracker, 'error'):
            progress_tracker.error("❌ Tidak ada packages yang dipilih")
    except Exception:
        pass

def _update_analysis_progress(progress_tracker, package_count: int, logger):
    """Update progress untuk analysis phase"""
    try:
        if progress_tracker and hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(20, "📊 Menganalisis packages...", "info")
            progress_tracker.update_current(30, f"🔍 Memeriksa {package_count} packages...")
    except Exception as e:
        if logger:
            logger.debug(f"🔄 Progress tracker error (non-critical): {str(e)}")

def _handle_all_packages_installed(ui_components: Dict[str, Any], progress_tracker, logger):
    """Handle case ketika semua packages sudah terinstall"""
    log_to_ui_safe(ui_components, "✅ Semua packages sudah terinstall dengan benar")
    
    try:
        if progress_tracker:
            if hasattr(progress_tracker, 'update_overall'):
                progress_tracker.update_overall(100, "✅ Semua packages sudah terinstall")
            if hasattr(progress_tracker, 'update_current'):
                progress_tracker.update_current(100, "✅ Complete")
            if hasattr(progress_tracker, 'complete'):
                progress_tracker.complete("✅ Semua packages sudah terinstall dengan benar")
    except Exception as e:
        if logger:
            logger.debug(f"🔄 Progress tracker completion error (non-critical): {str(e)}")
    
    update_status_panel(ui_components, "✅ Semua packages sudah terinstall", "success")

def _update_installation_start_progress(progress_tracker, package_count: int, logger):
    """Update progress untuk installation start"""
    try:
        if progress_tracker:
            if hasattr(progress_tracker, 'update_overall'):
                progress_tracker.update_overall(50, "📦 Menginstall packages...")
            if hasattr(progress_tracker, 'update_current'):
                progress_tracker.update_current(0, f"📦 Memulai instalasi {package_count} packages...")
    except Exception as e:
        if logger:
            logger.debug(f"🔄 Progress tracker error (non-critical): {str(e)}")

def _install_packages_parallel_with_utils(packages: list, ui_components: Dict[str, Any], 
                                         config: Dict[str, Any], logger_func, ctx, logger) -> Dict[str, bool]:
    """
    FIXED: Install packages dengan parallel processing dan consistent return handling
    """
    
    progress_tracker = ui_components.get('progress_tracker')
    installation_config = config.get('installation', {})
    
    max_workers = min(installation_config.get('parallel_workers', 3), 4)  # Cap at 4 untuk stability
    timeout = installation_config.get('timeout', 300)
    
    results = {}
    total_packages = len(packages)
    
    if logger:
        logger.info(f"🔧 Installation config: {max_workers} workers, {timeout}s timeout")
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_package = {
                executor.submit(install_single_package, package, timeout): package 
                for package in packages
            }
            
            if logger:
                logger.info(f"🚀 Started parallel installation dengan {len(future_to_package)} tasks")
            
            completed_count = 0
            
            # Process results as they complete
            for future in as_completed(future_to_package):
                package = future_to_package[future]
                completed_count += 1
                
                try:
                    # FIXED: Expect Dict return dari install_single_package
                    result_dict = future.result()
                    
                    if not isinstance(result_dict, dict):
                        # Fallback jika return type tidak sesuai
                        success = False
                        message = f"Unexpected return type: {type(result_dict)}"
                        package_name = package
                    else:
                        success = result_dict.get('success', False)
                        message = result_dict.get('message', 'No message')
                        package_name = result_dict.get('package_name', package)
                    
                    results[package] = success
                    
                    # Update progress dengan detail yang lebih baik
                    _update_package_installation_progress(
                        progress_tracker, completed_count, total_packages, 
                        package_name, success, logger
                    )
                    
                    # Log hasil instalasi
                    status_icon = "✅" if success else "❌"
                    status_text = "Berhasil" if success else "Gagal"
                    log_msg = f"{status_icon} {package_name}: {status_text}"
                    log_to_ui_safe(ui_components, log_msg, "success" if success else "error")
                    
                    # Update package status di UI
                    update_package_status_by_name(ui_components, package_name, 'installed' if success else 'error')
                    
                    if logger:
                        if success:
                            logger.debug(f"✅ {package_name}: {message}")
                        else:
                            logger.warning(f"⚠️ {package_name}: {message}")
                        
                except Exception as e:
                    # Handle error dalam future processing
                    error_msg = f"❌ Error saat menginstall {package}: {str(e)}"
                    log_to_ui_safe(ui_components, error_msg, "error")
                    results[package] = False
                    
                    if logger:
                        logger.error(f"💥 Installation error for {package}: {str(e)}")
                    
                    # Update progress dan status untuk failed package
                    _update_package_installation_progress(
                        progress_tracker, completed_count, total_packages, 
                        package, False, logger
                    )
                    update_package_status_by_name(ui_components, package, 'error')
        
        # Final summary
        success_count = sum(1 for r in results.values() if r)
        total = len(results)
        completion_msg = f"✅ Selesai! {success_count} dari {total} package berhasil diinstall"
        log_to_ui_safe(ui_components, completion_msg, "success")
        
        if logger:
            logger.info(f"📊 Installation summary: {success_count}/{total} successful")
        
        return results
        
    except Exception as e:
        error_msg = f"💥 Installation process failed: {str(e)}"
        logger_func(error_msg)
        if logger:
            logger.error(error_msg)
        
        # Return failure for all packages
        return {package: False for package in packages}

def _update_package_installation_progress(progress_tracker, completed: int, total: int, 
                                        package_name: str, success: bool, logger):
    """Update progress untuk individual package installation"""
    try:
        if progress_tracker and hasattr(progress_tracker, 'update_overall'):
            progress = int((completed / total) * 100)
            overall_progress = 30 + int(progress * 0.6)  # 30-90% of overall progress
            
            status_icon = "✅" if success else "❌"
            status_msg = "Berhasil" if success else "Gagal"
            
            progress_tracker.update_overall(
                overall_progress,
                f"📦 {status_icon} {package_name}: {status_msg}",
                "success" if success else "error"
            )
            progress_tracker.update_current(
                progress,
                f"Diproses: {completed}/{total}",
                "info"
            )
    except Exception as e:
        if logger:
            logger.debug(f"🔄 Progress update error (non-critical): {str(e)}")

def _finalize_installation_with_progress(ui_components: Dict[str, Any], installation_results: Dict[str, bool], 
                                       duration: float, progress_tracker, logger):
    """Finalize installation dengan progress completion"""
    try:
        if progress_tracker and hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(100, "✅ Instalasi selesai", "success")
            progress_tracker.update_current(100, "Semua proses selesai", "success")
            
            if hasattr(progress_tracker, 'complete'):
                progress_tracker.complete("✅ Instalasi package selesai")
    except Exception as e:
        if logger:
            logger.debug(f"🔄 Progress tracker completion error (non-critical): {str(e)}")
    
    # Generate detailed report
    _finalize_installation_results(ui_components, installation_results, duration, logger)

def _handle_installation_error(ui_components: Dict[str, Any], progress_tracker, logger, error_msg: str):
    """Handle installation error dengan progress tracker update"""
    full_error_msg = f"❌ Gagal menginstal dependensi: {error_msg}"
    log_to_ui_safe(ui_components, full_error_msg, "error")
    
    try:
        if progress_tracker and hasattr(progress_tracker, 'error'):
            progress_tracker.error(full_error_msg, delay=1.0)
    except Exception as err:
        if logger:
            logger.debug(f"🔄 Progress tracker error handling failed (non-critical): {str(err)}")
    
    if logger:
        logger.error(f"💥 Installation error: {error_msg}")
    
    update_status_panel(ui_components, full_error_msg, "error")

def _finalize_installation_results(ui_components: Dict[str, Any], installation_results: Dict[str, bool], 
                                  duration: float, logger):
    """Finalize installation results dengan comprehensive reporting"""
    
    success_count = sum(1 for result in installation_results.values() if result)
    total_count = len(installation_results)
    failed_count = total_count - success_count
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    
    # Log detailed summary
    if logger:
        logger.info("\n" + "="*50)
        logger.info("📊 RINGKASAN INSTALASI")
        logger.info("="*50)
        logger.info(f"✅ Berhasil: {success_count}/{total_count}")
        logger.info(f"❌ Gagal: {failed_count}")
        logger.info(f"⏱️  Durasi: {duration:.1f} detik")
        logger.info(f"📈 Tingkat Keberhasilan: {success_rate:.1f}%")
    
    # Log failed packages
    if failed_count > 0:
        failed_packages = [pkg for pkg, success in installation_results.items() if not success]
        failed_packages_str = ", ".join(failed_packages[:5])
        if len(failed_packages) > 5:
            failed_packages_str += f" dan {len(failed_packages)-5} lainnya"
            
        if logger:
            logger.warning(f"\n⚠️  Paket yang Gagal: {failed_packages_str}")
    
    # Generate dan display detailed report
    report_html = generate_installation_summary_report(installation_results, duration)
    
    # Display report in log output
    log_output = ui_components.get('log_output')
    if log_output and hasattr(log_output, 'clear_output'):
        from IPython.display import display, HTML
        with log_output:
            display(HTML(report_html))
    
    # Update status panel berdasarkan hasil
    if success_count == total_count:
        status_msg = f"✅ Instalasi berhasil: {success_count}/{total_count} paket"
        if logger:
            logger.success(f"🎉 Instalasi selesai: Semua {success_count} paket berhasil diinstal")
        update_status_panel(ui_components, status_msg, "success")
    elif success_count > 0:
        status_msg = f"⚠️  Instalasi sebagian: {success_count} berhasil, {failed_count} gagal"
        if logger:
            logger.warning(f"⚠️  Instalasi sebagian: {failed_count} paket gagal")
        update_status_panel(ui_components, status_msg, "warning")
    else:
        status_msg = f"❌ Instalasi gagal: {failed_count}/{total_count} packages gagal"
        if logger:
            logger.error(f"💥 Installation failed: all {failed_count} packages failed")
        update_status_panel(ui_components, status_msg, "error")
    
    # Update package status di UI
    status_mapping = {
        package.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip(): 
        'installed' if success else 'error'
        for package, success in installation_results.items()
    }
    
    batch_update_package_status(ui_components, status_mapping)
    if logger:
        logger.info(f"🔄 Updated UI status untuk {len(status_mapping)} packages")