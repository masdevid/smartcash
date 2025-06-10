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
        # Step 1: Get selected packages dengan emoji untuk visual feedback
        try:
            if progress_tracker:
                # Initialize progress tracker for installation
                if hasattr(progress_tracker, 'update_overall'):
                    # Dual progress tracker
                    progress_tracker.update_overall(0, "🔍 Mempersiapkan instalasi...", "info")
                    progress_tracker.update_current(0, "Menunggu...", "info")
                elif hasattr(progress_tracker, 'show'):
                    # Legacy progress tracker
                    progress_tracker.show("Instalasi Packages", [
                        "🔍 Analisis", 
                        "📦 Instalasi", 
                        "✅ Verifikasi"
                    ])
                
                # Progress tracking is now handled by the dual progress tracker initialization above
            else:
                # Fallback untuk progress tracking lama
                ctx.stepped_progress('INSTALL_INIT', "🚀 Mempersiapkan instalasi...")
        except Exception as e:
            # Silent fail untuk compatibility
            if logger:
                logger.debug(f"🔄 Progress tracker error (non-critical): {str(e)}")
            # Tetap lanjutkan proses
        
        log_to_ui_safe(ui_components, "🚀 Memulai proses instalasi packages")
        
        selected_packages = get_selected_packages(ui_components)
        if not selected_packages:
            update_status_panel(ui_components, "❌ Tidak ada packages yang dipilih", "error")
            log_to_ui_safe(ui_components, "⚠️ Tidak ada packages yang dipilih untuk instalasi", "warning")
            return
        
        # Step 2: Filter uninstalled packages dengan emoji untuk visual feedback
        try:
            if progress_tracker:
                if hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(30, "📊 Menganalisis packages...")
                if hasattr(progress_tracker, 'update_current'):
                    progress_tracker.update_current(50, f"🔍 Menganalisis {len(selected_packages)} packages...")
            else:
                ctx.stepped_progress('INSTALL_ANALYSIS', "📊 Menganalisis packages...")
        except Exception as e:
            # Silent fail untuk compatibility
            if logger:
                logger.debug(f"🔄 Progress tracker error (non-critical): {str(e)}")
            # Tetap lanjutkan proses
        
        log_to_ui_safe(ui_components, f"📦 Menganalisis {len(selected_packages)} packages yang dipilih")
        
        def package_logger_func(msg):
            log_to_ui_safe(ui_components, msg)
        
        packages_to_install = filter_uninstalled_packages(selected_packages, package_logger_func)
        
        if not packages_to_install:
            log_to_ui_safe(ui_components, "✅ Semua packages sudah terinstall dengan benar")
            
            # Complete operation dengan progress tracker baru dan emoji dengan safe error handling
            try:
                progress_tracker = ui_components.get('progress_tracker')
                if progress_tracker:
                    if hasattr(progress_tracker, 'update_overall'):
                        progress_tracker.update_overall(100, "✅ Semua packages sudah terinstall")
                    if hasattr(progress_tracker, 'update_current'):
                        progress_tracker.update_current(100, "✅ Complete")
                    if hasattr(progress_tracker, 'complete'):
                        progress_tracker.complete("✅ Semua packages sudah terinstall dengan benar")
                else:
                    # Fallback untuk progress tracking lama
                    update_progress = ui_components.get('update_progress')
                    if update_progress and callable(update_progress):
                        update_progress('overall', 100, "✅ Semua packages sudah terinstall")
                        update_progress('step', 100, "✅ Complete")
                    
                    complete_operation = ui_components.get('complete_operation')
                    if complete_operation and callable(complete_operation):
                        complete_operation("✅ Semua packages sudah terinstall dengan benar")
            except Exception as e:
                # Silent fail untuk compatibility
                if logger:
                    logger.debug(f"🔄 Progress tracker completion error (non-critical): {str(e)}")
                # Tetap lanjutkan proses
            
            update_status_panel(ui_components, "✅ Semua packages sudah terinstall", "success")
            
            # Hide progress bars setelah delay dengan threading untuk non-blocking
            import threading
            def hide_progress_delayed():
                time.sleep(2)
                try:
                    if progress_tracker and hasattr(progress_tracker, 'reset'):
                        # Reset semua level progress tracker dengan safe error handling
                        progress_tracker.reset()
                except Exception as e:
                    # Silent fail untuk compatibility
                    if logger:
                        logger.debug(f"🔄 Progress tracker reset error (non-critical): {str(e)}")
                    if hasattr(progress_tracker, 'update_current'):
                        progress_tracker.update_current(0, "")
                    if hasattr(progress_tracker, 'update_step_progress'):
                        progress_tracker.update_step_progress(0, "")
                else:
                    ui_components.get('reset_all', lambda: None)()
            
            threading.Thread(target=hide_progress_delayed, daemon=True).start()
            return
        
        # Step 3: Install packages dengan emoji untuk visual feedback
        try:
            if progress_tracker:
                if hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(50, "📦 Menginstall packages...")
                if hasattr(progress_tracker, 'update_current'):
                    progress_tracker.update_current(0, f"📦 Memulai instalasi {len(packages_to_install)} packages...")
            else:
                ctx.stepped_progress('INSTALL_PACKAGES', "📦 Menginstall packages...")
        except Exception as e:
            # Silent fail untuk compatibility
            if logger:
                logger.debug(f"🔄 Progress tracker error (non-critical): {str(e)}")
            # Tetap lanjutkan proses
        
        log_to_ui_safe(ui_components, f"📦 Installing {len(packages_to_install)} packages dengan parallel processing")
        
        installation_results = _install_packages_parallel_with_utils(
            packages_to_install, ui_components, config, package_logger_func, ctx, logger
        )
        
        # Step 4: Generate report dengan emoji untuk visual feedback
        try:
            if progress_tracker:
                if hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(80, "📊 Membuat laporan...")
                if hasattr(progress_tracker, 'update_current'):
                    progress_tracker.update_current(0, "📊 Membuat laporan instalasi...")
            else:
                ctx.stepped_progress('INSTALL_REPORT', "📊 Membuat laporan...")
        except Exception as e:
            # Silent fail untuk compatibility
            if logger:
                logger.debug(f"🔄 Progress tracker error (non-critical): {str(e)}")
            # Tetap lanjutkan proses
        
        log_to_ui_safe(ui_components, "📊 Generating installation report...")
        
        log_to_ui_safe(ui_components, f"⏱️ Installation selesai dalam {time.time() - start_time:.1f} detik")
        
        # Update all package status dan generate report
        _finalize_installation_results(ui_components, installation_results, time.time() - start_time, logger)
        
        # Complete operation dengan progress tracker baru dan emoji dengan safe error handling
        try:
            if progress_tracker:
                if hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(100, "✅ Instalasi selesai")
                if hasattr(progress_tracker, 'update_current'):
                    progress_tracker.update_current(100, "✅ Complete")
                if hasattr(progress_tracker, 'complete'):
                    progress_tracker.complete("✅ Instalasi package selesai")
            else:
                # Fallback untuk progress tracking lama
                update_progress = ui_components.get('update_progress')
                if update_progress and callable(update_progress):
                    update_progress('overall', 100, "✅ Instalasi selesai")
                    update_progress('step', 100, "✅ Complete")
                
                complete_operation = ui_components.get('complete_operation')
                if complete_operation and callable(complete_operation):
                    complete_operation("✅ Instalasi package selesai")
        except Exception as e:
            # Silent fail untuk compatibility
            if logger:
                logger.debug(f"🔄 Progress tracker completion error (non-critical): {str(e)}")
            # Tetap lanjutkan proses
        
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
        
        # Get progress tracker if available
        progress_tracker = ui_components.get('progress_tracker')
        if not progress_tracker:
            return
            
        # Prepare status message and color
        status_icon = "✅" if success else "❌"
        status_msg = "Berhasil" if success else "Gagal"
        color = "success" if success else "error"
        
        try:
            # Update progress based on tracker type
            if hasattr(progress_tracker, 'update_overall'):
                # Dual progress tracker
                overall_progress = 50 + int((completed_count / total_packages) * 30)
                progress_tracker.update_overall(
                    overall_progress, 
                    f"📦 Memproses {completed_count}/{total_packages} paket",
                    "info"
                )
                
                # Update current package status
                progress_tracker.update_current(
                    progress,
                    f"{status_icon} {package}: {status_msg}",
                    color
                )
                
                # Update status if available
                if hasattr(progress_tracker, 'update_status'):
                    progress_tracker.update_status(
                        f"{status_icon} {package}: {status_msg}",
                        level='info'
                    )
            elif hasattr(progress_tracker, 'update'):
                # Fallback to single progress bar
                progress_tracker.update(
                    progress,
                    f"{status_icon} {package}: {status_msg}",
                    color
                )
        except Exception as e:
            log_to_ui_safe(ui_components, f"Error updating progress: {str(e)}", "error")
        
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
                    error_msg = f"❌ Instalasi gagal: {str(e)}"
                    logger_func(error_msg)
                    
                    try:
                        if progress_tracker:
                            if hasattr(progress_tracker, 'error'):
                                progress_tracker.error(error_msg, delay=1.0)
                        else:
                            # Fallback untuk progress tracking lama
                            error_operation = ui_components.get('error_operation')
                            if error_operation and callable(error_operation):
                                error_operation(error_msg)
                    except Exception as err:
                        # Silent fail untuk compatibility
                        if logger:
                            logger.debug(f"🔄 Progress tracker error handling failed (non-critical): {str(err)}")
                    
                    update_status_panel(ui_components, error_msg, "error")
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
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    
    # Get progress tracker if available
    progress_tracker = ui_components.get('progress_tracker')
    
    # Prepare summary message
    summary_msg = (
        f"📊 Ringkasan Instalasi:\n"
        f"   ✅ Berhasil: {success_count}/{total_count}\n"
        f"   ❌ Gagal: {failed_count}\n"
        f"   ⏱️  Durasi: {duration:.1f} detik\n"
        f"   📈 Tingkat Keberhasilan: {success_rate:.1f}%"
    )
    
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
    
    # Update progress tracker if available
    if progress_tracker:
        try:
            # Update to 100% complete
            if hasattr(progress_tracker, 'update_overall'):
                progress_tracker.update_overall(
                    100,
                    "✅ Instalasi Selesai",
                    "success" if success_count == total_count else "warning"
                )
                progress_tracker.update_current(
                    100,
                    f"✅ {success_count} paket berhasil diinstal" if success_count > 0 else "❌ Gagal menginstal paket",
                    "success" if success_count > 0 else "error"
                )
                
                # Update status if available
                if hasattr(progress_tracker, 'update_status'):
                    progress_tracker.update_status(
                        summary_msg,
                        level='info'
                    )
            elif hasattr(progress_tracker, 'update'):
                # Fallback to single progress bar
                progress_tracker.update(
                    100,
                    f"✅ {success_count}/{total_count} paket berhasil" if success_count > 0 else "❌ Gagal",
                    "success" if success_count > 0 else "error"
                )
        except Exception as e:
            if logger:
                logger.error(f"Gagal memperbarui progress tracker: {str(e)}")
    
    # Display report in log output
    log_output = ui_components.get('log_output')
    if log_output:
        from IPython.display import display, HTML
        with log_output:
            display(HTML(report_html))
    
    # Update status panel
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
    
    # Update package status
    status_mapping = {
        package.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip(): 
        'installed' if success else 'error'
        for package, success in installation_results.items()
    }
    
    batch_update_package_status(ui_components, status_mapping)
    if logger:
        logger.info(f"🔄 Updated UI status untuk {len(status_mapping)} packages")