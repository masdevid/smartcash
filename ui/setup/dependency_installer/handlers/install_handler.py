"""
File: smartcash/ui/setup/dependency_installer/handlers/install_handler.py
Deskripsi: Enhanced install handler dengan integrasi observer pattern dan progress tracking yang konsisten
"""

from typing import Dict, Any, List
import time
from IPython.display import display, clear_output

def setup_install_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol install"""
    ui_components['install_button'].on_click(lambda b: on_install_click(b, ui_components))

def on_install_click(b, ui_components: Dict[str, Any]) -> None:
    """Handler untuk tombol install dengan integrasi observer pattern dan progress tracking yang konsisten"""
    # Import dari existing implementations
    from smartcash.ui.utils.alert_utils import create_info_alert
    from smartcash.ui.utils.fallback_utils import show_status_safe
    
    # Import utils
    from smartcash.ui.setup.dependency_installer.utils.package_utils import analyze_installed_packages
    from smartcash.ui.setup.dependency_installer.handlers.package_handler import get_all_missing_packages, run_batch_installation
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message
    
    # Get observer manager
    observer_manager = ui_components.get('observer_manager')
    
    # Reset progress tracking dengan tahapan yang jelas
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        progress_tracker.reset()
        progress_tracker.show('install')
        # Inisialisasi tahapan proses
        progress_tracker.update('step', 0, "Memulai proses instalasi...", "#007bff")
    elif 'show_for_operation' in ui_components and 'update_progress' in ui_components:
        ui_components['show_for_operation']('install')
        ui_components['update_progress']('step', 0, "Memulai proses instalasi...", "#007bff")
    
    # Display ringkasan
    log_message(ui_components, f"🚀 Memulai proses instalasi dependency", "info")
    show_status_safe("Memulai proses instalasi dependency", 'info', ui_components)
    
    # Notify start via observer
    if observer_manager and hasattr(observer_manager, 'notify'):
        try:
            observer_manager.notify('DEPENDENCY_INSTALL_START', None, {
                'message': "Memulai proses instalasi dependency",
                'timestamp': time.time()
            })
        except Exception:
            pass  # Silent fail untuk observer notification
    
    # Update progress untuk tahap analisis
    if progress_tracker:
        progress_tracker.update('step', 25, "Menganalisis packages yang perlu diinstall...", "#007bff")
    elif 'update_progress' in ui_components:
        ui_components['update_progress']('step', 25, "Menganalisis packages yang perlu diinstall...", "#007bff")
    
    # Dapatkan packages yang perlu diinstall dengan progress tracking
    missing_packages = get_all_missing_packages(ui_components)
    
    # Update progress tracking setelah analisis
    if len(missing_packages) > 0:
        # Update progress tracker
        if progress_tracker:
            progress_tracker.update('overall', 25, f"Ditemukan {len(missing_packages)} package untuk diinstall", "#007bff")
        elif 'update_progress' in ui_components:
            ui_components['update_progress']('overall', 25, f"Ditemukan {len(missing_packages)} package untuk diinstall", "#007bff")
        
        # Log message
        log_message(ui_components, f"🔍 Ditemukan {len(missing_packages)} package untuk diinstall", "info")
        
        # Notify via observer
        if observer_manager and hasattr(observer_manager, 'notify'):
            try:
                observer_manager.notify('DEPENDENCY_ANALYZE_COMPLETE', None, {
                    'message': f"Ditemukan {len(missing_packages)} package untuk diinstall",
                    'timestamp': time.time(),
                    'missing_packages': missing_packages,
                    'count': len(missing_packages)
                })
            except Exception:
                pass  # Silent fail untuk observer notification
    else:
        # Update progress tracker
        if progress_tracker:
            progress_tracker.update('overall', 100, "Semua package sudah terinstall", "#28a745")
            progress_tracker.update('step', 100, "Analisis selesai", "#28a745")
        elif 'update_progress' in ui_components:
            ui_components['update_progress']('overall', 100, "Semua package sudah terinstall", "#28a745")
            ui_components['update_progress']('step', 100, "Analisis selesai", "#28a745")
        
        # Log message
        log_message(ui_components, "✅ Semua package sudah terinstall", "success")
        show_status_safe("Semua package sudah terinstall", 'success', ui_components)
        
        # Notify via observer
        if observer_manager and hasattr(observer_manager, 'notify'):
            try:
                observer_manager.notify('DEPENDENCY_INSTALL_COMPLETE', None, {
                    'message': "Semua package sudah terinstall",
                    'timestamp': time.time(),
                    'success': True,
                    'missing_packages': [],
                    'count': 0
                })
            except Exception:
                pass  # Silent fail untuk observer notification
        
        # Complete operation jika tersedia
        if 'complete_operation' in ui_components:
            ui_components['complete_operation']("Semua package sudah terinstall")
        
        return
    
    # Jalankan instalasi dengan progress tracking
    success, stats = run_batch_installation(missing_packages, ui_components)
    
    # Format pesan ringkasan dengan parameter numerik yang di-highlight
    summary_message = f"📊 Instalasi selesai: {stats['success']}/{stats['total']} berhasil, {stats['failed']} gagal ({stats['duration']:.1f}s)"
    
    # Log ringkasan
    log_message(
        ui_components, 
        summary_message,
        "success" if success else "warning"
    )
    
    # Update progress tracking
    if progress_tracker:
        if success:
            progress_tracker.complete(f"Instalasi selesai: {stats['success']}/{stats['total']} berhasil")
        else:
            progress_tracker.error(f"Instalasi selesai dengan {stats['failed']} error")
    elif 'complete_operation' in ui_components and 'error_operation' in ui_components:
        if success:
            ui_components['complete_operation'](f"Instalasi selesai: {stats['success']}/{stats['total']} berhasil")
        else:
            ui_components['error_operation'](f"Instalasi selesai dengan {stats['failed']} error")
    
    # Notify via observer
    if observer_manager and hasattr(observer_manager, 'notify'):
        try:
            observer_manager.notify('DEPENDENCY_INSTALL_COMPLETE', None, {
                'message': summary_message,
                'timestamp': time.time(),
                'success': success,
                'stats': stats
            })
        except Exception:
            pass  # Silent fail untuk observer notification
    
    # Show final status
    final_message = f"{'✅' if success else '⚠️'} Instalasi selesai: {stats['success']}/{stats['total']} berhasil"
    show_status_safe(final_message, 'success' if success else 'warning', ui_components)
    
    # Error details jika ada
    if stats['errors'] and ui_components.get('status'):
        with ui_components['status']:
            error_details = "<br>".join([f"❌ {pkg}: {err}" for pkg, err in stats['errors']])
            display(create_info_alert(f"<h4>Detail Error</h4><div>{error_details}</div>", 'error', '❌'))
            
            # Notify error details via observer
            if observer_manager and hasattr(observer_manager, 'notify'):
                try:
                    observer_manager.notify('DEPENDENCY_INSTALL_ERROR', None, {
                        'message': f"Terdapat {len(stats['errors'])} error pada instalasi",
                        'timestamp': time.time(),
                        'errors': stats['errors']
                    })
                except Exception:
                    pass  # Silent fail untuk observer notification
    
    # Jalankan deteksi ulang dengan progress tracking
    try:
        # Tampilkan progress tracker untuk analisis final
        if progress_tracker:
            progress_tracker.update('step', 75, "Menjalankan deteksi ulang packages...", "#007bff")
        elif 'update_progress' in ui_components:
            ui_components['update_progress']('step', 75, "Menjalankan deteksi ulang packages...", "#007bff")
        
        # Log message
        log_message(ui_components, "🔄 Menjalankan deteksi ulang packages...", "info")
        
        # Jalankan analisis
        analyze_installed_packages(ui_components)
        
        # Update progress
        if progress_tracker:
            progress_tracker.update('step', 100, "Deteksi ulang selesai", "#28a745")
        elif 'update_progress' in ui_components:
            ui_components['update_progress']('step', 100, "Deteksi ulang selesai", "#28a745")
            
        # Log success
        log_message(ui_components, "✅ Deteksi ulang packages selesai", "success")
    except Exception as e:
        # Log error
        log_message(ui_components, f"⚠️ Gagal menjalankan deteksi ulang: {str(e)}", "error")
        
        # Notify error via observer
        if observer_manager and hasattr(observer_manager, 'notify'):
            try:
                observer_manager.notify('DEPENDENCY_ANALYZE_ERROR', None, {
                    'message': f"Gagal menjalankan deteksi ulang: {str(e)}",
                    'timestamp': time.time(),
                    'error': str(e)
                })
            except Exception:
                pass  # Silent fail untuk observer notification