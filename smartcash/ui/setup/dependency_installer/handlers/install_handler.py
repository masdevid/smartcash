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
    """Handler untuk tombol install dengan integrasi observer pattern dan progress tracking yang konsisten menggunakan pendekatan one-liner"""
    # Import dari existing implementations
    from smartcash.ui.utils.alert_utils import create_info_alert
    from smartcash.ui.utils.fallback_utils import show_status_safe
    
    # Import utils
    from smartcash.ui.setup.dependency_installer.utils.package_utils import analyze_installed_packages
    from smartcash.ui.setup.dependency_installer.handlers.package_handler import get_all_missing_packages, run_batch_installation
    
    # Get observer manager
    observer_manager = ui_components.get('observer_manager')
    
    # Reset progress tracking dengan pendekatan one-liner
    ui_components['show_for_operation']('install')
    ui_components['reset_progress_bar'](0, "Memulai proses instalasi...", True)
    
    # Display ringkasan dengan pendekatan one-liner
    start_message = "🚀 Memulai proses instalasi dependency"
    ui_components['log_message'](start_message, "info")
    ui_components['update_status_panel']("info", start_message)
    
    # Notify start via observer dengan pendekatan one-liner
    try:
        observer_manager.notify('DEPENDENCY_INSTALL_START', None, {
            'message': start_message,
            'timestamp': time.time()
        }) if observer_manager and hasattr(observer_manager, 'notify') else None
    except Exception:
        pass  # Silent fail untuk observer notification
    
    # Update progress untuk tahap analisis dengan pendekatan one-liner
    ui_components['update_progress']('step', 25, "Menganalisis packages yang perlu diinstall...", "#007bff")
    
    # Dapatkan packages yang perlu diinstall dengan progress tracking
    missing_packages = get_all_missing_packages(ui_components)
    
    # Update progress tracking setelah analisis dengan pendekatan one-liner
    if len(missing_packages) > 0:
        # Update progress dan log dengan pendekatan one-liner
        analysis_message = f"Ditemukan {len(missing_packages)} package untuk diinstall"
        ui_components['update_progress']('overall', 25, analysis_message, "#007bff")
        ui_components['log_message'](f"🔍 {analysis_message}", "info")
        
        # Notify via observer dengan pendekatan one-liner
        try:
            observer_manager.notify('DEPENDENCY_ANALYZE_COMPLETE', None, {
                'message': analysis_message,
                'timestamp': time.time(),
                'missing_packages': missing_packages,
                'count': len(missing_packages)
            }) if observer_manager and hasattr(observer_manager, 'notify') else None
        except Exception:
            pass  # Silent fail untuk observer notification
    else:
        # Tidak ada package yang perlu diinstall - handle dengan pendekatan one-liner
        complete_message = "Semua package sudah terinstall"
        ui_components['complete_operation'](complete_message)
        ui_components['log_message'](f"✅ {complete_message}", "success")
        ui_components['update_status_panel']("success", complete_message)
        
        # Notify via observer dengan pendekatan one-liner
        try:
            observer_manager.notify('DEPENDENCY_INSTALL_COMPLETE', None, {
                'message': complete_message,
                'timestamp': time.time(),
                'success': True,
                'stats': {'total': 0, 'success': 0, 'failed': 0, 'errors': []}
            }) if observer_manager and hasattr(observer_manager, 'notify') else None
        except Exception:
            pass  # Silent fail untuk observer notification
        
        return
    
    # Update progress untuk tahap instalasi dengan pendekatan one-liner
    install_message = f"Menginstall {len(missing_packages)} package..."
    ui_components['update_progress']('step', 50, install_message, "#007bff")
    ui_components['log_message'](f"💾 {install_message}", "info")
    ui_components['update_status_panel']("info", install_message)
    
    # Jalankan instalasi dengan progress tracking menggunakan pendekatan one-liner
    stats = run_batch_installation(missing_packages, ui_components)
    
    # Cek hasil instalasi dengan pendekatan one-liner
    success = stats['failed'] == 0
    summary_message = f"Instalasi selesai: {stats['success']}/{stats['total']} berhasil"
    if not success:
        summary_message += f", {stats['failed']} gagal"
    
    # Log ringkasan dan update progress dengan pendekatan one-liner
    ui_components['log_message'](summary_message, "success" if success else "warning")
    
    if success:
        ui_components['complete_operation'](summary_message)
    else:
        ui_components['error_operation'](f"Instalasi selesai dengan {stats['failed']} error")
    
    # Notify via observer dengan pendekatan one-liner
    try:
        observer_manager.notify('DEPENDENCY_INSTALL_COMPLETE', None, {
            'message': summary_message,
            'timestamp': time.time(),
            'success': success,
            'stats': stats
        }) if observer_manager and hasattr(observer_manager, 'notify') else None
    except Exception:
        pass  # Silent fail untuk observer notification
    
    # Show final status dengan pendekatan one-liner
    final_message = f"{'✅' if success else '⚠️'} {summary_message}"
    ui_components['update_status_panel']("success" if success else "warning", final_message)
    
    # Error details jika ada dengan pendekatan one-liner
    if stats['errors'] and ui_components.get('status'):
        with ui_components['status']:
            error_details = "<br>".join([f"❌ {pkg}: {err}" for pkg, err in stats['errors']])
            display(create_info_alert(f"<h4>Detail Error</h4><div>{error_details}</div>", 'error', '❌'))
            
            # Notify error details via observer dengan pendekatan one-liner
            try:
                observer_manager.notify('DEPENDENCY_INSTALL_ERROR', None, {
                    'message': f"Terdapat {len(stats['errors'])} error pada instalasi",
                    'timestamp': time.time(),
                    'errors': stats['errors']
                }) if observer_manager and hasattr(observer_manager, 'notify') else None
            except Exception:
                pass  # Silent fail untuk observer notification
    
    # Jalankan deteksi ulang dengan progress tracking dan pendekatan one-liner
    try:
        # Update progress dan log dengan pendekatan one-liner
        redetect_message = "Menjalankan deteksi ulang packages..."
        ui_components['update_progress']('step', 75, redetect_message, "#007bff")
        ui_components['log_message'](f"🔄 {redetect_message}", "info")
        
        # Jalankan analisis
        analyze_installed_packages(ui_components)
        
        # Update progress dan log dengan pendekatan one-liner
        complete_message = "Deteksi ulang selesai"
        ui_components['update_progress']('step', 100, complete_message, "#28a745")
        ui_components['log_message'](f"✅ {complete_message}", "success")
    except Exception as e:
        # Handle error dengan pendekatan one-liner
        error_message = f"Gagal menjalankan deteksi ulang: {str(e)}"
        ui_components['log_message'](f"⚠️ {error_message}", "error")
        ui_components['update_status_panel']("error", error_message)
        
        # Notify error via observer dengan pendekatan one-liner
        try:
            observer_manager.notify('DEPENDENCY_ANALYZE_ERROR', None, {
                'message': error_message,
                'timestamp': time.time(),
                'error': str(e)
            }) if observer_manager and hasattr(observer_manager, 'notify') else None
        except Exception:
            pass  # Silent fail untuk observer notification