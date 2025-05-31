"""
File: smartcash/ui/setup/dependency_installer/handlers/install_handler.py
Deskripsi: Enhanced install handler dengan progress tracking yang lebih informatif
"""

from typing import Dict, Any, List
from IPython.display import display, clear_output

def setup_install_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol install"""
    ui_components['install_button'].on_click(lambda b: on_install_click(b, ui_components))

def on_install_click(b, ui_components: Dict[str, Any]) -> None:
    """Handler untuk tombol install dengan progress tracking yang lebih informatif"""
    # Import dari existing implementations
    from smartcash.ui.utils.alert_utils import create_info_alert
    from smartcash.ui.utils.fallback_utils import show_status_safe
    
    # Import utils
    from smartcash.ui.setup.dependency_installer.utils.package_utils import analyze_installed_packages
    from smartcash.ui.setup.dependency_installer.handlers.package_handler import get_all_missing_packages, run_batch_installation
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message
    
    # Reset progress tracking dengan tahapan yang jelas
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        progress_tracker.reset()
        progress_tracker.show('install')
        # Inisialisasi tahapan proses
        progress_tracker.update('step', 0, "Memulai proses instalasi...", "#007bff")
    
    # Display ringkasan
    log_message(ui_components, f"🚀 Memulai proses instalasi dependency", "info")
    show_status_safe("Memulai proses instalasi dependency", 'info', ui_components)
    
    # Dapatkan packages yang perlu diinstall dengan progress tracking
    missing_packages = get_all_missing_packages(ui_components)
    
    # Update progress tracking setelah analisis
    if progress_tracker:
        if len(missing_packages) > 0:
            progress_tracker.update('overall', 25, f"Ditemukan {len(missing_packages)} package untuk diinstall", "#007bff")
            log_message(ui_components, f"🔍 Ditemukan {len(missing_packages)} package untuk diinstall", "info")
        else:
            progress_tracker.update('overall', 100, "Semua package sudah terinstall", "#28a745")
            progress_tracker.update('step', 100, "Analisis selesai", "#28a745")
            log_message(ui_components, "✅ Semua package sudah terinstall", "success")
            show_status_safe("Semua package sudah terinstall", 'success', ui_components)
            return
    
    # Jalankan instalasi dengan progress tracking
    success, stats = run_batch_installation(missing_packages, ui_components)
    
    # Log ringkasan dengan parameter numerik yang di-highlight
    log_message(
        ui_components, 
        f"📊 Instalasi selesai: {stats['success']}/{stats['total']} berhasil, {stats['failed']} gagal ({stats['duration']:.1f}s)",
        "success" if success else "warning"
    )
    
    # Update progress tracking
    if progress_tracker:
        if success:
            progress_tracker.complete(f"Instalasi selesai: {stats['success']}/{stats['total']} berhasil")
        else:
            progress_tracker.error(f"Instalasi selesai dengan {stats['failed']} error")
    
    # Show final status
    final_message = f"{'✅' if success else '⚠️'} Instalasi selesai: {stats['success']}/{stats['total']} berhasil"
    show_status_safe(final_message, 'success' if success else 'warning', ui_components)
    
    # Error details jika ada
    if stats['errors'] and ui_components.get('status'):
        with ui_components['status']:
            error_details = "<br>".join([f"❌ {pkg}: {err}" for pkg, err in stats['errors']])
            display(create_info_alert(f"<h4>Detail Error</h4><div>{error_details}</div>", 'error', '❌'))
    
    # Jalankan deteksi ulang
    analyze_installed_packages(ui_components)