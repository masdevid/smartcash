"""
File: smartcash/ui/setup/dependency_installer/handlers/install_handler.py
Deskripsi: Handler untuk instalasi package
"""

from typing import Dict, Any, List
from IPython.display import display, clear_output

def setup_install_handler(ui_components: Dict[str, Any]) -> None:
    """
    Setup handler untuk tombol install
    
    Args:
        ui_components: Dictionary UI components
    """
    # Register event handler
    ui_components['install_button'].on_click(
        lambda b: on_install_click(b, ui_components)
    )

def on_install_click(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol install
    
    Args:
        b: Button widget
        ui_components: Dictionary UI components
    """
    # Import komponen standar
    from smartcash.ui.utils.alert_utils import create_info_alert
    from smartcash.ui.utils.metric_utils import create_metric_display
    from smartcash.ui.utils.fallback_utils import update_status_panel
    
    # Import utils
    from smartcash.ui.setup.dependency_installer.utils.package_utils import analyze_installed_packages
    from smartcash.ui.setup.dependency_installer.handlers.package_handler import get_all_missing_packages, run_batch_installation
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message
    
    # Dapatkan packages yang perlu diinstall
    missing_packages = get_all_missing_packages(ui_components)
    
    # Reset progress bar dan label
    progress_bar = ui_components.get('install_progress')
    progress_label = ui_components.get('progress_label')
    
    if progress_bar:
        progress_bar.value = 0
        if hasattr(progress_bar, 'layout') and hasattr(progress_bar.layout, 'visibility'):
            progress_bar.layout.visibility = 'visible'
    
    if progress_label:
        progress_label.value = "Memulai instalasi packages..."
        if hasattr(progress_label, 'layout') and hasattr(progress_label.layout, 'visibility'):
            progress_label.layout.visibility = 'visible'
    
    # Reset tracker jika tersedia
    tracker = ui_components.get('dependency_installer_tracker')
    if tracker:
        tracker.reset()
    
    # Display ringkasan menggunakan logger helper
    log_message(ui_components, f"Memulai instalasi {len(missing_packages)} package", "info")
    status_output = ui_components.get('status')
    
    if status_output:
        with status_output:
            clear_output()
            display(create_info_alert(
                f"Memulai instalasi {len(missing_packages)} package",
                'info',
                '🚀'
            ))
    
    # Jalankan instalasi
    success, stats = run_batch_installation(missing_packages, ui_components)
    
    # Log ringkasan dengan namespace khusus
    log_message(
        ui_components, 
        f"Instalasi selesai: {stats['success']}/{stats['total']} berhasil, {stats['failed']} gagal ({stats['duration']:.1f} detik)",
        "success" if success else "warning"
    )
    
    # Tampilkan ringkasan hasil
    if status_output:
        with status_output:
            # Header ringkasan
            display(create_info_alert(
                f"Ringkasan Instalasi ({stats['duration']:.1f} detik)",
                'success' if success else 'warning',
                '✅' if success else '⚠️'
            ))
            
            # Metrik
            display(create_metric_display("Total", stats['total']))
            display(create_metric_display("Berhasil", stats['success'], is_good=stats['success'] > 0))
            display(create_metric_display("Gagal", stats['failed'], is_good=stats['failed'] == 0))
            display(create_metric_display("Waktu", f"{stats['duration']:.1f} detik"))
            
            # Error details jika ada
            if stats['errors']:
                error_details = "<br>".join([f"❌ {pkg}: {err}" for pkg, err in stats['errors']])
                display(create_info_alert(
                    f"<h4>Detail Error</h4><div>{error_details}</div>",
                    'error',
                    '❌'
                ))
                
                # Log error details
                for pkg, err in stats['errors']:
                    log_message(ui_components, f"Instalasi gagal untuk {pkg}: {err}", "error")
    
    # Update status panel
    completion_status = "success" if success else "warning"
    update_status_panel(
        ui_components, 
        f"{'✅' if success else '⚠️'} Instalasi selesai: {stats['success']}/{stats['total']} berhasil, {stats['failed']} gagal",
        completion_status
    )
    
    # Jalankan deteksi ulang untuk update status
    analyze_installed_packages(ui_components)
