"""
File: smartcash/ui/setup/dependency_installer_handler.py
Deskripsi: Handler untuk instalasi dependencies dengan alur 3 tahap: deteksi → filter → install
"""

from typing import List, Dict, Any, Tuple
from IPython.display import display, clear_output

def setup_dependency_installer_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handlers untuk UI dependency installer.
    
    Args:
        ui_components: Dictionary UI components
        env: Environment manager (opsional)
        config: Konfigurasi aplikasi (opsional)
        
    Returns:
        Dictionary UI components yang diupdate
    """
    # Import komponen standar
    from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
    from smartcash.ui.utils.metric_utils import create_metric_display
    from smartcash.ui.utils.fallback_utils import update_status_panel
    from smartcash.ui.utils.constants import COLORS
    from smartcash.ui.handlers.single_progress import setup_progress_tracking
    
    # Import utils yang sudah dipisahkan
    from smartcash.ui.setup.package_analyzer import get_installed_packages, check_missing_packages
    from smartcash.ui.setup.package_requirements import get_package_groups
    from smartcash.ui.setup.package_installer import run_batch_installation
    
    # Setup progress tracking
    setup_progress_tracking(
        ui_components, 
        tracker_name="dependency_installer",
        progress_widget_key="install_progress",
        progress_label_key="progress_label",
        total=100,
        description="Instalasi dependencies"
    )
    
    # Dapatkan package groups
    PACKAGE_GROUPS = get_package_groups()
    
    def get_all_missing_packages(ui_components: Dict[str, Any]) -> List[str]:
        """
        Dapatkan semua package yang perlu diinstall berdasarkan UI state.
        
        Args:
            ui_components: Dictionary UI components
            
        Returns:
            List package yang perlu diinstall
        """
        # Get installed packages
        installed_packages = get_installed_packages()
        
        # Collect packages to install
        missing_packages = []
        
        # Add selected packages from groups
        for pkg_key, pkg_list in PACKAGE_GROUPS.items():
            checkbox = ui_components.get(pkg_key)
            if checkbox and checkbox.value:
                package_list = pkg_list() if callable(pkg_list) else pkg_list
                missing = check_missing_packages(package_list, installed_packages)
                missing_packages.extend(missing)
        
        # Add custom packages
        custom_text = ui_components.get('custom_packages').value.strip()
        if custom_text:
            custom_packages = [pkg.strip() for pkg in custom_text.split('\n') if pkg.strip()]
            missing = check_missing_packages(custom_packages, installed_packages)
            missing_packages.extend(missing)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(missing_packages))
    
    def analyze_installed_packages(ui_components: Dict[str, Any]) -> None:
        """
        Analisis packages yang sudah terinstall dan update UI.
        
        Args:
            ui_components: Dictionary UI components
        """
        logger = ui_components.get('logger')
        if logger: logger.info("🔍 Menganalisis packages yang terinstall...")
        
        # Dapatkan semua installed packages
        installed_packages = get_installed_packages()
        
        # Update status setiap package
        for pkg_key, pkg_list in PACKAGE_GROUPS.items():
            status_widget = ui_components.get(f"{pkg_key}_status")
            checkbox = ui_components.get(pkg_key)
            
            if not status_widget or not checkbox:
                continue
            
            # Dapatkan list package
            package_list = pkg_list() if callable(pkg_list) else pkg_list
            
            # Cek apakah semua package sudah terinstall
            missing_packages = check_missing_packages(package_list, installed_packages)
            
            if missing_packages:
                ratio = len(missing_packages) / len(package_list)
                if ratio > 0.7:  # Jika lebih dari 70% missing
                    status_widget.value = f"<div style='width:100px;color:{COLORS['danger']}'>⚠️ Belum diinstall</div>"
                    checkbox.value = True  # Set checked untuk install
                else:
                    status_widget.value = f"<div style='width:100px;color:{COLORS['warning']}'>⚠️ Perlu update</div>"
                    checkbox.value = True  # Set checked untuk install
            else:
                status_widget.value = f"<div style='width:100px;color:{COLORS['success']}'>✅ Terinstall</div>"
                checkbox.value = False  # Uncheck karena sudah terinstall
        
        # Update status panel
        total_missing = len(get_all_missing_packages(ui_components))
        if total_missing > 0:
            update_status_panel(
                ui_components, 
                f"🔍 Terdeteksi {total_missing} package yang perlu diinstall",
                "warning"
            )
        else:
            update_status_panel(
                ui_components, 
                "✅ Semua package sudah terinstall",
                "success"
            )
        
        if logger: logger.info(f"✅ Analisis package selesai: {total_missing} package perlu diinstall")
    
    def on_install_click(b):
        """Handler untuk tombol install."""
        # Dapatkan packages yang perlu diinstall
        missing_packages = get_all_missing_packages(ui_components)
        
        # Display ringkasan
        logger = ui_components.get('logger')
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
        
        # Update status panel
        completion_status = "success" if success else "warning"
        update_status_panel(
            ui_components, 
            f"{'✅' if success else '⚠️'} Instalasi selesai: {stats['success']}/{stats['total']} berhasil, {stats['failed']} gagal",
            completion_status
        )
        
        # Jalankan deteksi ulang untuk update status
        analyze_installed_packages(ui_components)
    
    # Register event handler
    ui_components['install_button'].on_click(on_install_click)
    
    # Expose function untuk analisis otomatis
    ui_components['analyze_installed_packages'] = analyze_installed_packages
    
    return ui_components