"""
File: smartcash/ui/setup/dependency_installer/handlers/analyzer_handler.py
Deskripsi: Enhanced handler untuk analisis package yang terinstall dengan integrasi observer pattern dan pendekatan DRY
"""

from typing import Dict, Any, List
import time
from smartcash.ui.setup.dependency_installer.utils.observer_helper import (
    notify_analyze_start, notify_analyze_progress, notify_analyze_error, notify_analyze_complete
)
from smartcash.ui.setup.dependency_installer.utils.progress_helper import (
    update_progress_step, start_operation, complete_operation, handle_item_error
)
from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message

def setup_analyzer_handler(ui_components: Dict[str, Any]) -> None:
    """
    Setup handler untuk analisis package dengan integrasi observer pattern dan pendekatan one-liner
    
    Args:
        ui_components: Dictionary UI components
    """
    # Import utils
    from smartcash.ui.setup.dependency_installer.utils.package_utils import analyze_installed_packages
    
    # Wrap analyze_installed_packages dengan progress tracking dan observer notification menggunakan pendekatan DRY
    def enhanced_analyze_installed_packages() -> List[str]:
        # Mulai operasi dengan progress tracking dan logging menggunakan pendekatan DRY
        start_operation(ui_components, "analisis packages", 0)
        
        # Update status panel dengan pendekatan one-liner
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            ui_components['update_status_panel']("info", "Memulai analisis packages terinstall")
        
        # Notify start via observer dengan pendekatan DRY
        notify_analyze_start(ui_components)
        
        try:
            # Jalankan analisis dengan timing
            start_time = time.time()
            result = analyze_installed_packages(ui_components)
            duration = time.time() - start_time
            
            # Selesaikan operasi dengan progress tracking dan logging menggunakan pendekatan DRY
            stats = {
                'duration': duration,
                'success': True,
                'total': 1,
                'result': result
            }
            complete_operation(ui_components, "analisis packages", stats)
            
            # Update status panel dengan pendekatan one-liner
            if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
                ui_components['update_status_panel']("success", "Analisis selesai")
            
            # Notify complete via observer dengan pendekatan DRY
            notify_analyze_complete(ui_components, duration, result)
            
            return result
            
        except Exception as e:
            # Handle error dengan pendekatan DRY
            error_msg = str(e)
            error_message = f"Gagal menganalisis packages: {error_msg}"
            handle_item_error(ui_components, "analisis packages", error_msg)
            
            # Update status panel dengan pendekatan one-liner
            if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
                ui_components['update_status_panel']("error", error_message)
            
            # Notify error via observer dengan pendekatan DRY
            notify_analyze_error(ui_components, error_msg)
            
            # Re-raise exception
            raise
    
    # Expose enhanced function
    ui_components['analyze_installed_packages'] = enhanced_analyze_installed_packages
    
    # Setup handler untuk tombol analyze dengan pendekatan one-liner
    def on_analyze_click(b, ui_components=ui_components):
        """Handler untuk tombol analyze dengan integrasi observer pattern dan progress tracking menggunakan pendekatan one-liner"""
        try:
            # Jalankan analisis packages dengan enhanced function
            enhanced_analyze_installed_packages()
        except Exception:
            # Error sudah ditangani di enhanced_analyze_installed_packages
            pass
    
    # Register handler jika tombol analyze tersedia
    if 'analyze_button' in ui_components:
        ui_components['analyze_button'].on_click(on_analyze_click)

# Fungsi on_analyze_click sudah diimplementasikan di dalam setup_analyzer_handler
