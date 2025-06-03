"""
File: smartcash/ui/setup/dependency_installer/handlers/analyzer_handler.py
Deskripsi: Wrapper untuk kompatibilitas dengan implementasi lama, menggunakan analyzer_utils.py
"""

from typing import Dict, Any, List
from smartcash.ui.setup.dependency_installer.utils.analyzer_utils import analyze_installed_packages

def setup_analyzer_handler(ui_components: Dict[str, Any]) -> None:
    """
    Setup handler untuk analisis package (wrapper untuk kompatibilitas)
    
    Args:
        ui_components: Dictionary UI components
    """
    # Setup handler untuk tombol analyze dengan pendekatan one-liner
    def on_analyze_click(b, ui_components=ui_components):
        """Handler untuk tombol analyze (wrapper untuk kompatibilitas)"""
        try:
            # Reset log output
            if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
                ui_components['log_output'].clear_output()
            
            # Tampilkan progress container
            if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
                ui_components['progress_container'].layout.visibility = 'visible'
            
            # Jalankan analisis menggunakan fungsi dari analyzer_utils.py
            analyze_installed_packages(ui_components)
        except Exception as e:
            # Log error
            if 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message'](f"❌ Error saat menganalisis packages: {str(e)}", "error")
    
    # Expose enhanced function untuk kompatibilitas
    ui_components['analyze_installed_packages'] = lambda: analyze_installed_packages(ui_components)
    
    # Register handler jika tombol analyze tersedia
    if 'analyze_button' in ui_components:
        ui_components['analyze_button'].on_click(on_analyze_click)
