"""
File: smartcash/ui/setup/dependency_installer/handlers/button_handlers.py
Deskripsi: Handler untuk tombol-tombol di dependency installer
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display

from smartcash.ui.setup.dependency_installer.utils.ui_utils import toggle_ui_visibility, reset_ui_logs
from smartcash.ui.setup.dependency_installer.utils.package_installer import install_required_packages
from smartcash.ui.setup.dependency_installer.utils.logger_helper import get_module_logger, log_message

def setup_install_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol install dengan reset log dan tampilkan progress
    
    Args:
        ui_components: Dictionary komponen UI
    """
    if 'install_button' not in ui_components or not hasattr(ui_components['install_button'], 'on_click'):
        return
    
    # Hapus handler yang mungkin sudah ada
    ui_components['install_button']._click_handlers.callbacks.clear()
    
    # Tambahkan handler yang menggunakan install_required_packages dari package_installer.py
    def install_click_handler(b):
        # Setup logger
        logger = get_module_logger()
        logger.info("Install button clicked")
        
        # Reset log output
        reset_ui_logs(ui_components)
        logger.debug("UI logs reset")
        
        # Tampilkan progress container
        toggle_ui_visibility(ui_components, 'progress_container', True)
        logger.debug("Progress container visibility set to visible")
        
        # Jalankan analisis jika belum ada hasil analisis
        if 'analysis_result' not in ui_components and 'run_delayed_analysis' in ui_components:
            log_message(ui_components, "🔍 Menjalankan analisis package terlebih dahulu...", "info")
            logger.info("Running delayed analysis before installation")
            ui_components['run_delayed_analysis']()
        
        # Install package yang dibutuhkan menggunakan fungsi dari package_installer.py
        logger.info("Starting package installation")
        install_required_packages(ui_components)
    
    ui_components['install_button'].on_click(install_click_handler)

def setup_reset_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol reset
    
    Args:
        ui_components: Dictionary komponen UI
    """
    if 'reset_button' not in ui_components or not hasattr(ui_components['reset_button'], 'on_click'):
        return
    
    def reset_handler(b):
        # Setup logger
        logger = get_module_logger()
        logger.info("Reset button clicked")
        
        # Reset log output
        reset_ui_logs(ui_components)
        logger.debug("UI logs reset")
        
        # Sembunyikan progress container
        toggle_ui_visibility(ui_components, 'progress_container', False)
        logger.debug("Progress container visibility set to hidden")
        
        # Reset progress bar menggunakan fungsi dari logger_helper
        from smartcash.ui.setup.dependency_installer.utils.logger_helper import reset_progress_bar as helper_reset_progress_bar
        try:
            helper_reset_progress_bar(ui_components, 0, "Reset selesai", False)
            logger.debug("Progress bar reset using helper function")
        except Exception as e:
            logger.debug(f"Fallback to local reset_progress_bar: {str(e)}")
            if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
                ui_components['reset_progress_bar'](0, "Reset selesai", False)
        
        # Log reset menggunakan log_message dari logger_helper
        log_message(ui_components, "🔄 UI berhasil direset", "info")
        logger.info("UI reset completed successfully")
    
    ui_components['reset_button'].on_click(reset_handler)
