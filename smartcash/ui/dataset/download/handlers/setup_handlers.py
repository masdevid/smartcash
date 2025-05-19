"""
File: smartcash/ui/dataset/download/handlers/setup_handlers.py
Deskripsi: Setup handler untuk UI dataset download yang terintegrasi dengan observer pattern
"""

from typing import Dict, Any, Optional

def setup_download_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI dataset downloader.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup observer untuk menerima event notifikasi
    _setup_observers(ui_components)
    
    # Setup API key handler dan verifikasi
    _setup_api_key_handler(ui_components)
    
    # Setup handlers untuk UI events
    _setup_endpoint_handlers(ui_components)
    _setup_download_button_handler(ui_components)
    _setup_check_button_handler(ui_components)
    
    # Setup multi-progress tracking untuk download
    _setup_progress_tracking(ui_components)
    
    # Setup cleanup function
    _setup_cleanup(ui_components)
    
    # Save config yang sudah ada ke UI components
    ui_components['config'] = config or {}
    
    logger = ui_components.get('logger')
    if logger:
        logger.info("✅ Dataset downloader handlers berhasil diinisialisasi")
    
    return ui_components

def _setup_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observer handlers."""
    try:
        from smartcash.ui.handlers.observer_handler import setup_observer_handlers
        from smartcash.ui.dataset.download.handlers.download_progress_observer import setup_download_progress_observer
        
        # Setup basic observers
        ui_components = setup_observer_handlers(ui_components, "dataset_download_observers")
        
        # Setup download progress observer
        setup_download_progress_observer(ui_components)
    except ImportError as e:
        # Log gagal import jika logger tersedia
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"ℹ️ Observer handler tidak tersedia: {str(e)}")

def _setup_api_key_handler(ui_components: Dict[str, Any]) -> None:
    """Setup API key handler untuk Roboflow."""
    try:
        from smartcash.ui.dataset.download.handlers.api_key_handler import setup_api_key_input
        setup_api_key_input(ui_components)
    except ImportError:
        # Log gagal import jika logger tersedia
        logger = ui_components.get('logger')
        if logger:
            logger.debug("ℹ️ API key handler tidak tersedia")

def _setup_endpoint_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup handlers untuk endpoint Roboflow."""
    # Fungsi ini dipertahankan untuk kompatibilitas dengan kode lama
    # tetapi tidak lagi memerlukan handler untuk dropdown karena kita hanya menggunakan Roboflow
    
    logger = ui_components.get('logger')
    if logger:
        logger.debug("ℹ️ Setup endpoint handler tidak diperlukan karena hanya menggunakan Roboflow")

def _setup_download_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol download."""
    from smartcash.ui.dataset.download.handlers.download_handler import handle_download_button_click
    
    if 'download_button' in ui_components:
        ui_components['download_button'].on_click(
            lambda b: handle_download_button_click(b, ui_components)
        )

def _setup_check_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol check status."""
    from smartcash.ui.dataset.download.handlers.check_handler import handle_check_button_click
    
    if 'check_button' in ui_components:
        ui_components['check_button'].on_click(
            lambda b: handle_check_button_click(b, ui_components)
        )

def _setup_progress_tracking(ui_components: Dict[str, Any]) -> None:
    """Setup progress tracking untuk download."""
    try:
        from smartcash.ui.handlers.multi_progress import setup_multi_progress_tracking
        
        # Setup multi-progress tracking dengan tracker untuk keseluruhan dan step
        setup_multi_progress_tracking(
            ui_components=ui_components,
            overall_tracker_name="download",
            step_tracker_name="download_step",
            overall_progress_key='progress_bar',
            step_progress_key='progress_bar',
            overall_label_key='overall_label',
            step_label_key='step_label'
        )
        
        # Log setup berhasil jika ada logger
        logger = ui_components.get('logger')
        if logger:
            logger.debug("✅ Progress tracking berhasil disetup")
    except Exception as e:
        # Log error jika ada logger
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Error saat setup progress tracking: {str(e)}")

def _setup_cleanup(ui_components: Dict[str, Any]) -> None:
    """Setup cleanup function."""
    def cleanup_resources():
        """Fungsi untuk membersihkan resources."""
        try:
            # Reset progress
            if 'progress_bar' in ui_components:
                if hasattr(ui_components['progress_bar'], 'layout'):
                    ui_components['progress_bar'].layout.visibility = 'hidden'
                ui_components['progress_bar'].value = 0
            
            # Reset progress labels
            for label_key in ['overall_label', 'step_label', 'progress_message']:
                if label_key in ui_components and hasattr(ui_components[label_key], 'layout'):
                    ui_components[label_key].layout.visibility = 'hidden'
                    ui_components[label_key].value = ""
            
            # Unregister observer group jika ada
            if 'observer_manager' in ui_components and 'observer_group' in ui_components:
                try:
                    ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
                except Exception:
                    pass
            
            # Reset API key highlight jika ada
            if 'rf_apikey' in ui_components:
                ui_components['rf_apikey'].layout.border = ""
            
            # Reset logging
            try:
                from smartcash.ui.utils.logging_utils import reset_logging
                reset_logging()
            except ImportError:
                pass
            
            # Log cleanup
            logger = ui_components.get('logger')
            if logger:
                logger.debug("🧹 Cleanup dataset_downloader berhasil")
        except Exception as e:
            # Ignore exceptions during cleanup
            pass
    
    # Tetapkan fungsi cleanup ke ui_components
    ui_components['cleanup'] = cleanup_resources
    
    # Register cleanup dengan IPython event
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython:
            ipython.events.register('pre_run_cell', cleanup_resources)
    except (ImportError, AttributeError):
        pass
