"""
File: smartcash/ui/dataset/dataset_downloader_handler.py
Deskripsi: Setup handler untuk UI dataset downloader dengan integrasi ke berbagai endpoint dan API key management
"""

from typing import Dict, Any, Optional
from IPython.display import display

def setup_dataset_downloader_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI dataset downloader.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup progress tracking
    _setup_progress_tracking(ui_components)
    
    # Setup observer untuk menerima event notifikasi
    _setup_observers(ui_components)
    
    # Setup API key handler dan verifikasi
    _setup_api_key_handler(ui_components)
    
    # Setup handlers untuk UI events
    _setup_endpoint_handlers(ui_components)
    _setup_download_button_handler(ui_components)
    _setup_check_button_handler(ui_components)
    
    # Setup cleanup function
    _setup_cleanup(ui_components)
    
    # Save config yang sudah ada ke UI components
    ui_components['config'] = config or {}
    
    logger = ui_components.get('logger')
    if logger:
        logger.info("✅ Dataset downloader handlers berhasil diinisialisasi")
    
    return ui_components

def _setup_progress_tracking(ui_components: Dict[str, Any]) -> None:
    """Setup progress tracking."""
    try:
        from smartcash.ui.handlers.single_progress import setup_progress_tracking
        setup_progress_tracking(
            ui_components, 
            tracker_name="dataset_downloader",
            progress_widget_key="progress_bar",
            progress_label_key="progress_message",
            total=100,
            description="Download dataset"
        )
    except ImportError:
        pass

def _setup_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observer handlers."""
    try:
        from smartcash.ui.handlers.observer_handler import setup_observer_handlers
        from smartcash.ui.dataset.handlers.download_progress_observer import setup_download_progress_observer
        
        # Setup basic observers
        ui_components = setup_observer_handlers(ui_components, "dataset_downloader_observers")
        
        # Setup download progress observer
        setup_download_progress_observer(ui_components)
    except ImportError:
        pass

def _setup_api_key_handler(ui_components: Dict[str, Any]) -> None:
    """Setup API key handler untuk Roboflow."""
    try:
        from smartcash.ui.dataset.handlers.api_key_handler import setup_api_key_input
        
        # Setup API key input dan validasi
        setup_api_key_input(ui_components)
    except ImportError:
        # Log gagal import jika logger tersedia
        logger = ui_components.get('logger')
        if logger:
            logger.debug("ℹ️ API key handler tidak tersedia")

def _setup_endpoint_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup handlers untuk endpoint dropdown."""
    # Import handler spesifik
    from smartcash.ui.dataset.handlers.endpoint_handler import handle_endpoint_change
    
    # Bind handler ke dropdown
    if 'endpoint_dropdown' in ui_components:
        ui_components['endpoint_dropdown'].observe(
            lambda change: handle_endpoint_change(change, ui_components),
            names='value'
        )
        
        # Inisialisasi tampilan awal
        handle_endpoint_change({'new': ui_components['endpoint_dropdown'].value}, ui_components)

def _setup_download_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol download."""
    # Import handler spesifik
    from smartcash.ui.dataset.handlers.download_handler import handle_download_button_click
    
    # Bind handler ke tombol
    if 'download_button' in ui_components:
        ui_components['download_button'].on_click(
            lambda b: handle_download_button_click(b, ui_components)
        )

def _setup_check_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol check status."""
    # Import handler spesifik
    from smartcash.ui.dataset.handlers.check_handler import handle_check_button_click
    
    # Bind handler ke tombol
    if 'check_button' in ui_components:
        ui_components['check_button'].on_click(
            lambda b: handle_check_button_click(b, ui_components)
        )

def _setup_cleanup(ui_components: Dict[str, Any]) -> None:
    """Setup cleanup function."""
    def cleanup_resources():
        """Fungsi untuk membersihkan resources."""
        try:
            # Reset progress
            if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                ui_components['progress_bar'].layout.visibility = 'hidden'
                ui_components['progress_message'].layout.visibility = 'hidden'
                ui_components['progress_bar'].value = 0
                ui_components['progress_message'].value = ""
            
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