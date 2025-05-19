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
    
    # Load konfigurasi dan update UI
    try:
        from smartcash.ui.dataset.download.handlers.config_handler import load_config, update_ui_from_config
        loaded_config = load_config()
        update_ui_from_config(loaded_config, ui_components)
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Gagal memuat konfigurasi: {str(e)}")
    
    # Setup handlers untuk UI events
    _setup_endpoint_handlers(ui_components)
    _setup_download_button_handler(ui_components)
    _setup_check_button_handler(ui_components)
    _setup_reset_button_handler(ui_components)
    _setup_save_button_handler(ui_components)
    
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
    """Setup observer handlers untuk sistem notifikasi baru."""
    try:
        # Import sistem notifikasi baru
        from smartcash.ui.dataset.download.utils.ui_observers import register_ui_observers
        from smartcash.ui.dataset.download.utils.notification_manager import get_observer_manager
        
        # Setup observer manager dan register UI observers
        observer_manager = get_observer_manager()
        ui_components['observer_manager'] = observer_manager
        
        # Register UI observers untuk log dan progress
        register_ui_observers(ui_components)
        
        # Log setup berhasil
        logger = ui_components.get('logger')
        if logger:
            logger.debug("✅ Observer untuk sistem notifikasi berhasil disetup")
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

def _setup_reset_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol reset."""
    from smartcash.ui.dataset.download.handlers.reset_handler import handle_reset_button_click
    
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(
            lambda b: handle_reset_button_click(b, ui_components)
        )

def _setup_save_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol save."""
    # Gunakan config_handler untuk menyimpan konfigurasi
    from smartcash.ui.dataset.download.handlers.config_handler import update_config_from_ui, save_config_with_manager, load_config
    
    def handle_save_button_click(b, ui_components):
        # Reset log output saat tombol diklik
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        
        logger = ui_components.get('logger')
        
        try:
            # Simpan konfigurasi dengan config_handler
            config = load_config()
            config = update_config_from_ui(config, ui_components)
            success = save_config_with_manager(config, ui_components, logger)
            
            # Update status panel jika tersedia
            from smartcash.ui.utils.ui_logger import log_to_ui
            if success:
                success_msg = "Konfigurasi berhasil disimpan"
                if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
                    ui_components['update_status_panel'](ui_components, "success", success_msg)
                else:
                    log_to_ui(ui_components, success_msg, "success", "💾")
            else:
                error_msg = "Gagal menyimpan konfigurasi"
                log_to_ui(ui_components, error_msg, "error", "❌")
        except Exception as e:
            # Tampilkan error
            from smartcash.ui.utils.ui_logger import log_to_ui
            error_msg = f"Error saat menyimpan konfigurasi: {str(e)}"
            log_to_ui(ui_components, error_msg, "error", "❌")
            if logger: logger.error(f"❌ {error_msg}")
    
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(
            lambda b: handle_save_button_click(b, ui_components)
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
            if 'api_key' in ui_components:
                ui_components['api_key'].layout.border = ""
            
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
