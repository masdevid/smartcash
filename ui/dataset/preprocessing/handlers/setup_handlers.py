"""
File: smartcash/ui/dataset/preprocessing/handlers/setup_handlers.py
Deskripsi: Setup handler untuk UI preprocessing dataset yang terintegrasi dengan observer pattern
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel, ensure_confirmation_area
from smartcash.ui.dataset.preprocessing.utils.progress_manager import setup_multi_progress, setup_progress_indicator

def setup_preprocessing_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Setup observer untuk menerima event notifikasi
    _setup_observers(ui_components)
    
    # Pastikan area konfirmasi tersedia
    ui_components = ensure_confirmation_area(ui_components)
    
    # Load konfigurasi dan update UI
    try:
        from smartcash.ui.dataset.preprocessing.handlers.config_handler import update_ui_from_config
        config_manager = get_config_manager()
        
        # Get dataset config
        dataset_config = config_manager.get_module_config('dataset')
        
        # Pastikan config valid
        if dataset_config and isinstance(dataset_config, dict):
            # Update UI dengan config yang valid
            update_ui_from_config(ui_components, dataset_config)
        else:
            # Gunakan config yang diberikan jika config manager tidak valid
            if config and isinstance(config, dict):
                update_ui_from_config(ui_components, config)
    except Exception as e:
        # Log error dengan logger helper
        log_message(ui_components, f"Gagal memuat konfigurasi: {str(e)}", "warning", "⚠️")
    
    # Setup multi-progress tracking untuk preprocessing
    _setup_progress_tracking(ui_components)
    
    # Setup handlers untuk UI events
    _setup_preprocessing_button_handler(ui_components)
    _setup_stop_button_handler(ui_components)
    _setup_reset_button_handler(ui_components)
    _setup_save_button_handler(ui_components)
    _setup_cleanup_button_handler(ui_components)
    
    # Setup cleanup function
    _setup_cleanup(ui_components)
    
    # Tambahkan flag untuk tracking status jika belum ada
    if 'preprocessing_running' not in ui_components:
        ui_components['preprocessing_running'] = False
    if 'cleanup_running' not in ui_components:
        ui_components['cleanup_running'] = False
    if 'stop_requested' not in ui_components:
        ui_components['stop_requested'] = False
    
    # Save config yang sudah ada ke UI components
    ui_components['config'] = config or {}
    
    # Tambahkan direktori default jika belum ada
    if 'data_dir' not in ui_components:
        ui_components['data_dir'] = 'data'
    if 'preprocessed_dir' not in ui_components:
        ui_components['preprocessed_dir'] = 'data/preprocessed'
    
    # Log info dengan logger helper
    log_message(ui_components, "Preprocessing handlers berhasil diinisialisasi", "success", "✅")
    
    return ui_components

def _setup_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observer handlers untuk sistem notifikasi."""
    try:
        # Import sistem notifikasi 
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import register_ui_observers, MockObserverManager
        
        try:
            # Coba import observer dari modul yang berbeda-beda
            try:
                from smartcash.components.observer import ObserverManager
                observer_class = ObserverManager
            except ImportError:
                try:
                    from smartcash.common.observer import ObserverManager
                    observer_class = ObserverManager
                except ImportError:
                    # Gunakan mock observer jika tidak ada modul observer
                    observer_class = MockObserverManager
                    log_message(ui_components, "Menggunakan mock observer karena modul observer tidak tersedia", "info", "ℹ️")
            
            # Setup observer manager jika belum ada
            if 'observer_manager' not in ui_components:
                ui_components['observer_manager'] = observer_class()
            
            # Register UI observers untuk log dan progress
            register_ui_observers(ui_components)
            
            # Log setup berhasil
            log_message(ui_components, "Observer untuk sistem notifikasi berhasil disetup", "debug", "✅")
        except Exception as e:
            # Log error jika gagal setup observer
            log_message(ui_components, f"Error saat setup observer: {str(e)}", "warning", "⚠️")
            
            # Gunakan mock observer sebagai fallback
            ui_components['observer_manager'] = MockObserverManager()
            log_message(ui_components, "Menggunakan mock observer sebagai fallback", "info", "ℹ️")
    except ImportError as e:
        # Log gagal import
        log_message(ui_components, f"Observer handler tidak tersedia: {str(e)}", "debug", "ℹ️")

def _setup_progress_tracking(ui_components: Dict[str, Any]) -> None:
    """Setup progress tracking untuk preprocessing."""
    try:
        # Gunakan modul progress_manager untuk setup
        setup_multi_progress(ui_components)
        
        # Setup progress indicator jika diperlukan
        setup_progress_indicator(ui_components)
        
        # Log setup berhasil
        log_message(ui_components, "Progress tracking berhasil disetup", "debug", "✅")
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat setup progress tracking: {str(e)}", "warning", "⚠️")

def _setup_preprocessing_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol preprocessing."""
    from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handler import handle_preprocessing_button_click
    
    if 'preprocess_button' in ui_components:
        ui_components['preprocess_button'].on_click(
            lambda b: handle_preprocessing_button_click(b, ui_components)
        )

def _setup_stop_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol stop."""
    from smartcash.ui.dataset.preprocessing.handlers.stop_handler import handle_stop_button_click
    
    if 'stop_button' in ui_components:
        ui_components['stop_button'].on_click(
            lambda b: handle_stop_button_click(b, ui_components)
        )

def _setup_reset_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol reset."""
    from smartcash.ui.dataset.preprocessing.handlers.reset_handler import handle_reset_button_click
    
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(
            lambda b: handle_reset_button_click(b, ui_components)
        )

def _setup_cleanup_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol cleanup."""
    from smartcash.ui.dataset.preprocessing.handlers.cleanup_handler import handle_cleanup_button_click
    
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].on_click(
            lambda b: handle_cleanup_button_click(b, ui_components)
        )

def _setup_save_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol save."""
    from smartcash.ui.dataset.preprocessing.handlers.save_handler import handle_save_button_click
    
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(
            lambda b: handle_save_button_click(b, ui_components)
        )

def _setup_cleanup(ui_components: Dict[str, Any]) -> None:
    """Setup fungsi cleanup untuk membersihkan resources saat selesai."""
    if 'cleanup_ui' not in ui_components:
        def cleanup_resources():
            # Reset semua flag
            ui_components['preprocessing_running'] = False
            ui_components['cleanup_running'] = False
            ui_components['stop_requested'] = False
            
            # Bersihkan observer manager
            if 'observer_manager' in ui_components and hasattr(ui_components['observer_manager'], 'cleanup'):
                ui_components['observer_manager'].cleanup()
        
        ui_components['cleanup_ui'] = cleanup_resources
