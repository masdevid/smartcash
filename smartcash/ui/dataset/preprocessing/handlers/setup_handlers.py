"""
File: smartcash/ui/dataset/preprocessing/handlers/setup_handlers.py
Deskripsi: Setup handler untuk UI preprocessing dataset yang terintegrasi dengan observer pattern
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel
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
    _setup_confirmation_area(ui_components)
    
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
    
    # Log info dengan logger helper
    log_message(ui_components, "Preprocessing handlers berhasil diinisialisasi", "success", "✅")
    
    return ui_components

def _setup_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observer handlers untuk sistem notifikasi."""
    try:
        # Import sistem notifikasi 
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import register_ui_observers
        
        try:
            # Setup observer manager dan register UI observers
            from smartcash.common.observer import ObserverManager
            
            # Setup observer manager jika belum ada
            if 'observer_manager' not in ui_components:
                ui_components['observer_manager'] = ObserverManager()
            
            # Register UI observers untuk log dan progress
            register_ui_observers(ui_components)
            
            # Log setup berhasil
            log_message(ui_components, "Observer untuk sistem notifikasi berhasil disetup", "debug", "✅")
        except (ImportError, AttributeError) as e:
            # Log error jika ObserverManager tidak tersedia
            log_message(ui_components, f"Observer manager tidak tersedia: {str(e)}", "warning", "⚠️")
    except ImportError as e:
        # Log gagal import
        log_message(ui_components, f"Observer handler tidak tersedia: {str(e)}", "debug", "ℹ️")

def _setup_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Setup area konfirmasi untuk dialog konfirmasi."""
    if 'confirmation_area' not in ui_components:
        from IPython.display import display
        import ipywidgets as widgets
        
        # Buat output widget untuk area konfirmasi
        ui_components['confirmation_area'] = widgets.Output()
        
        # Tambahkan ke UI jika ui adalah VBox
        if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            try:
                # Cari posisi yang tepat (setelah tombol action atau progress container)
                children = list(ui_components['ui'].children)
                insert_pos = -1
                
                # Cari posisi setelah button container
                for i, child in enumerate(children):
                    if child == ui_components.get('button_container') or child == ui_components.get('progress_container'):
                        insert_pos = i + 1
                        break
                
                # Jika tidak ditemukan, tambahkan di akhir
                if insert_pos == -1:
                    insert_pos = len(children)
                
                # Sisipkan confirmation area
                children.insert(insert_pos, ui_components['confirmation_area'])
                ui_components['ui'].children = children
            except Exception as e:
                # Log error jika gagal menambahkan area konfirmasi
                log_message(ui_components, f"Gagal menambahkan area konfirmasi: {str(e)}", "warning", "⚠️")

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
            for label_key in ['overall_label', 'step_label', 'current_progress']:
                if label_key in ui_components and hasattr(ui_components[label_key], 'layout'):
                    ui_components[label_key].layout.visibility = 'hidden'
                    ui_components[label_key].value = ""
            
            # Unregister observer group jika ada
            if 'observer_manager' in ui_components and 'observer_group' in ui_components:
                try:
                    ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
                except Exception:
                    pass
            
            # Reset logging
            try:
                from smartcash.ui.utils.logging_utils import reset_logging
                reset_logging()
            except ImportError:
                pass
            
            # Log cleanup berhasil
            if 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message']("Cleanup preprocessing berhasil", "debug", "🧹")
        except Exception as e:
            # Ignore exceptions during cleanup
            pass
    
    # Tetapkan fungsi cleanup ke ui_components
    ui_components['cleanup'] = cleanup_resources
    
    # Register cleanup dengan IPython event
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.events.register('pre_run_cell', lambda: cleanup_resources())
    except (ImportError, AttributeError):
        # Skip jika tidak di IPython environment
        pass
