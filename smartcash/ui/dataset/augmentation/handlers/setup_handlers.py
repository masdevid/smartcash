"""
File: smartcash/ui/dataset/augmentation/handlers/setup_handlers.py
Deskripsi: Setup handler untuk UI augmentasi dataset
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import update_status_panel
from smartcash.ui.dataset.augmentation.utils.progress_manager import setup_multi_progress, setup_progress_indicator

def setup_augmentation_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Setup observer untuk notifikasi
    _setup_observers(ui_components)
    
    # Load konfigurasi dan update UI
    try:
        from smartcash.ui.dataset.augmentation.handlers.config_handler import update_ui_from_config
        config_manager = get_config_manager()
        
        # Get augmentation config
        augmentation_config = config_manager.get_module_config('augmentation')
        
        if augmentation_config and isinstance(augmentation_config, dict):
            update_ui_from_config(ui_components, augmentation_config)
        elif config and isinstance(config, dict):
            update_ui_from_config(ui_components, config)
    except Exception as e:
        log_message(ui_components, f"Gagal memuat konfigurasi: {str(e)}", "warning", "⚠️")
    
    # Setup handlers untuk UI events
    _setup_button_handlers(ui_components)
    
    # Setup progress tracking
    _setup_progress_tracking(ui_components)
    
    # Setup cleanup function
    _setup_cleanup(ui_components)
    
    # Save config ke UI components
    ui_components['config'] = config or {}
    
    log_message(ui_components, "Augmentasi dataset handlers berhasil diinisialisasi", "success", "✅")
    
    return ui_components

def _setup_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observer handlers untuk sistem notifikasi."""
    try:
        from smartcash.ui.dataset.augmentation.utils.ui_observers import register_ui_observers
        
        try:
            from smartcash.ui.dataset.augmentation.utils.notification_manager import get_observer_manager
            observer_manager = get_observer_manager()
            ui_components['observer_manager'] = observer_manager
            register_ui_observers(ui_components)
            log_message(ui_components, "Observer sistem notifikasi berhasil disetup", "debug", "✅")
        except (ImportError, AttributeError) as e:
            log_message(ui_components, f"Observer manager tidak tersedia: {str(e)}", "warning", "⚠️")
            from smartcash.components.observer import ObserverManager
            ui_components['observer_manager'] = ObserverManager()
            register_ui_observers(ui_components)
            log_message(ui_components, "Menggunakan observer manager fallback", "info", "ℹ️")
    except ImportError as e:
        log_message(ui_components, f"Observer handler tidak tersedia: {str(e)}", "debug", "ℹ️")

def _setup_button_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup handlers untuk tombol UI."""
    from smartcash.ui.dataset.augmentation.handlers.augmentation_handler import handle_augmentation_button_click
    from smartcash.ui.dataset.augmentation.handlers.cleanup_handler import handle_cleanup_button_click
    from smartcash.ui.dataset.augmentation.handlers.reset_handler import handle_reset_button_click
    from smartcash.ui.dataset.augmentation.handlers.save_handler import handle_save_button_click
    
    # Setup augmentation button
    if 'augment_button' in ui_components:
        ui_components['augment_button'].on_click(
            lambda b: handle_augmentation_button_click(ui_components, b)
        )
    
    # Setup cleanup button
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].on_click(
            lambda b: handle_cleanup_button_click(ui_components, b)
        )
    
    # Setup reset button
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(
            lambda b: handle_reset_button_click(ui_components, b)
        )
    
    # Setup save button
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(
            lambda b: handle_save_button_click(ui_components, b)
        )

def _setup_progress_tracking(ui_components: Dict[str, Any]) -> None:
    """Setup progress tracking untuk augmentasi."""
    try:
        setup_multi_progress(ui_components)
        setup_progress_indicator(ui_components)
        log_message(ui_components, "Progress tracking berhasil disetup", "debug", "✅")
    except Exception as e:
        log_message(ui_components, f"Error saat setup progress tracking: {str(e)}", "warning", "⚠️")

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
            
            # Unregister observer group
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
            
            # Log cleanup
            if 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message']("Cleanup augmentasi berhasil", "debug", "🧹")
        except Exception:
            pass
    
    ui_components['cleanup'] = cleanup_resources
    
    # Register cleanup dengan IPython event
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.events.register('pre_run_cell', lambda: cleanup_resources())
    except (ImportError, AttributeError):
        pass