"""
File: smartcash/ui/dataset/augmentation/handlers/setup_handlers.py
Deskripsi: Setup handler untuk UI augmentasi dataset dengan logger bridge
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

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
    ui_logger = create_ui_logger_bridge(ui_components, "setup_handlers")
    
    # Setup observer untuk notifikasi
    _setup_observers(ui_components, ui_logger)
    
    # Load konfigurasi dan update UI
    try:
        from smartcash.ui.dataset.augmentation.handlers.config_handler import load_augmentation_config
        config_manager = get_config_manager()
        
        # Get augmentation config
        augmentation_config = config_manager.get_module_config('augmentation')
        
        if augmentation_config and isinstance(augmentation_config, dict):
            _update_ui_from_config(ui_components, augmentation_config, ui_logger)
        elif config and isinstance(config, dict):
            _update_ui_from_config(ui_components, config, ui_logger)
    except Exception as e:
        ui_logger.warning(f"Gagal memuat konfigurasi: {str(e)}")
    
    # Setup handlers untuk UI events
    _setup_button_handlers(ui_components)
    
    # Setup progress tracking
    _setup_progress_tracking(ui_components, ui_logger)
    
    # Setup cleanup function
    _setup_cleanup(ui_components, ui_logger)
    
    # Save config ke UI components
    ui_components['config'] = config or {}
    
    ui_logger.success("✅ Augmentasi dataset handlers berhasil diinisialisasi")
    
    return ui_components

def _setup_observers(ui_components: Dict[str, Any], ui_logger) -> None:
    """Setup observer handlers untuk sistem notifikasi."""
    try:
        from smartcash.ui.dataset.augmentation.utils.ui_observers import register_ui_observers
        
        try:
            from smartcash.ui.dataset.augmentation.utils.notification_manager import get_observer_manager
            observer_manager = get_observer_manager()
            ui_components['observer_manager'] = observer_manager
            register_ui_observers(ui_components)
            ui_logger.debug("✅ Observer sistem notifikasi berhasil disetup")
        except (ImportError, AttributeError) as e:
            ui_logger.warning(f"Observer manager tidak tersedia: {str(e)}")
            from smartcash.components.observer import ObserverManager
            ui_components['observer_manager'] = ObserverManager()
            register_ui_observers(ui_components)
            ui_logger.info("ℹ️ Menggunakan observer manager fallback")
    except ImportError as e:
        ui_logger.debug(f"Observer handler tidak tersedia: {str(e)}")

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

def _setup_progress_tracking(ui_components: Dict[str, Any], ui_logger) -> None:
    """Setup progress tracking untuk augmentasi."""
    try:
        from smartcash.ui.dataset.augmentation.utils.progress_manager import setup_multi_progress, setup_progress_indicator
        setup_multi_progress(ui_components)
        setup_progress_indicator(ui_components)
        ui_logger.debug("✅ Progress tracking berhasil disetup")
    except Exception as e:
        ui_logger.warning(f"Error saat setup progress tracking: {str(e)}")

def _setup_cleanup(ui_components: Dict[str, Any], ui_logger) -> None:
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
                from smartcash.ui.utils.logging_utils import restore_console_logs
                restore_console_logs(ui_components)
            except ImportError:
                pass
            
            ui_logger.debug("🧹 Cleanup augmentasi berhasil")
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

def _update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any], ui_logger) -> None:
    """Update UI components dari konfigurasi."""
    try:
        from smartcash.ui.dataset.augmentation.handlers.config_handler import _update_ui_from_config_values
        _update_ui_from_config_values(ui_components, config, ui_logger)
        ui_logger.debug("🔄 UI berhasil diupdate dari konfigurasi")
    except Exception as e:
        ui_logger.warning(f"Gagal update UI dari config: {str(e)}")