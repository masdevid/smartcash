"""
File: smartcash/ui/dataset/preprocessing/handlers/setup_handlers.py
Deskripsi: Setup handler untuk preprocessing dataset
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager

# Import utils dari modul preprocessing
from smartcash.ui.dataset.preprocessing.utils.notification_manager import (
    PREPROCESSING_LOGGER_NAMESPACE,
    MODULE_LOGGER_NAME,
    get_observer_manager
)
from smartcash.ui.dataset.preprocessing.utils.logger_helper import setup_ui_logger, log_message
from smartcash.ui.dataset.preprocessing.utils.ui_observers import register_ui_observers
from smartcash.ui.dataset.preprocessing.utils.progress_manager import setup_multi_progress, setup_progress_indicator
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel

def setup_preprocessing_handlers(ui_components: Dict[str, Any], env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Setup semua handler untuk preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        config: Konfigurasi aplikasi (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup logger
    ui_components = setup_ui_logger(ui_components)
    logger = ui_components.get('logger', get_logger(PREPROCESSING_LOGGER_NAMESPACE))
    
    # Log informasi setup
    log_message(ui_components, "Memulai setup preprocessing handlers", "info", "ℹ️")
    
    # Setup observer sistem
    _setup_observers(ui_components)
    
    # Load konfigurasi dan update UI
    try:
        from smartcash.ui.dataset.preprocessing.handlers.config_handler import setup_preprocessing_config_handler
        
        # Pastikan config valid
        if config and isinstance(config, dict):
            # Update UI dengan config yang diberikan
            ui_components = setup_preprocessing_config_handler(ui_components, config, env)
        else:
            # Gunakan config manager jika config tidak diberikan
            config_manager = get_config_manager()
            dataset_config = config_manager.get_module_config('dataset')
            
            if dataset_config and isinstance(dataset_config, dict):
                ui_components = setup_preprocessing_config_handler(ui_components, dataset_config, env)
    except Exception as e:
        # Log error
        log_message(ui_components, f"Gagal memuat konfigurasi: {str(e)}", "warning", "⚠️")
    
    # Setup button handlers
    _setup_button_handlers(ui_components)
    
    # Setup confirmation handler
    _setup_confirmation_handler(ui_components)
    
    # Setup multi-progress tracking
    _setup_progress_tracking(ui_components)
    
    # Setup state handler
    _setup_state_handler(ui_components)
    
    # Setup service handler
    _setup_service_handler(ui_components)
    
    # Setup status handler
    _setup_status_handler(ui_components)
    
    # Setup cleanup handler
    _setup_cleanup_handler(ui_components)
    
    # Setup persistence handler
    _setup_persistence_handler(ui_components)
    
    # Save config yang ada ke ui_components
    ui_components['config'] = config or {}
    
    # Log setup berhasil
    log_message(ui_components, "Semua handler preprocessing berhasil disetup", "success", "✅")
    
    return ui_components

def _setup_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observer handlers untuk sistem notifikasi."""
    try:
        # Coba setup observer manager
        try:
            # Get observer manager singleton
            observer_manager = get_observer_manager()
            ui_components['observer_manager'] = observer_manager
            
            # Register UI observers
            register_ui_observers(ui_components)
            
            # Log setup berhasil
            log_message(ui_components, "Observer untuk sistem notifikasi berhasil disetup", "debug", "✅")
        except (ImportError, AttributeError) as e:
            # Log error jika get_observer_manager tidak tersedia
            log_message(ui_components, f"Observer manager tidak tersedia: {str(e)}", "warning", "⚠️")
            
            # Gunakan observer lama jika ada
            from smartcash.components.observer import ObserverManager
            ui_components['observer_manager'] = ObserverManager()
            
            # Register UI observers
            register_ui_observers(ui_components)
            
            log_message(ui_components, "Menggunakan observer manager fallback", "info", "ℹ️")
    except ImportError as e:
        # Log gagal import
        log_message(ui_components, f"Observer handler tidak tersedia: {str(e)}", "debug", "ℹ️")

def _setup_button_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup handlers untuk semua button."""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.button_handler import setup_preprocessing_button_handlers
        
        # Setup button handlers
        ui_components = setup_preprocessing_button_handlers(ui_components, 'preprocessing', ui_components.get('config', {}), None)
        
        # Log setup berhasil
        log_message(ui_components, "Button handlers berhasil disetup", "debug", "✅")
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat setup button handlers: {str(e)}", "warning", "⚠️")

def _setup_confirmation_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk konfirmasi preprocessing."""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.confirmation_handler import setup_confirmation_handler
        
        # Setup confirmation handler
        ui_components = setup_confirmation_handler(ui_components)
        
        # Log setup berhasil
        log_message(ui_components, "Confirmation handler berhasil disetup", "debug", "✅")
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat setup confirmation handler: {str(e)}", "warning", "⚠️")

def _setup_progress_tracking(ui_components: Dict[str, Any]) -> None:
    """Setup progress tracking untuk preprocessing."""
    try:
        # Setup multi-progress tracking
        setup_multi_progress(ui_components)
        
        # Setup progress indicator
        setup_progress_indicator(ui_components)
        
        # Log setup berhasil
        log_message(ui_components, "Progress tracking berhasil disetup", "debug", "✅")
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat setup progress tracking: {str(e)}", "warning", "⚠️")

def _setup_state_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk state preprocessing."""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.state_handler import setup_state_handler, detect_preprocessing_state
        
        # Setup state handler
        ui_components = setup_state_handler(ui_components)
        
        # Deteksi state preprocessing
        ui_components = detect_preprocessing_state(ui_components)
        
        # Log setup berhasil
        log_message(ui_components, "State handler berhasil disetup", "debug", "✅")
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat setup state handler: {str(e)}", "warning", "⚠️")

def _setup_service_handler(ui_components: Dict[str, Any]) -> None:
    """Setup service handler untuk preprocessing."""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.preprocessing_service_handler import initialize_preprocessing
        
        # Initialize preprocessing service
        ui_components = initialize_preprocessing(ui_components)
        
        # Log setup berhasil
        log_message(ui_components, "Service handler berhasil disetup", "debug", "✅")
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat setup service handler: {str(e)}", "warning", "⚠️")

def _setup_status_handler(ui_components: Dict[str, Any]) -> None:
    """Setup status handler untuk preprocessing."""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.status_handler import setup_status_handler
        
        # Setup status handler
        ui_components = setup_status_handler(ui_components)
        
        # Set status awal
        update_status_panel(ui_components, "idle", "Preprocessing siap dijalankan")
        
        # Log setup berhasil
        log_message(ui_components, "Status handler berhasil disetup", "debug", "✅")
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat setup status handler: {str(e)}", "warning", "⚠️")

def _setup_cleanup_handler(ui_components: Dict[str, Any]) -> None:
    """Setup cleanup handler untuk preprocessing."""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.cleanup_handler import setup_cleanup_handler
        
        # Setup cleanup handler
        ui_components = setup_cleanup_handler(ui_components)
        
        # Log setup berhasil
        log_message(ui_components, "Cleanup handler berhasil disetup", "debug", "✅")
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat setup cleanup handler: {str(e)}", "warning", "⚠️")
        
    # Setup cleanup function untuk IPython cell
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
            
            # Reset logging
            try:
                from smartcash.ui.utils.logging_utils import reset_logging
                reset_logging()
            except ImportError:
                pass
            
            # Log cleanup
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

def _setup_persistence_handler(ui_components: Dict[str, Any]) -> None:
    """Setup persistence handler untuk preprocessing."""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.persistence_handler import setup_persistence_handler
        
        # Setup persistence handler
        ui_components = setup_persistence_handler(ui_components)
        
        # Log setup berhasil
        log_message(ui_components, "Persistence handler berhasil disetup", "debug", "✅")
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat setup persistence handler: {str(e)}", "warning", "⚠️")
