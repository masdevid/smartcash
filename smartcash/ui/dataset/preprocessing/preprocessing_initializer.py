"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Initializer utama untuk UI preprocessing dataset dengan arsitektur handler yang modular dan SRP
"""

from typing import Dict, Any, Optional
from pathlib import Path
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.common.config.manager import get_config_manager

# Konstanta untuk backward compatibility
PREPROCESSING_LOGGER_NAMESPACE = "smartcash.ui.dataset.preprocessing"
MODULE_LOGGER_NAME = "PREPROC"

# Flag global untuk mencegah inisialisasi ulang
_PREPROCESSING_MODULE_INITIALIZED = False

def initialize_dataset_preprocessing_ui(env=None, config=None) -> Any:
    """
    Inisialisasi UI untuk dataset preprocessing dengan arsitektur handler modular.
    
    Args:
        env: Environment manager (opsional)
        config: Konfigurasi aplikasi (opsional)
        
    Returns:
        Widget UI utama yang bisa ditampilkan
    """
    global _PREPROCESSING_MODULE_INITIALIZED
    
    # Setup logger dasar
    logger = get_logger(PREPROCESSING_LOGGER_NAMESPACE)
    
    # Cek inisialisasi ulang
    if _PREPROCESSING_MODULE_INITIALIZED:
        logger.debug(f"[{MODULE_LOGGER_NAME}] UI preprocessing sudah diinisialisasi, menggunakan yang ada")
    else:
        logger.info(f"[{MODULE_LOGGER_NAME}] 🚀 Memulai inisialisasi UI preprocessing dataset")
        _PREPROCESSING_MODULE_INITIALIZED = True
    
    try:
        # Phase 1: Environment & Config Setup
        ui_components = _initialize_core_services(env, config, logger)
        
        # Phase 2: UI Components Creation
        ui_components = _create_ui_components(ui_components, logger)
        
        # Phase 3: Logger Handler Setup
        ui_components = _setup_logger_handler(ui_components, logger)
        
        # Phase 4: Handlers Initialization
        ui_components = _initialize_handlers(ui_components, env, config, logger)
        
        # Phase 5: Final Setup & Validation
        ui_components = _finalize_setup(ui_components, logger)
        
        # Success log
        logger.success(f"[{MODULE_LOGGER_NAME}] ✅ UI preprocessing dataset berhasil diinisialisasi")
        
        # Return main UI widget
        return ui_components.get('ui')
        
    except Exception as e:
        logger.error(f"[{MODULE_LOGGER_NAME}] ❌ Error saat inisialisasi: {str(e)}")
        return _create_error_ui(str(e), logger)

def _initialize_core_services(env, config, logger) -> Dict[str, Any]:
    """Inisialisasi core services (environment, config)."""
    ui_components = {}
    
    try:
        # Environment Manager
        if env:
            environment_manager = env
        else:
            environment_manager = get_environment_manager()
        
        ui_components['environment_manager'] = environment_manager
        
        # Config Manager
        config_manager = get_config_manager()
        ui_components['config_manager'] = config_manager
        
        # Load dataset config
        dataset_config = config_manager.get_config().get('preprocessing', {})
        if config and isinstance(config, dict):
            dataset_config.update(config)
        
        ui_components['config'] = dataset_config
        
        # Basic directories
        ui_components['data_dir'] = dataset_config.get('data', {}).get('dir', 'data')
        ui_components['preprocessed_dir'] = dataset_config.get('output_dir', 'data/preprocessed')
        
        logger.info(f"[{MODULE_LOGGER_NAME}] ⚙️ Core services berhasil diinisialisasi")
        
    except Exception as e:
        logger.warning(f"[{MODULE_LOGGER_NAME}] ⚠️ Error init core services: {str(e)}")
        # Fallback minimal setup
        ui_components.update({
            'config': {},
            'data_dir': 'data',
            'preprocessed_dir': 'data/preprocessed'
        })
    
    return ui_components

def _create_ui_components(ui_components: Dict[str, Any], logger) -> Dict[str, Any]:
    """Buat komponen UI menggunakan components yang sudah direfactor."""
    try:
        from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_ui_components
        
        # Buat UI components dengan config yang ada
        config = ui_components.get('config', {})
        ui_components.update(create_preprocessing_ui_components(config))
        
        logger.info(f"[{MODULE_LOGGER_NAME}] 🎨 UI components berhasil dibuat")
        
    except Exception as e:
        logger.error(f"[{MODULE_LOGGER_NAME}] ❌ Error create UI components: {str(e)}")
        # Fallback ke UI error
        ui_components['ui'] = _create_minimal_error_ui(str(e))
    
    return ui_components

def _setup_logger_handler(ui_components: Dict[str, Any], logger) -> Dict[str, Any]:
    """Setup logger handler untuk UI logging."""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.logger_handler import setup_preprocessing_logger
        
        # Setup logger handler
        logger_handler = setup_preprocessing_logger(ui_components)
        
        # Log dengan logger handler yang baru
        logger_handler.info("Logger handler berhasil disetup")
        
    except Exception as e:
        logger.warning(f"[{MODULE_LOGGER_NAME}] ⚠️ Error setup logger handler: {str(e)}")
        # Fallback ke logger basic
        ui_components['logger'] = logger
        ui_components['log_message'] = lambda msg, level='info', icon='': logger.info(f"{icon} {msg}")
    
    return ui_components

def _initialize_handlers(ui_components: Dict[str, Any], env, config, logger) -> Dict[str, Any]:
    """Inisialisasi semua handlers menggunakan setup handler."""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.setup_handler import initialize_preprocessing_handlers
        
        # Initialize semua handlers
        ui_components = initialize_preprocessing_handlers(ui_components, env, config)
        
        logger.info(f"[{MODULE_LOGGER_NAME}] 🔧 Handlers berhasil diinisialisasi")
        
    except Exception as e:
        logger.error(f"[{MODULE_LOGGER_NAME}] ❌ Error initialize handlers: {str(e)}")
        # Setup fallback handlers
        ui_components = _setup_fallback_handlers(ui_components, logger)
    
    return ui_components

def _setup_fallback_handlers(ui_components: Dict[str, Any], logger) -> Dict[str, Any]:
    """Setup handlers fallback jika handler utama gagal."""
    try:
        # Setup basic button handlers
        if 'preprocess_button' in ui_components:
            ui_components['preprocess_button'].on_click(
                lambda b: _fallback_preprocessing_handler(b, ui_components, logger)
            )
        
        # Setup basic cleanup handler
        if 'cleanup_button' in ui_components:
            ui_components['cleanup_button'].on_click(
                lambda b: _fallback_cleanup_handler(b, ui_components, logger)
            )
        
        # Setup basic config handlers
        if 'save_button' in ui_components:
            ui_components['save_button'].on_click(
                lambda b: _fallback_save_handler(b, ui_components, logger)
            )
        
        if 'reset_button' in ui_components:
            ui_components['reset_button'].on_click(
                lambda b: _fallback_reset_handler(b, ui_components, logger)
            )
        
        logger.info(f"[{MODULE_LOGGER_NAME}] 🔄 Fallback handlers berhasil disetup")
        
    except Exception as e:
        logger.warning(f"[{MODULE_LOGGER_NAME}] ⚠️ Error setup fallback handlers: {str(e)}")
    
    return ui_components

def _finalize_setup(ui_components: Dict[str, Any], logger) -> Dict[str, Any]:
    """Finalisasi setup dan validasi."""
    try:
        # Set initialization flags
        ui_components['preprocessing_initialized'] = True
        ui_components['logger_namespace'] = PREPROCESSING_LOGGER_NAMESPACE
        ui_components['module_name'] = 'preprocessing'
        
        # Set runtime flags
        ui_components['preprocessing_running'] = False
        ui_components['cleanup_running'] = False
        ui_components['stop_requested'] = False
        
        # Pastikan tombol stop tersembunyi di awal
        if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
            ui_components['stop_button'].layout.display = 'none'
        
        # Setup cleanup function
        ui_components['cleanup_ui'] = lambda: _cleanup_preprocessing_resources(ui_components, logger)
        
        logger.info(f"[{MODULE_LOGGER_NAME}] 🎯 Setup berhasil diselesaikan")
        
    except Exception as e:
        logger.warning(f"[{MODULE_LOGGER_NAME}] ⚠️ Error finalize setup: {str(e)}")
    
    return ui_components

def _cleanup_preprocessing_resources(ui_components: Dict[str, Any], logger) -> None:
    """Cleanup resources saat preprocessing selesai atau dibatalkan."""
    try:
        # Reset flags
        ui_components['preprocessing_running'] = False
        ui_components['cleanup_running'] = False  
        ui_components['stop_requested'] = False
        
        # Cleanup handlers
        for handler_key in ['setup_handler', 'config_handler', 'cleanup_handler', 'logger_handler']:
            if handler_key in ui_components:
                handler = ui_components[handler_key]
                if hasattr(handler, 'cleanup_resources'):
                    handler.cleanup_resources()
        
        logger.debug(f"[{MODULE_LOGGER_NAME}] 🧹 Resources berhasil dibersihkan")
        
    except Exception as e:
        logger.warning(f"[{MODULE_LOGGER_NAME}] ⚠️ Error cleanup resources: {str(e)}")

# Fallback handlers untuk kasus error
def _fallback_preprocessing_handler(button, ui_components: Dict[str, Any], logger) -> None:
    """Fallback handler untuk preprocessing button."""
    button.disabled = True
    try:
        logger.info(f"[{MODULE_LOGGER_NAME}] 🔄 Menggunakan fallback preprocessing handler")
        
        # Update status
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = "<div class='alert alert-warning'>⚠️ Menggunakan mode fallback</div>"
        
        # Log message
        log_func = ui_components.get('log_message', lambda msg, level, icon: logger.info(f"{icon} {msg}"))
        log_func("Preprocessing dalam mode fallback - fitur terbatas", "warning", "⚠️")
        
    except Exception as e:
        logger.error(f"[{MODULE_LOGGER_NAME}] ❌ Error fallback preprocessing: {str(e)}")
    finally:
        button.disabled = False

def _fallback_cleanup_handler(button, ui_components: Dict[str, Any], logger) -> None:
    """Fallback handler untuk cleanup button."""
    button.disabled = True
    try:
        logger.info(f"[{MODULE_LOGGER_NAME}] 🧹 Menggunakan fallback cleanup handler")
        
        log_func = ui_components.get('log_message', lambda msg, level, icon: logger.info(f"{icon} {msg}"))
        log_func("Cleanup dalam mode fallback - gunakan manual cleanup", "warning", "⚠️")
        
    except Exception as e:
        logger.error(f"[{MODULE_LOGGER_NAME}] ❌ Error fallback cleanup: {str(e)}")
    finally:
        button.disabled = False

def _fallback_save_handler(button, ui_components: Dict[str, Any], logger) -> None:
    """Fallback handler untuk save button."""
    button.disabled = True
    try:
        logger.info(f"[{MODULE_LOGGER_NAME}] 💾 Menggunakan fallback save handler")
        
        log_func = ui_components.get('log_message', lambda msg, level, icon: logger.info(f"{icon} {msg}"))
        log_func("Save config dalam mode fallback", "info", "💾")
        
    except Exception as e:
        logger.error(f"[{MODULE_LOGGER_NAME}] ❌ Error fallback save: {str(e)}")
    finally:
        button.disabled = False

def _fallback_reset_handler(button, ui_components: Dict[str, Any], logger) -> None:
    """Fallback handler untuk reset button."""
    button.disabled = True
    try:
        logger.info(f"[{MODULE_LOGGER_NAME}] 🔄 Menggunakan fallback reset handler")
        
        log_func = ui_components.get('log_message', lambda msg, level, icon: logger.info(f"{icon} {msg}"))
        log_func("Reset config dalam mode fallback", "info", "🔄")
        
    except Exception as e:
        logger.error(f"[{MODULE_LOGGER_NAME}] ❌ Error fallback reset: {str(e)}")
    finally:
        button.disabled = False

def _create_error_ui(error_message: str, logger) -> Any:
    """Buat UI error lengkap jika inisialisasi gagal."""
    try:
        import ipywidgets as widgets
        from smartcash.ui.utils.constants import COLORS, ICONS
        
        error_ui = widgets.VBox([
            widgets.HTML(f"""
            <div style="padding: 25px; background-color: #f8d7da; 
                       border: 1px solid #f5c6cb; border-radius: 8px; margin: 10px 0;">
                <h2 style="color: #721c24; margin-top: 0; display: flex; align-items: center;">
                    <span style="font-size: 1.5em; margin-right: 10px;">❌</span>
                    Error Inisialisasi Preprocessing
                </h2>
                <div style="background-color: #fff; padding: 15px; border-radius: 5px; 
                           margin: 15px 0; border-left: 4px solid #dc3545;">
                    <h4 style="margin-top: 0; color: #721c24;">Detail Error:</h4>
                    <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 3px; 
                              overflow-x: auto; color: #495057; font-size: 0.9em;">{error_message}</pre>
                </div>
                <div style="background-color: #d1ecf1; padding: 15px; border-radius: 5px; 
                           border-left: 4px solid #bee5eb;">
                    <h4 style="margin-top: 0; color: #0c5460;">🔧 Solusi:</h4>
                    <ul style="margin-bottom: 0; color: #0c5460;">
                        <li>Restart kernel Colab dan jalankan ulang cell</li>
                        <li>Pastikan semua dependencies terinstall dengan benar</li>
                        <li>Periksa koneksi internet untuk download dependencies</li>
                        <li>Hubungi administrator jika masalah berlanjut</li>
                    </ul>
                </div>
                <div style="margin-top: 20px; padding: 10px; background-color: #fff3cd; 
                           border-radius: 5px; border-left: 4px solid #ffeaa7;">
                    <p style="margin: 0; color: #856404;">
                        <strong>💡 Tips:</strong> Coba gunakan menu Runtime → Restart and run all untuk reset lengkap.
                    </p>
                </div>
            </div>
            """)
        ])
        
        return error_ui
        
    except Exception as e:
        logger.error(f"[{MODULE_LOGGER_NAME}] ❌ Error saat membuat error UI: {str(e)}")
        return _create_minimal_error_ui(f"Multiple errors: {error_message} | UI creation: {str(e)}")

def _create_minimal_error_ui(error_message: str) -> Any:
    """Buat UI error minimal sebagai fallback terakhir."""
    try:
        import ipywidgets as widgets
        
        return widgets.HTML(f"""
        <div style="padding: 20px; background-color: #f8d7da; border: 1px solid #f5c6cb; 
                   border-radius: 5px; color: #721c24;">
            <h3>❌ Error Preprocessing Initialization</h3>
            <p><strong>Error:</strong> {error_message}</p>
            <p><strong>Action:</strong> Please restart the kernel and run the cell again.</p>
        </div>
        """)
        
    except Exception:
        # Absolute fallback - return None, akan di-handle oleh caller
        return None

# Alias untuk backward compatibility
initialize_preprocessing_ui = initialize_dataset_preprocessing_ui

# Export untuk compatibility dengan old imports
__all__ = [
    'initialize_dataset_preprocessing_ui',
    'initialize_preprocessing_ui', 
    'PREPROCESSING_LOGGER_NAMESPACE',
    'MODULE_LOGGER_NAME'
]