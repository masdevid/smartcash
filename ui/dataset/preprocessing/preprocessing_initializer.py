"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Updated initializer menggunakan refactored SRP handlers
"""

import logging
from typing import Dict, Any, Optional
from smartcash.common.config.manager import get_config_manager
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.utils.ui_logger_namespace import PREPROCESSING_LOGGER_NAMESPACE

# Import components
from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui

# Import refactored SRP handlers
from smartcash.ui.dataset.preprocessing.handlers.config_handler import setup_config_handlers
from smartcash.ui.dataset.preprocessing.handlers.dataset_checker import setup_dataset_checker
from smartcash.ui.dataset.preprocessing.handlers.cleanup_executor import setup_cleanup_executor  
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_executor import setup_preprocessing_executor

# Global state management untuk caching
_MODULE_INITIALIZED = False
_CACHED_UI_COMPONENTS = None

def initialize_dataset_preprocessing_ui(env=None, config=None, force_refresh=False):
    """
    Main initializer untuk preprocessing UI dengan refactored SRP handlers.
    
    Args:
        env: Environment manager (optional)
        config: Konfigurasi preprocessing (optional)
        force_refresh: Force refresh UI components
        
    Returns:
        Widget UI preprocessing atau error fallback UI
    """
    global _MODULE_INITIALIZED, _CACHED_UI_COMPONENTS
    
    # Return cached UI jika sudah initialized
    if _MODULE_INITIALIZED and _CACHED_UI_COMPONENTS and not force_refresh:
        return _get_cached_ui_or_refresh(config)
    
    try:
        # 1. Setup comprehensive log suppression
        _setup_comprehensive_log_suppression()
        
        # 2. Get dan merge config
        merged_config = _get_merged_config(config)
        
        # 3. Create UI components
        ui_components = _create_ui_components_safe(merged_config)
        
        # 4. Setup logger bridge
        logger_bridge = _setup_logger_bridge_safe(ui_components)
        ui_components['logger'] = logger_bridge
        ui_components['logger_namespace'] = PREPROCESSING_LOGGER_NAMESPACE
        
        # 5. Setup handlers dengan SRP approach
        ui_components = _setup_refactored_handlers(ui_components, merged_config, env)
        
        # 6. Validation dan final setup
        validation_result = _validate_and_finalize_setup(ui_components)
        
        if not validation_result['valid']:
            return _create_error_fallback_ui(f"Validation failed: {validation_result['message']}")
        
        # 7. Cache dan return
        _CACHED_UI_COMPONENTS = ui_components
        _MODULE_INITIALIZED = True
        
        logger_bridge.success("✅ Preprocessing UI berhasil diinisialisasi dengan SRP handlers")
        return ui_components['ui']
        
    except Exception as e:
        return _create_error_fallback_ui(f"Initialization error: {str(e)}")

def _setup_refactored_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handlers dengan SRP approach."""
    handlers_setup = [
        ("Config Environment", lambda: setup_config_handlers(ui_components, config)),
        ("Dataset Checker", lambda: setup_dataset_checker(ui_components)),
        ("Cleanup Executor", lambda: setup_cleanup_executor(ui_components)),
        ("Preprocessing Executor", lambda: setup_preprocessing_executor(ui_components, env))
    ]
    
    failed_handlers = []
    
    for handler_name, setup_func in handlers_setup:
        try:
            ui_components = setup_func()
        except Exception as e:
            failed_handlers.append(f"{handler_name}: {str(e)}")
    
    if failed_handlers:
        ui_components['setup_warnings'] = failed_handlers
    
    return ui_components

def _setup_comprehensive_log_suppression():
    """Setup comprehensive log suppression untuk prevent verbose backend logs."""
    loggers_to_suppress = [
        'smartcash.dataset.services', 'smartcash.dataset.utils', 'smartcash.common.threadpools',
        'concurrent.futures', 'PIL', 'cv2', 'numpy', 'matplotlib', 'tqdm', 'requests', 'urllib3'
    ]
    
    for logger_name in loggers_to_suppress:
        try:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
        except Exception:
            pass

def _get_merged_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Get dan merge semua level konfigurasi."""
    try:
        # 1. Default config
        default_config = {
            'preprocessing': {
                'img_size': [640, 640],
                'normalization': 'minmax',
                'num_workers': 4,
                'split': 'all',
                'normalize': True,
                'preserve_aspect_ratio': True
            }
        }
        
        # 2. Saved config
        config_manager = get_config_manager()
        saved_config = config_manager.get_config('preprocessing') if hasattr(config_manager, 'get_config') else {}
        
        # 3. Runtime config
        runtime_config = config or {}
        
        # 4. Environment config
        env_manager = get_environment_manager()
        paths = get_paths_for_environment(env_manager.is_colab, env_manager.is_drive_mounted)
        env_config = {
            'data_dir': paths['data_root'],
            'preprocessed_dir': f"{paths['data_root']}/preprocessed"
        }
        
        # Merge order: default -> saved -> runtime -> environment
        merged_config = {**default_config, **saved_config, **runtime_config, **env_config}
        return merged_config
        
    except Exception:
        return config or {'preprocessing': {'img_size': [640, 640], 'normalization': 'minmax', 'num_workers': 4, 'split': 'all'}}

def _create_ui_components_safe(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create UI components dengan error handling."""
    try:
        ui_components = create_preprocessing_main_ui(config)
        
        # Add essential config dan paths
        ui_components.update({
            'preprocessing_initialized': True,
            'config_manager': get_config_manager(),
            'data_dir': config.get('data_dir', 'data'),
            'preprocessed_dir': config.get('preprocessed_dir', 'data/preprocessed')
        })
        
        return ui_components
        
    except Exception as e:
        raise Exception(f"UI component creation failed: {str(e)}")

def _setup_logger_bridge_safe(ui_components: Dict[str, Any]):
    """Setup logger bridge dengan error handling."""
    try:
        logger_bridge = create_ui_logger_bridge(ui_components, PREPROCESSING_LOGGER_NAMESPACE)
        return logger_bridge
    except Exception as e:
        # Fallback logger
        from smartcash.common.logger import get_logger
        return get_logger(PREPROCESSING_LOGGER_NAMESPACE)

def _validate_and_finalize_setup(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validasi comprehensive setup."""
    # Check required components
    required_components = [
        'ui', 'preprocess_button', 'cleanup_button', 'check_button',
        'save_button', 'reset_button', 'log_output', 'status_panel'
    ]
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    if missing_components:
        return {
            'valid': False,
            'message': f"Missing critical components: {', '.join(missing_components)}"
        }
    
    # Check handler setup
    setup_warnings = ui_components.get('setup_warnings', [])
    if setup_warnings:
        # Log warnings tapi tetap valid
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Handler setup warnings: {'; '.join(setup_warnings)}")
    
    return {'valid': True, 'message': 'Setup berhasil'}

def _get_cached_ui_or_refresh(config: Optional[Dict[str, Any]]):
    """Get cached UI atau refresh jika config berbeda."""
    global _CACHED_UI_COMPONENTS
    
    if config and _CACHED_UI_COMPONENTS:
        # Update config jika ada perubahan
        try:
            current_config = _get_merged_config(config)
            _update_ui_with_config(_CACHED_UI_COMPONENTS, current_config)
        except Exception:
            pass
    
    return _CACHED_UI_COMPONENTS['ui'] if _CACHED_UI_COMPONENTS else None

def _update_ui_with_config(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Update UI components dengan config baru menggunakan config extractor."""
    try:
        from smartcash.ui.dataset.preprocessing.utils import get_config_extractor
        config_extractor = get_config_extractor(ui_components)
        config_extractor.apply_config_to_ui(config)
    except Exception:
        pass

def _create_error_fallback_ui(error_message: str):
    """Create enhanced error fallback UI."""
    import ipywidgets as widgets
    
    error_html = f"""
    <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffc107; 
                border-radius: 8px; color: #856404; margin: 10px 0;">
        <h4 style="color: #721c24; margin-top: 0;">⚠️ Error Inisialisasi Preprocessing UI</h4>
        <div style="margin: 15px 0;">
            <strong>Error Detail:</strong><br>
            <code style="background: #f8f9fa; padding: 5px; border-radius: 3px;">
                {error_message}
            </code>
        </div>
        <div style="margin: 15px 0;">
            <strong>🔧 Solusi yang Bisa Dicoba:</strong>
            <ol>
                <li>Restart kernel Colab dan jalankan ulang cell</li>
                <li>Clear output semua cell dan jalankan dari awal</li>
                <li>Periksa koneksi internet dan Google Drive</li> 
                <li>Pastikan dataset sudah di-download terlebih dahulu</li>
            </ol>
        </div>
        <div style="margin-top: 15px; padding: 10px; background: #e7f3ff; border-radius: 5px;">
            <strong>💡 Tip:</strong> Jika masalah berlanjut, coba gunakan parameter <code>force_refresh=True</code>
        </div>
    </div>
    """
    
    return widgets.HTML(error_html)

def reset_preprocessing_module():
    """Reset module initialization untuk testing/debugging."""
    global _MODULE_INITIALIZED, _CACHED_UI_COMPONENTS
    _MODULE_INITIALIZED = False
    _CACHED_UI_COMPONENTS = None

# Alias untuk kompatibilitas
initialize_preprocessing_ui = initialize_dataset_preprocessing_ui