"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Enhanced initializer dengan complete service layer integration dan optimized setup
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

# Import enhanced SRP handlers dengan service integration
from smartcash.ui.dataset.preprocessing.handlers.config_handler import setup_config_handlers
from smartcash.ui.dataset.preprocessing.handlers.dataset_checker import setup_dataset_checker
from smartcash.ui.dataset.preprocessing.handlers.cleanup_executor import setup_cleanup_executor  
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_executor import setup_preprocessing_executor
from smartcash.ui.dataset.preprocessing.handlers.progress_handlers import setup_progress_handlers

# Import service layer
from smartcash.ui.dataset.preprocessing.services.ui_preprocessing_service import create_ui_preprocessing_service

# Global state management
_MODULE_INITIALIZED = False
_CACHED_UI_COMPONENTS = None

def initialize_dataset_preprocessing_ui(env=None, config=None, force_refresh=False):
    """
    Enhanced initializer dengan complete service layer integration.
    
    Args:
        env: Environment manager (optional)
        config: Konfigurasi preprocessing (optional)
        force_refresh: Force refresh UI components
        
    Returns:
        Widget UI preprocessing atau error fallback UI
    """
    global _MODULE_INITIALIZED, _CACHED_UI_COMPONENTS
    
    # Return cached UI jika sudah initialized dan tidak force refresh
    if _MODULE_INITIALIZED and _CACHED_UI_COMPONENTS and not force_refresh:
        return _get_cached_ui_or_refresh(config)
    
    try:
        # 1. Setup comprehensive log suppression
        _setup_comprehensive_log_suppression()
        
        # 2. Get merged config dengan environment detection
        merged_config = _get_enhanced_merged_config(config)
        
        # 3. Create UI components dengan responsive layout
        ui_components = _create_ui_components_safe(merged_config)
        
        # 4. Setup logger bridge dengan namespace
        logger_bridge = _setup_logger_bridge_safe(ui_components)
        ui_components['logger'] = logger_bridge
        ui_components['logger_namespace'] = PREPROCESSING_LOGGER_NAMESPACE
        
        # 5. Setup progress handlers FIRST (critical untuk service integration)
        ui_components = setup_progress_handlers(ui_components)
        
        # 6. Setup service layer integration
        ui_service = create_ui_preprocessing_service(ui_components)
        ui_components['ui_service'] = ui_service
        
        # 7. Setup enhanced handlers dengan service integration
        ui_components = _setup_service_integrated_handlers(ui_components, merged_config, env)
        
        # 8. Comprehensive validation
        validation_result = _validate_service_integration_setup(ui_components)
        
        if not validation_result['valid']:
            return _create_error_fallback_ui(f"Service integration failed: {validation_result['message']}")
        
        # 9. Cache dan finalize
        _CACHED_UI_COMPONENTS = ui_components
        _MODULE_INITIALIZED = True
        
        logger_bridge.success("✅ Preprocessing UI initialized dengan complete service layer integration")
        return ui_components['ui']
        
    except Exception as e:
        return _create_error_fallback_ui(f"Initialization error: {str(e)}")

def _setup_service_integrated_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan complete service layer integration."""
    handlers_setup = [
        ("Config Handler", lambda: setup_config_handlers(ui_components, config)),
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
    
    # Log warnings untuk failed handlers tapi tetap lanjut
    if failed_handlers:
        ui_components['setup_warnings'] = failed_handlers
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Some handlers failed: {'; '.join(failed_handlers)}")
    
    return ui_components

def _validate_service_integration_setup(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive validation untuk service integration setup."""
    # Critical UI components
    required_ui_components = [
        'ui', 'preprocess_button', 'cleanup_button', 'check_button',
        'save_button', 'reset_button', 'log_output', 'status_panel'
    ]
    
    # Critical service integration components
    required_service_components = [
        'ui_service', 'show_for_operation', 'update_progress', 
        'complete_operation', 'error_operation'
    ]
    
    # Critical handler functions
    required_handlers = [
        'execute_preprocessing', 'execute_check', 'execute_cleanup',
        'save_config', 'reset_config'
    ]
    
    missing_components = []
    
    # Check UI components
    for comp in required_ui_components:
        if comp not in ui_components:
            missing_components.append(f"UI: {comp}")
    
    # Check service components
    for comp in required_service_components:
        if comp not in ui_components:
            missing_components.append(f"Service: {comp}")
    
    # Check handlers
    for handler in required_handlers:
        if handler not in ui_components:
            missing_components.append(f"Handler: {handler}")
    
    if missing_components:
        return {
            'valid': False,
            'message': f"Missing critical components: {', '.join(missing_components)}"
        }
    
    # Validate service layer functionality
    service_status = ui_components['ui_service'].get_service_status()
    if not any(service_status.values()):
        return {
            'valid': False,
            'message': "Service layer tidak properly initialized"
        }
    
    return {
        'valid': True,
        'message': 'Complete service integration setup berhasil',
        'service_status': service_status
    }

def _get_enhanced_merged_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Enhanced config merging dengan environment detection dan path resolution."""
    try:
        # 1. Environment-aware default config
        env_manager = get_environment_manager()
        paths = get_paths_for_environment(env_manager.is_colab, env_manager.is_drive_mounted)
        
        default_config = {
            'data': {'dir': paths['data_root']},
            'preprocessing': {
                'img_size': [640, 640],
                'normalize': True,
                'normalization_method': 'minmax',
                'num_workers': 4,
                'split': 'all',
                'preserve_aspect_ratio': True,
                'output_dir': paths.get('preprocessed', 'data/preprocessed')
            }
        }
        
        # 2. Saved config dari config manager
        config_manager = get_config_manager()
        saved_config = {}
        try:
            saved_config = config_manager.get_config('preprocessing')
        except Exception:
            pass
        
        # 3. Runtime config
        runtime_config = config or {}
        
        # 4. Merge dengan priority: default -> saved -> runtime
        merged_config = {**default_config}
        
        # Merge saved config intelligently
        if saved_config and 'preprocessing' in saved_config:
            merged_config['preprocessing'].update(saved_config['preprocessing'])
        
        # Merge runtime config
        if 'preprocessing' in runtime_config:
            merged_config['preprocessing'].update(runtime_config['preprocessing'])
        if 'data' in runtime_config:
            merged_config['data'].update(runtime_config['data'])
        
        return merged_config
        
    except Exception as e:
        # Fallback ke basic config
        return {
            'data': {'dir': 'data'},
            'preprocessing': {
                'img_size': [640, 640], 'normalize': True, 
                'num_workers': 4, 'split': 'all'
            }
        }

def _setup_comprehensive_log_suppression():
    """Enhanced log suppression untuk service layer."""
    loggers_to_suppress = [
        # Core preprocessing services
        'smartcash.dataset.preprocessor', 'smartcash.dataset.preprocessor.core',
        'smartcash.dataset.preprocessor.processors', 'smartcash.dataset.preprocessor.operations',
        
        # Existing suppressions
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

def _create_ui_components_safe(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create UI components dengan enhanced error handling."""
    try:
        ui_components = create_preprocessing_main_ui(config)
        
        # Add essential metadata
        ui_components.update({
            'preprocessing_initialized': True,
            'service_integration_enabled': True,
            'config_manager': get_config_manager(),
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'preprocessed_dir': config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        })
        
        return ui_components
        
    except Exception as e:
        raise Exception(f"Enhanced UI component creation failed: {str(e)}")

def _setup_logger_bridge_safe(ui_components: Dict[str, Any]):
    """Setup logger bridge dengan enhanced error handling."""
    try:
        logger_bridge = create_ui_logger_bridge(ui_components, PREPROCESSING_LOGGER_NAMESPACE)
        return logger_bridge
    except Exception as e:
        # Fallback ke common logger
        from smartcash.common.logger import get_logger
        return get_logger(PREPROCESSING_LOGGER_NAMESPACE)

def _get_cached_ui_or_refresh(config: Optional[Dict[str, Any]]):
    """Get cached UI dengan config refresh capability."""
    global _CACHED_UI_COMPONENTS
    
    if config and _CACHED_UI_COMPONENTS:
        try:
            # Update config jika ada
            current_config = _get_enhanced_merged_config(config)
            
            # Update UI service dengan new config
            if 'ui_service' in _CACHED_UI_COMPONENTS:
                _CACHED_UI_COMPONENTS['ui_service'].cleanup_service_cache()
            
            # Apply config ke UI
            from smartcash.ui.dataset.preprocessing.utils.config_extractor import get_config_extractor
            config_extractor = get_config_extractor(_CACHED_UI_COMPONENTS)
            config_extractor.apply_config_to_ui(current_config)
            
        except Exception as e:
            logger = _CACHED_UI_COMPONENTS.get('logger')
            if logger:
                logger.warning(f"⚠️ Config refresh error: {str(e)}")
    
    return _CACHED_UI_COMPONENTS['ui'] if _CACHED_UI_COMPONENTS else None

def _create_error_fallback_ui(error_message: str):
    """Create enhanced error fallback UI dengan service integration info."""
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
                <li>Coba dengan parameter <code>force_refresh=True</code></li>
            </ol>
        </div>
        <div style="margin-top: 15px; padding: 10px; background: #e7f3ff; border-radius: 5px;">
            <strong>🚀 Enhanced Features:</strong>
            <ul style="margin: 5px 0; padding-left: 20px;">
                <li>Service layer integration dengan preprocessing factory</li>
                <li>Multi-level progress tracking dengan tqdm compatibility</li>
                <li>Optimized batch processing dengan ThreadPoolExecutor</li>
                <li>Symlink-safe cleanup operations</li>
            </ul>
        </div>
    </div>
    """
    
    return widgets.HTML(error_html)

def reset_preprocessing_module():
    """Reset module initialization untuk testing/debugging."""
    global _MODULE_INITIALIZED, _CACHED_UI_COMPONENTS
    
    # Cleanup service cache jika ada
    if _CACHED_UI_COMPONENTS and 'ui_service' in _CACHED_UI_COMPONENTS:
        try:
            _CACHED_UI_COMPONENTS['ui_service'].cleanup_service_cache()
        except Exception:
            pass
    
    _MODULE_INITIALIZED = False
    _CACHED_UI_COMPONENTS = None

# Alias untuk kompatibilitas
initialize_preprocessing_ui = initialize_dataset_preprocessing_ui