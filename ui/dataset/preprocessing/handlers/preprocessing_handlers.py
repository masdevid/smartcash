"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Fixed handlers dengan direct import resolution dan simplified backend integration
"""

from typing import Dict, Any, Callable
from smartcash.ui.dataset.preprocessing.utils import (
    clear_outputs, handle_ui_error, show_ui_success, log_to_accordion,
    create_dual_progress_callback, setup_dual_progress_tracker,
    complete_progress_tracker, error_progress_tracker,
    setup_backend_button_management, validate_dataset_ready,
    check_preprocessed_exists, create_backend_preprocessor,
    create_backend_checker, create_backend_cleanup_service,
    _convert_ui_to_backend_config
)
from smartcash.common.logger import get_logger

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """🚀 Setup handlers dengan complete backend integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Phase 1: Setup Progress Bridge Integration
        _setup_progress_integration(ui_components)
        
        # Phase 2: Setup Backend Button Management
        _setup_button_management(ui_components)
        
        # Phase 3: Setup Config Integration
        _setup_config_integration(ui_components, config)
        
        # Phase 4: Setup Operation Handlers
        _setup_operation_handlers(ui_components, config)
        
        # Phase 5: Test Connectivity
        _test_and_log_connectivity(ui_components, config)
        
        logger.info("✅ Preprocessing handlers setup completed")
        return ui_components
        
    except Exception as e:
        error_msg = f"❌ Error setup handlers: {str(e)}"
        logger.error(error_msg)
        handle_ui_error(ui_components, error_msg)
        return ui_components

def _setup_progress_integration(ui_components: Dict[str, Any]):
    """📊 Setup progress integration"""
    try:
        if 'progress_callback' not in ui_components:
            progress_callback = create_dual_progress_callback(ui_components)
            ui_components['progress_callback'] = progress_callback
        
        # Register utility functions
        ui_components.update({
            'clear_outputs': clear_outputs,
            'handle_ui_error': handle_ui_error,
            'show_ui_success': show_ui_success,
            'log_to_accordion': log_to_accordion,
            'setup_dual_progress_tracker': setup_dual_progress_tracker,
            'complete_progress_tracker': complete_progress_tracker,
            'error_progress_tracker': error_progress_tracker,
            'validate_dataset_ready': lambda cfg: validate_dataset_ready(cfg),
            'check_preprocessed_exists': check_preprocessed_exists,
            'create_backend_preprocessor': lambda cfg: create_backend_preprocessor(cfg, progress_callback=ui_components.get('progress_callback')),
            'create_backend_checker': create_backend_checker,
            'create_backend_cleanup_service': lambda cfg: create_backend_cleanup_service(cfg, ui_components=ui_components),
            '_convert_ui_to_backend_config': lambda: _convert_ui_to_backend_config(ui_components)
        })
    except Exception as e:
        get_logger('preprocessing_handlers').warning(f"⚠️ Progress integration warning: {str(e)}")

def _setup_button_management(ui_components: Dict[str, Any]):
    """🔘 Setup backend-aware button management"""
    try:
        button_manager = setup_backend_button_management(ui_components)
        ui_components['button_manager'] = button_manager
        
        # Enhanced progress callback dengan button updates
        if 'progress_callback' in ui_components:
            original_callback = ui_components['progress_callback']
            enhanced_callback = button_manager.register_progress_updates(original_callback)
            ui_components['progress_callback'] = enhanced_callback
            
    except Exception as e:
        get_logger('preprocessing_handlers').warning(f"⚠️ Button management warning: {str(e)}")

def _setup_config_integration(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """🔄 Setup config handler integration"""
    try:
        config_handler = ui_components.get('config_handler')
        if config_handler:
            # Set UI components for logging
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            # Set progress callback
            if hasattr(config_handler, 'set_progress_callback'):
                progress_callback = ui_components.get('progress_callback')
                if progress_callback:
                    config_handler.set_progress_callback(progress_callback)
            
            # Setup save/reset handlers
            _setup_config_handlers(ui_components)
            
    except Exception as e:
        get_logger('preprocessing_handlers').warning(f"⚠️ Config integration warning: {str(e)}")

def _setup_config_handlers(ui_components: Dict[str, Any]):
    """⚙️ Setup save/reset button handlers"""
    def enhanced_save_config(button=None):
        clear_outputs(ui_components)
        
        try:
            button_manager = ui_components.get('button_manager')
            if button_manager:
                button_manager.push_backend_operation('config_save')
            
            config_handler = ui_components.get('config_handler')
            if config_handler:
                success = config_handler.save_config(ui_components)
                if button_manager:
                    button_manager.pop_backend_operation(success, "Config save completed" if success else "Config save failed")
            else:
                handle_ui_error(ui_components, "❌ Config handler tidak tersedia")
                
        except Exception as e:
            if 'button_manager' in ui_components:
                ui_components['button_manager'].pop_backend_operation(False, f"Config save error: {str(e)}")
            handle_ui_error(ui_components, f"❌ Error save config: {str(e)}")
    
    def enhanced_reset_config(button=None):
        clear_outputs(ui_components)
        
        try:
            button_manager = ui_components.get('button_manager')
            if button_manager:
                button_manager.push_backend_operation('config_reset')
            
            config_handler = ui_components.get('config_handler')
            if config_handler:
                success = config_handler.reset_config(ui_components)
                if button_manager:
                    button_manager.pop_backend_operation(success, "Config reset completed" if success else "Config reset failed")
            else:
                handle_ui_error(ui_components, "❌ Config handler tidak tersedia")
                
        except Exception as e:
            if 'button_manager' in ui_components:
                ui_components['button_manager'].pop_backend_operation(False, f"Config reset error: {str(e)}")
            handle_ui_error(ui_components, f"❌ Error reset config: {str(e)}")
    
    # Bind handlers
    if save_button := ui_components.get('save_button'):
        save_button.on_click(enhanced_save_config)
    if reset_button := ui_components.get('reset_button'):
        reset_button.on_click(enhanced_reset_config)

def _setup_operation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """⚙️ Setup operation button handlers"""
    def coordinated_preprocessing_handler(button=None):
        return _execute_coordinated_operation(ui_components, 'preprocessing', config)
    
    def coordinated_check_handler(button=None):
        return _execute_coordinated_operation(ui_components, 'validation', config)
    
    def coordinated_cleanup_handler(button=None):
        return _execute_coordinated_operation(ui_components, 'cleanup', config)
    
    # Bind handlers
    operation_handlers = {
        'preprocess_button': coordinated_preprocessing_handler,
        'check_button': coordinated_check_handler,
        'cleanup_button': coordinated_cleanup_handler
    }
    
    for button_key, handler in operation_handlers.items():
        if button := ui_components.get(button_key):
            if hasattr(button, 'on_click'):
                button.on_click(handler)

def _execute_coordinated_operation(ui_components: Dict[str, Any], operation_type: str, config: Dict[str, Any]) -> bool:
    """🎯 Execute operation dengan complete coordination"""
    logger = get_logger('preprocessing_handlers')
    button_manager = ui_components.get('button_manager')
    
    try:
        clear_outputs(ui_components)
        log_to_accordion(ui_components, f"🚀 Memulai {operation_type}", "info")
        
        if button_manager:
            button_manager.push_backend_operation(operation_type)
        
        # Extract config
        ui_config = ui_components['_convert_ui_to_backend_config']()
        
        # Execute by type
        if operation_type == 'preprocessing':
            success, result = _execute_preprocessing(ui_components, ui_config)
        elif operation_type == 'validation':
            success, result = _execute_validation(ui_components, ui_config)
        elif operation_type == 'cleanup':
            success, result = _execute_cleanup(ui_components, ui_config)
        else:
            success, result = False, f"Unknown operation: {operation_type}"
        
        # Handle results
        if success:
            _handle_success(ui_components, operation_type, result)
            if button_manager:
                button_manager.pop_backend_operation(True, f"{operation_type} completed")
        else:
            _handle_failure(ui_components, operation_type, result)
            if button_manager:
                button_manager.pop_backend_operation(False, f"{operation_type} failed")
        
        return success
        
    except Exception as e:
        error_msg = f"❌ Error {operation_type}: {str(e)}"
        logger.error(error_msg)
        handle_ui_error(ui_components, error_msg)
        
        if button_manager:
            button_manager.pop_backend_operation(False, f"{operation_type} error")
        
        return False

def _execute_preprocessing(ui_components: Dict[str, Any], config: Dict[str, Any]) -> tuple[bool, Any]:
    """🚀 Execute preprocessing operation"""
    try:
        # Validation
        is_valid, validation_msg = ui_components['validate_dataset_ready'](config)
        if not is_valid:
            return False, f"Pre-validation failed: {validation_msg}"
        
        log_to_accordion(ui_components, f"✅ {validation_msg}", "success")
        
        # Setup progress
        progress_manager = ui_components.get('progress_manager')
        if progress_manager:
            progress_manager.setup_for_operation("Dataset Preprocessing")
        
        # Create service
        preprocessor = ui_components['create_backend_preprocessor'](config)
        if not preprocessor:
            return False, "Failed to create preprocessing service"
        
        # Execute
        result = preprocessor.preprocess_dataset()
        
        if result.get('success', False):
            stats = result.get('stats', {})
            message = f"Preprocessing berhasil: {stats.get('output', {}).get('total_processed', 0):,} gambar diproses"
            return True, {'message': message, 'stats': stats}
        else:
            return False, result.get('message', 'Preprocessing failed')
            
    except Exception as e:
        return False, f"Preprocessing error: {str(e)}"

def _execute_validation(ui_components: Dict[str, Any], config: Dict[str, Any]) -> tuple[bool, Any]:
    """🔍 Execute validation operation"""
    try:
        progress_manager = ui_components.get('progress_manager')
        if progress_manager:
            progress_manager.setup_for_operation("Dataset Validation")
        
        is_valid, source_msg = ui_components['validate_dataset_ready'](config)
        preprocessed_exists, preprocessed_count = ui_components['check_preprocessed_exists'](config)
        
        if is_valid:
            message = f"Dataset valid"
            if preprocessed_exists:
                message += f" + {preprocessed_count:,} preprocessed file tersedia"
            
            return True, {
                'message': message,
                'source_valid': True,
                'preprocessed_exists': preprocessed_exists,
                'preprocessed_count': preprocessed_count
            }
        else:
            return False, source_msg
            
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def _execute_cleanup(ui_components: Dict[str, Any], config: Dict[str, Any]) -> tuple[bool, Any]:
    """🧹 Execute cleanup operation"""
    try:
        progress_manager = ui_components.get('progress_manager')
        if progress_manager:
            progress_manager.setup_for_operation("Dataset Cleanup")
        
        cleanup_service = ui_components['create_backend_cleanup_service'](config)
        if not cleanup_service:
            return False, "Failed to create cleanup service"
        
        result = cleanup_service.cleanup_preprocessed_data()
        
        if result.get('success', False):
            if result.get('cancelled'):
                return True, {'message': "Cleanup dibatalkan", 'cancelled': True}
            
            stats = result.get('stats', {})
            files_removed = stats.get('files_removed', 0)
            message = f"Cleanup berhasil: {files_removed:,} file dihapus"
            return True, {'message': message, 'stats': stats}
        else:
            return False, result.get('message', 'Cleanup failed')
            
    except Exception as e:
        return False, f"Cleanup error: {str(e)}"

def _handle_success(ui_components: Dict[str, Any], operation_type: str, result: Dict[str, Any]):
    """✅ Handle successful operation"""
    message = result.get('message', f"{operation_type} completed")
    
    progress_manager = ui_components.get('progress_manager')
    if progress_manager:
        progress_manager.complete_operation(message)
    
    log_to_accordion(ui_components, f"✅ {message}", "success")
    show_ui_success(ui_components, message)
    
    # Operation-specific handling
    if operation_type == 'preprocessing':
        stats = result.get('stats', {})
        if stats:
            performance = stats.get('performance', {})
            processing_time = performance.get('processing_time_seconds', 0)
            if processing_time > 0:
                log_to_accordion(ui_components, f"⏱️ Processing time: {processing_time:.1f} detik", "info")
    
    elif operation_type == 'validation':
        if result.get('preprocessed_exists'):
            count = result.get('preprocessed_count', 0)
            log_to_accordion(ui_components, f"💾 Preprocessed data: {count:,} file tersedia", "info")

def _handle_failure(ui_components: Dict[str, Any], operation_type: str, error_result: str):
    """❌ Handle failed operation"""
    progress_manager = ui_components.get('progress_manager')
    if progress_manager:
        progress_manager.error_operation(f"{operation_type} failed")
    
    handle_ui_error(ui_components, f"❌ {operation_type.capitalize()} gagal: {error_result}")

def _test_and_log_connectivity(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """🧪 Test dan log backend connectivity"""
    try:
        # Simple connectivity test
        results = {
            'validation_service': False,
            'preprocessing_service': False,
            'cleanup_service': False,
            'progress_integration': False
        }
        
        # Test each service
        try:
            checker = create_backend_checker(config)
            results['validation_service'] = checker is not None
        except Exception:
            pass
        
        try:
            preprocessor = create_backend_preprocessor(config)
            results['preprocessing_service'] = preprocessor is not None
        except Exception:
            pass
        
        try:
            cleanup = create_backend_cleanup_service(config, ui_components=ui_components)
            results['cleanup_service'] = cleanup is not None
        except Exception:
            pass
        
        try:
            if 'progress_callback' in ui_components:
                ui_components['progress_callback']('test', 50, 100, 'Test message')
                results['progress_integration'] = True
        except Exception:
            pass
        
        # Log results
        working_services = sum(results.values())
        total_services = len(results)
        
        if working_services == total_services:
            log_to_accordion(ui_components, f"✅ Backend integration: {working_services}/{total_services} services ready", "success")
        else:
            log_to_accordion(ui_components, f"⚠️ Backend integration: {working_services}/{total_services} services ready", "warning")
            
    except Exception as e:
        get_logger('preprocessing_handlers').warning(f"⚠️ Connectivity test warning: {str(e)}")

# Backward compatibility
def setup_config_handlers_fixed(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """🔧 Backward compatibility"""
    _setup_config_handlers(ui_components)