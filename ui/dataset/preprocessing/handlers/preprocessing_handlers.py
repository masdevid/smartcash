"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Updated handlers dengan complete Progress Bridge integration dan seamless backend-UI coordination
"""

from typing import Dict, Any, Callable
from smartcash.ui.dataset.preprocessing.utils import (
    clear_outputs, handle_ui_error, show_ui_success, log_to_accordion,
    setup_backend_integration, test_backend_connectivity
)
from smartcash.common.logger import get_logger

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """🚀 Enhanced setup dengan complete backend integration dan Progress Bridge coordination"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Phase 1: Complete Backend Integration Setup
        logger.info("🔗 Setting up complete backend integration")
        ui_components = setup_backend_integration(ui_components, config)
        
        # Phase 2: Config Handler Integration
        _setup_enhanced_config_integration(ui_components, config)
        
        # Phase 3: Operation Handlers dengan Backend Coordination
        _setup_coordinated_operation_handlers(ui_components, config)
        
        # Phase 4: Real-time Status Sync
        _setup_realtime_status_sync(ui_components)
        
        # Phase 5: Backend Connectivity Test
        connectivity_results = test_backend_connectivity(ui_components, config)
        _log_connectivity_status(ui_components, connectivity_results)
        
        logger.info("✅ Preprocessing handlers setup completed dengan backend integration")
        return ui_components
        
    except Exception as e:
        error_msg = f"❌ Error setup preprocessing handlers: {str(e)}"
        logger.error(error_msg)
        handle_ui_error(ui_components, error_msg)
        return ui_components

def _setup_enhanced_config_integration(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """🔄 Enhanced config handler integration dengan Progress Bridge"""
    try:
        config_handler = ui_components.get('config_handler')
        if not config_handler:
            return
        
        # Set UI components untuk logging
        if hasattr(config_handler, 'set_ui_components'):
            config_handler.set_ui_components(ui_components)
        
        # Set progress callback untuk real-time feedback
        if hasattr(config_handler, 'set_progress_callback'):
            progress_callback = ui_components.get('progress_callback')
            if progress_callback:
                config_handler.set_progress_callback(progress_callback)
        
        # Enhanced config operation handlers
        def enhanced_save_config(button=None):
            clear_outputs(ui_components)
            
            try:
                # Notify button manager
                button_manager = ui_components.get('button_manager')
                if button_manager:
                    button_manager.push_backend_operation('config_save')
                
                # Execute save dengan progress tracking
                success = config_handler.save_config(ui_components)
                
                # Complete operation
                if button_manager:
                    button_manager.pop_backend_operation(success, "Config save completed" if success else "Config save failed")
                
            except Exception as e:
                if button_manager:
                    button_manager.pop_backend_operation(False, f"Config save error: {str(e)}")
                handle_ui_error(ui_components, f"❌ Error save config: {str(e)}")
        
        def enhanced_reset_config(button=None):
            clear_outputs(ui_components)
            
            try:
                # Notify button manager
                button_manager = ui_components.get('button_manager')
                if button_manager:
                    button_manager.push_backend_operation('config_reset')
                
                # Execute reset dengan progress tracking
                success = config_handler.reset_config(ui_components)
                
                # Complete operation
                if button_manager:
                    button_manager.pop_backend_operation(success, "Config reset completed" if success else "Config reset failed")
                
            except Exception as e:
                if button_manager:
                    button_manager.pop_backend_operation(False, f"Config reset error: {str(e)}")
                handle_ui_error(ui_components, f"❌ Error reset config: {str(e)}")
        
        # Bind enhanced handlers
        if save_button := ui_components.get('save_button'):
            save_button.on_click(enhanced_save_config)
        if reset_button := ui_components.get('reset_button'):
            reset_button.on_click(enhanced_reset_config)
        
    except Exception as e:
        get_logger('preprocessing_handlers').warning(f"⚠️ Config integration warning: {str(e)}")

def _setup_coordinated_operation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """⚙️ Setup operation handlers dengan complete backend coordination"""
    
    def coordinated_preprocessing_handler(button=None):
        """🚀 Coordinated preprocessing dengan full backend integration"""
        return _execute_coordinated_operation(ui_components, 'preprocessing', config)
    
    def coordinated_check_handler(button=None):
        """🔍 Coordinated check dengan backend validation"""
        return _execute_coordinated_operation(ui_components, 'validation', config)
    
    def coordinated_cleanup_handler(button=None):
        """🧹 Coordinated cleanup dengan UI confirmation integration"""
        return _execute_coordinated_operation(ui_components, 'cleanup', config)
    
    # Bind coordinated handlers
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
    """🎯 Execute operation dengan complete coordination antara UI dan backend"""
    logger = get_logger('preprocessing_handlers')
    button_manager = ui_components.get('button_manager')
    
    try:
        # Phase 1: Setup dan Preparation
        clear_outputs(ui_components)
        log_to_accordion(ui_components, f"🚀 Memulai {operation_type}", "info")
        
        # Notify button manager untuk operation start
        if button_manager:
            button_manager.push_backend_operation(operation_type)
        
        # Phase 2: Extract dan Convert Config
        ui_config = ui_components['_convert_ui_to_backend_config']()
        
        # Phase 3: Execute Operation berdasarkan type
        success, result = _execute_operation_by_type(ui_components, operation_type, ui_config)
        
        # Phase 4: Process Results dan UI Updates
        if success:
            _handle_operation_success(ui_components, operation_type, result)
            if button_manager:
                button_manager.pop_backend_operation(True, f"{operation_type} completed successfully")
        else:
            _handle_operation_failure(ui_components, operation_type, result)
            if button_manager:
                button_manager.pop_backend_operation(False, f"{operation_type} failed")
        
        return success
        
    except Exception as e:
        error_msg = f"❌ Error {operation_type}: {str(e)}"
        logger.error(error_msg)
        handle_ui_error(ui_components, error_msg)
        
        if button_manager:
            button_manager.pop_backend_operation(False, f"{operation_type} error: {str(e)}")
        
        return False

def _execute_operation_by_type(ui_components: Dict[str, Any], operation_type: str, config: Dict[str, Any]) -> tuple[bool, Any]:
    """🔧 Execute specific operation dengan backend service"""
    
    if operation_type == 'preprocessing':
        return _execute_preprocessing_operation(ui_components, config)
    elif operation_type == 'validation':
        return _execute_validation_operation(ui_components, config)
    elif operation_type == 'cleanup':
        return _execute_cleanup_operation(ui_components, config)
    else:
        return False, f"Unknown operation type: {operation_type}"

def _execute_preprocessing_operation(ui_components: Dict[str, Any], config: Dict[str, Any]) -> tuple[bool, Any]:
    """🚀 Execute preprocessing dengan backend service integration"""
    try:
        # Pre-validation
        is_valid, validation_msg = ui_components['validate_dataset_ready'](config)
        if not is_valid:
            return False, f"Pre-validation failed: {validation_msg}"
        
        log_to_accordion(ui_components, f"✅ {validation_msg}", "success")
        
        # Setup progress tracking
        progress_manager = ui_components.get('progress_manager')
        if progress_manager:
            progress_manager.setup_for_operation("Dataset Preprocessing")
        
        # Create backend service dengan progress callback
        preprocessor = ui_components['create_backend_preprocessor'](config)
        if not preprocessor:
            return False, "Failed to create preprocessing service"
        
        # Execute preprocessing
        result = preprocessor.preprocess_dataset()
        
        if result.get('success', False):
            stats = result.get('stats', {})
            message = f"Preprocessing berhasil: {stats.get('output', {}).get('total_processed', 0):,} gambar diproses"
            return True, {'message': message, 'stats': stats}
        else:
            return False, result.get('message', 'Preprocessing failed')
            
    except Exception as e:
        return False, f"Preprocessing error: {str(e)}"

def _execute_validation_operation(ui_components: Dict[str, Any], config: Dict[str, Any]) -> tuple[bool, Any]:
    """🔍 Execute validation dengan backend checker"""
    try:
        # Setup progress
        progress_manager = ui_components.get('progress_manager')
        if progress_manager:
            progress_manager.setup_for_operation("Dataset Validation")
        
        # Validate source dataset
        is_valid, source_msg = ui_components['validate_dataset_ready'](config)
        
        # Check preprocessed data
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

def _execute_cleanup_operation(ui_components: Dict[str, Any], config: Dict[str, Any]) -> tuple[bool, Any]:
    """🧹 Execute cleanup dengan UI confirmation integration"""
    try:
        # Setup progress
        progress_manager = ui_components.get('progress_manager')
        if progress_manager:
            progress_manager.setup_for_operation("Dataset Cleanup")
        
        # Create cleanup service dengan UI integration
        cleanup_service = ui_components['create_backend_cleanup_service'](config)
        if not cleanup_service:
            return False, "Failed to create cleanup service"
        
        # Execute cleanup (akan handle UI confirmation internally)
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

def _handle_operation_success(ui_components: Dict[str, Any], operation_type: str, result: Dict[str, Any]):
    """✅ Handle successful operation dengan appropriate UI updates"""
    message = result.get('message', f"{operation_type} completed")
    
    # Update progress tracker
    progress_manager = ui_components.get('progress_manager')
    if progress_manager:
        progress_manager.complete_operation(message)
    
    # Log success
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

def _handle_operation_failure(ui_components: Dict[str, Any], operation_type: str, error_result: str):
    """❌ Handle failed operation dengan appropriate error handling"""
    # Update progress tracker
    progress_manager = ui_components.get('progress_manager')
    if progress_manager:
        progress_manager.error_operation(f"{operation_type} failed")
    
    # Handle error
    handle_ui_error(ui_components, f"❌ {operation_type.capitalize()} gagal: {error_result}")

def _setup_realtime_status_sync(ui_components: Dict[str, Any]):
    """📡 Setup real-time status sync antara UI dan backend"""
    try:
        # Register status update handlers
        def sync_status_with_backend():
            """Sync UI status dengan backend state"""
            try:
                backend_service = ui_components.get('backend_service')
                button_manager = ui_components.get('button_manager')
                
                if backend_service and button_manager:
                    button_manager.sync_with_backend_status(backend_service)
                    
            except Exception:
                pass  # Silent sync untuk prevent breaking UI
        
        # Register periodic sync jika diperlukan
        ui_components['sync_status_with_backend'] = sync_status_with_backend
        
        # Setup progress callback enhancement untuk real-time feedback
        progress_callback = ui_components.get('progress_callback')
        if progress_callback:
            def enhanced_progress_callback(level: str, current: int, total: int, message: str):
                """Enhanced callback dengan status sync"""
                try:
                    # Call original callback
                    progress_callback(level, current, total, message)
                    
                    # Update status panel dengan milestone progress
                    if current in [0, 25, 50, 75, 100] or current == total:
                        update_status_safe = ui_components.get('update_status_safe')
                        if update_status_safe:
                            status_type = "success" if current == total else "info"
                            update_status_safe(message, status_type)
                    
                except Exception:
                    pass  # Silent fail untuk prevent breaking main process
            
            ui_components['progress_callback'] = enhanced_progress_callback
    
    except Exception as e:
        get_logger('preprocessing_handlers').warning(f"⚠️ Real-time status sync warning: {str(e)}")

def _log_connectivity_status(ui_components: Dict[str, Any], connectivity_results: Dict[str, bool]):
    """📊 Log backend connectivity test results"""
    try:
        total_services = len(connectivity_results)
        working_services = sum(connectivity_results.values())
        
        if working_services == total_services:
            status_msg = f"✅ Backend integration: {working_services}/{total_services} services ready"
            log_to_accordion(ui_components, status_msg, "success")
        else:
            status_msg = f"⚠️ Backend integration: {working_services}/{total_services} services ready"
            log_to_accordion(ui_components, status_msg, "warning")
            
            # Log detailed status
            for service, status in connectivity_results.items():
                service_status = "✅" if status else "❌"
                log_to_accordion(ui_components, f"  {service_status} {service.replace('_', ' ').title()}", "info")
    
    except Exception:
        pass  # Silent logging untuk prevent breaking setup

def register_backend_service_to_ui(ui_components: Dict[str, Any], backend_service) -> bool:
    """🔗 Register backend service ke UI dengan complete integration"""
    try:
        # Store backend service reference
        ui_components['backend_service'] = backend_service
        
        # Register progress callback
        progress_callback = ui_components.get('progress_callback')
        if progress_callback and hasattr(backend_service, 'register_progress_callback'):
            backend_service.register_progress_callback(progress_callback)
        
        # Sync button manager
        button_manager = ui_components.get('button_manager')
        if button_manager and hasattr(button_manager, 'sync_with_backend_status'):
            button_manager.sync_with_backend_status(backend_service)
        
        # Register UI updates ke backend service
        if hasattr(backend_service, 'set_ui_components'):
            backend_service.set_ui_components(ui_components)
        
        get_logger('preprocessing_handlers').info("🔗 Backend service registered to UI")
        return True
        
    except Exception as e:
        get_logger('preprocessing_handlers').warning(f"⚠️ Backend service registration warning: {str(e)}")
        return False

def create_operation_wrapper(ui_components: Dict[str, Any], operation_name: str) -> Callable:
    """🎁 Create operation wrapper dengan backend coordination"""
    def operation_wrapper(config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Wrapped operation dengan complete coordination"""
        try:
            # Use current config jika tidak disediakan
            if not config:
                config = ui_components['_convert_ui_to_backend_config']()
            
            # Execute coordinated operation
            success = _execute_coordinated_operation(ui_components, operation_name, config)
            
            return {
                'success': success,
                'operation': operation_name,
                'timestamp': __import__('datetime').datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'operation': operation_name,
                'error': str(e),
                'timestamp': __import__('datetime').datetime.now().isoformat()
            }
    
    return operation_wrapper

def get_operation_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """📊 Get current operation status dari UI dan backend"""
    try:
        button_manager = ui_components.get('button_manager')
        backend_service = ui_components.get('backend_service')
        progress_manager = ui_components.get('progress_manager')
        
        status = {
            'ui_ready': True,
            'backend_connected': backend_service is not None,
            'operation_active': False,
            'current_operation': None,
            'progress_tracking': progress_manager is not None
        }
        
        # Check button manager status
        if button_manager:
            status['operation_active'] = button_manager.is_backend_operation_active()
            status['current_operation'] = button_manager.get_current_backend_operation()
        
        # Check backend service status
        if backend_service and hasattr(backend_service, 'is_processing'):
            try:
                status['backend_processing'] = backend_service.is_processing()
            except Exception:
                status['backend_processing'] = False
        
        return status
        
    except Exception as e:
        return {
            'ui_ready': False,
            'error': str(e),
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }

# Enhanced utility exports
def create_preprocessing_operation_wrapper(ui_components: Dict[str, Any]) -> Callable:
    """🚀 Create preprocessing operation wrapper"""
    return create_operation_wrapper(ui_components, 'preprocessing')

def create_validation_operation_wrapper(ui_components: Dict[str, Any]) -> Callable:
    """🔍 Create validation operation wrapper"""
    return create_operation_wrapper(ui_components, 'validation')

def create_cleanup_operation_wrapper(ui_components: Dict[str, Any]) -> Callable:
    """🧹 Create cleanup operation wrapper"""
    return create_operation_wrapper(ui_components, 'cleanup')

# Backward compatibility functions
def setup_config_handlers_fixed(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """🔧 Backward compatibility untuk config handlers"""
    _setup_enhanced_config_integration(ui_components, config)

def execute_preprocessing_with_backend(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🚀 Backward compatibility untuk preprocessing execution"""
    return _execute_coordinated_operation(ui_components, 'preprocessing', config)

def execute_validation_with_backend(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🔍 Backward compatibility untuk validation execution"""
    return _execute_coordinated_operation(ui_components, 'validation', config)

def execute_cleanup_with_backend(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """🧹 Backward compatibility untuk cleanup execution"""
    return _execute_coordinated_operation(ui_components, 'cleanup', config)

# One-liner utilities untuk convenience
setup_handlers = lambda ui_components, config, env=None: setup_preprocessing_handlers(ui_components, config, env)
get_status = lambda ui_components: get_operation_status(ui_components)
is_ready = lambda ui_components: get_operation_status(ui_components).get('ui_ready', False)
is_processing = lambda ui_components: get_operation_status(ui_components).get('operation_active', False)
register_backend = lambda ui_components, service: register_backend_service_to_ui(ui_components, service)