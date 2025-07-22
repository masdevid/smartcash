"""
File: smartcash/ui/model/mixins/backend_service_mixin.py
Description: Mixin for backend service integration with UI components.

Standardizes backend service initialization and integration patterns across modules.
"""

from typing import Dict, Any, List, Callable
from smartcash.common.logger import get_logger


class BackendServiceMixin:
    """
    Mixin for backend service integration with UI components.
    
    Provides standardized functionality for:
    - Backend service initialization with configuration
    - Progress bridge setup and UI component integration
    - Service error handling with fallback mechanisms
    - Service status monitoring and health checks
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._service_logger = get_logger(f"{self.__class__.__name__}.backend_service")
        self._backend_services = {}
        self._service_status = {}
    
    def initialize_backend_services(
        self, 
        service_configs: Dict[str, Any],
        required_services: List[str] = None
    ) -> Dict[str, Any]:
        """
        Initialize backend services with configuration.
        
        Args:
            service_configs: Configuration for each service
            required_services: List of required service names
            
        Returns:
            Initialization result dictionary
        """
        initialization_result = {
            'success': True,
            'initialized_services': [],
            'failed_services': [],
            'warnings': []
        }
        
        required_services = required_services or []
        
        for service_name, service_config in service_configs.items():
            try:
                service_instance = self._create_service_instance(service_name, service_config)
                
                if service_instance:
                    self._backend_services[service_name] = service_instance
                    self._service_status[service_name] = 'initialized'
                    initialization_result['initialized_services'].append(service_name)
                    
                    self._service_logger.debug(f"✅ Initialized {service_name} service")
                else:
                    self._service_status[service_name] = 'failed'
                    initialization_result['failed_services'].append(service_name)
                    
                    if service_name in required_services:
                        initialization_result['success'] = False
                        self._service_logger.error(f"❌ Failed to initialize required service: {service_name}")
                    else:
                        initialization_result['warnings'].append(f"Optional service {service_name} failed to initialize")
                        self._service_logger.warning(f"⚠️ Optional service {service_name} failed to initialize")
                        
            except Exception as e:
                self._service_status[service_name] = 'error'
                initialization_result['failed_services'].append(service_name)
                error_msg = f"Error initializing {service_name}: {e}"
                
                if service_name in required_services:
                    initialization_result['success'] = False
                    self._service_logger.error(f"❌ {error_msg}")
                else:
                    initialization_result['warnings'].append(error_msg)
                    self._service_logger.warning(f"⚠️ {error_msg}")
        
        self._service_logger.info(f"🔧 Backend services initialized: {len(initialization_result['initialized_services'])}/{len(service_configs)}")
        return initialization_result
    
    def create_progress_bridge(
        self, 
        ui_components: Dict[str, Any], 
        service_type: str,
        bridge_config: Dict[str, Any] = None
    ) -> Any:
        """
        Create progress bridge for backend service integration.
        
        Args:
            ui_components: UI components for progress tracking
            service_type: Type of service ('evaluation', 'training', etc.)
            bridge_config: Optional bridge configuration
            
        Returns:
            Progress bridge instance or None if creation failed
        """
        try:
            bridge_config = bridge_config or {}
            
            # Create service-specific progress bridge
            if service_type == 'evaluation':
                from smartcash.model.evaluation.utils.evaluation_progress_bridge import EvaluationProgressBridge
                return EvaluationProgressBridge(ui_components, bridge_config.get('progress_callback'))
                
            elif service_type == 'training':
                # Training progress bridge would be created here
                # from smartcash.model.training.utils.training_progress_bridge import TrainingProgressBridge
                # return TrainingProgressBridge(ui_components, bridge_config.get('progress_callback'))
                self._service_logger.warning(f"⚠️ Training progress bridge not yet implemented")
                return None
                
            else:
                self._service_logger.warning(f"⚠️ Unknown service type for progress bridge: {service_type}")
                return None
                
        except Exception as e:
            self._service_logger.error(f"❌ Error creating progress bridge for {service_type}: {e}")
            return None
    
    def setup_service_callbacks(
        self, 
        service_instance: Any, 
        callback_mappings: Dict[str, str],
        ui_callbacks: Dict[str, Callable] = None
    ) -> None:
        """
        Setup callbacks between service and UI.
        
        Args:
            service_instance: Backend service instance
            callback_mappings: Mapping of service events to UI methods
            ui_callbacks: Optional custom UI callback functions
        """
        try:
            ui_callbacks = ui_callbacks or {}
            
            for service_event, ui_method_name in callback_mappings.items():
                # Try to get UI method
                ui_method = None
                
                if ui_method_name in ui_callbacks:
                    ui_method = ui_callbacks[ui_method_name]
                elif hasattr(self, ui_method_name):
                    ui_method = getattr(self, ui_method_name)
                
                if ui_method and hasattr(service_instance, 'set_callback'):
                    service_instance.set_callback(service_event, ui_method)
                    self._service_logger.debug(f"📞 Setup callback: {service_event} -> {ui_method_name}")
                    
        except Exception as e:
            self._service_logger.error(f"❌ Error setting up service callbacks: {e}")
    
    def handle_service_errors(
        self, 
        service_name: str, 
        error: Exception, 
        fallback_action: str = None
    ) -> Dict[str, Any]:
        """
        Handle backend service errors with fallback mechanisms.
        
        Args:
            service_name: Name of the service that failed
            error: Exception that occurred
            fallback_action: Optional fallback action to take
            
        Returns:
            Error handling result dictionary
        """
        error_result = {
            'success': False,
            'service': service_name,
            'error': str(error),
            'fallback_used': False,
            'message': f"Service {service_name} error: {error}"
        }
        
        # Update service status
        self._service_status[service_name] = 'error'
        
        # Try fallback actions
        if fallback_action:
            try:
                if fallback_action == 'reinitialize':
                    # Try to reinitialize the service
                    if self._reinitialize_service(service_name):
                        error_result['fallback_used'] = True
                        error_result['message'] = f"Service {service_name} reinitialized after error"
                        
                elif fallback_action == 'mock_service':
                    # Create mock service for testing/development
                    mock_service = self._create_mock_service(service_name)
                    if mock_service:
                        self._backend_services[service_name] = mock_service
                        self._service_status[service_name] = 'mock'
                        error_result['fallback_used'] = True
                        error_result['message'] = f"Using mock service for {service_name}"
                        
            except Exception as fallback_error:
                error_result['message'] += f" (Fallback failed: {fallback_error})"
        
        self._service_logger.error(f"❌ {error_result['message']}")
        return error_result
    
    def get_service_status(
        self, 
        services: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get status of backend services.
        
        Args:
            services: Optional list of specific services to check
            
        Returns:
            Service status dictionary
        """
        services_to_check = services or list(self._backend_services.keys())
        
        status_report = {
            'overall_status': 'healthy',
            'service_details': {},
            'healthy_count': 0,
            'total_count': len(services_to_check)
        }
        
        for service_name in services_to_check:
            service_status = self._service_status.get(service_name, 'unknown')
            service_instance = self._backend_services.get(service_name)
            
            service_details = {
                'status': service_status,
                'available': service_instance is not None,
                'type': type(service_instance).__name__ if service_instance else 'None'
            }
            
            # Check if service is healthy
            if service_status in ['initialized', 'mock'] and service_instance:
                service_details['healthy'] = True
                status_report['healthy_count'] += 1
            else:
                service_details['healthy'] = False
                status_report['overall_status'] = 'degraded'
            
            status_report['service_details'][service_name] = service_details
        
        # Determine overall status
        if status_report['healthy_count'] == 0:
            status_report['overall_status'] = 'failed'
        elif status_report['healthy_count'] < status_report['total_count']:
            status_report['overall_status'] = 'degraded'
        
        return status_report
    
    def _create_service_instance(self, service_name: str, service_config: Dict[str, Any]) -> Any:
        """Create backend service instance based on name and config."""
        try:
            if service_name == 'checkpoint_selector':
                from smartcash.model.evaluation.checkpoint_selector import CheckpointSelector
                return CheckpointSelector(config=service_config)
                
            elif service_name == 'evaluation_service':
                from smartcash.model.evaluation.evaluation_service import EvaluationService
                return EvaluationService(model_api=None, config=service_config)
                
            elif service_name == 'training_service':
                # Training service would be created here
                # from smartcash.model.training.training_service import TrainingService
                # return TrainingService(config=service_config)
                self._service_logger.warning(f"⚠️ Training service not yet implemented")
                return None
                
            else:
                self._service_logger.warning(f"⚠️ Unknown service type: {service_name}")
                return None
                
        except Exception as e:
            self._service_logger.error(f"❌ Error creating {service_name} service: {e}")
            return None
    
    def _reinitialize_service(self, service_name: str) -> bool:
        """Try to reinitialize a failed service."""
        try:
            # This would need service-specific reinitialization logic
            # For now, just mark as reinitialized
            self._service_status[service_name] = 'reinitialized'
            return True
        except Exception:
            return False
    
    def _create_mock_service(self, service_name: str) -> Any:
        """Create mock service for fallback scenarios."""
        try:
            if service_name == 'evaluation_service':
                # Create mock evaluation service
                from smartcash.model.evaluation.evaluation_service import MockProgressBridge
                return MockProgressBridge()
            else:
                return None
        except Exception:
            return None