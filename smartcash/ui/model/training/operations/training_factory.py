"""
File: smartcash/ui/model/training/operations/training_factory.py
Description: Factory for creating training operation handlers.
"""

from typing import Dict, Any, Optional, Callable, TYPE_CHECKING
from enum import Enum

# Import operation handlers
from .training_start_operation import TrainingStartOperationHandler
from .training_stop_operation import TrainingStopOperationHandler
from .training_resume_operation import TrainingResumeOperationHandler
from .training_validate_operation import TrainingValidateOperationHandler

if TYPE_CHECKING:
    from smartcash.ui.model.training.training_uimodule import TrainingUIModule


class TrainingOperationType(Enum):
    """Enumeration of available training operations."""
    START = "start"
    STOP = "stop"
    RESUME = "resume"
    VALIDATE = "validate"


class TrainingOperationFactory:
    """
    Factory class for creating training operation handlers.
    
    Features:
    - 🏭 Centralized operation handler creation
    - 🔧 Consistent handler initialization with dependencies
    - 📋 Operation type validation and management
    - 🛠️ Easy extension for new training operations
    """
    
    # Map operation types to their handler classes
    _operation_handlers = {
        TrainingOperationType.START: TrainingStartOperationHandler,
        TrainingOperationType.STOP: TrainingStopOperationHandler,
        TrainingOperationType.RESUME: TrainingResumeOperationHandler,
        TrainingOperationType.VALIDATE: TrainingValidateOperationHandler,
    }
    
    @classmethod
    def create_operation_handler(
        cls, 
        operation_type: str,
        ui_module: 'TrainingUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> Any:
        """
        Create an operation handler for the specified operation type.
        
        Args:
            operation_type: The type of operation to create handler for
            ui_module: Reference to the TrainingUIModule instance
            config: Configuration dictionary for the operation
            callbacks: Optional callbacks for operation events
            
        Returns:
            Operation handler instance
            
        Raises:
            ValueError: If operation_type is not supported
        """
        try:
            # Convert string to enum if needed
            if isinstance(operation_type, str):
                operation_enum = TrainingOperationType(operation_type.lower())
            else:
                operation_enum = operation_type
            
            # Get handler class
            handler_class = cls._operation_handlers.get(operation_enum)
            if not handler_class:
                raise ValueError(f"Unsupported operation type: {operation_type}")
            
            # Create and return handler instance
            handler = handler_class(
                ui_module=ui_module,
                config=config,
                callbacks=callbacks or {}
            )
            
            return handler
            
        except ValueError as e:
            # Re-raise ValueError for unsupported operations
            raise e
        except Exception as e:
            raise ValueError(f"Failed to create operation handler for {operation_type}: {str(e)}")
    
    @classmethod
    def get_available_operations(cls) -> Dict[str, str]:
        """
        Get list of available operation types with descriptions.
        
        Returns:
            Dictionary mapping operation types to descriptions
        """
        return {
            TrainingOperationType.START.value: "Start training process with model selection and live charts",
            TrainingOperationType.STOP.value: "Stop current training process safely with checkpoint saving",
            TrainingOperationType.RESUME.value: "Resume training from previous checkpoint",
            TrainingOperationType.VALIDATE.value: "Validate trained model performance on test dataset"
        }
    
    @classmethod
    def validate_operation_type(cls, operation_type: str) -> bool:
        """
        Validate if operation type is supported.
        
        Args:
            operation_type: Operation type to validate
            
        Returns:
            True if operation type is supported, False otherwise
        """
        try:
            TrainingOperationType(operation_type.lower())
            return True
        except ValueError:
            return False
    
    @classmethod
    def get_operation_requirements(cls, operation_type: str) -> Dict[str, Any]:
        """
        Get requirements for a specific operation type.
        
        Args:
            operation_type: Operation type to get requirements for
            
        Returns:
            Dictionary with operation requirements
        """
        requirements = {
            TrainingOperationType.START.value: {
                "required_config": ["training", "model_selection"],
                "optional_config": ["data", "monitoring", "charts", "output"],
                "prerequisites": ["backbone_configuration"],
                "description": "Requires backbone configuration and training parameters"
            },
            TrainingOperationType.STOP.value: {
                "required_config": [],
                "optional_config": ["output"],
                "prerequisites": ["active_training"],
                "description": "Requires active training process"
            },
            TrainingOperationType.RESUME.value: {
                "required_config": ["model_selection"],
                "optional_config": ["training", "output"],
                "prerequisites": ["checkpoint_file"],
                "description": "Requires valid checkpoint file"
            },
            TrainingOperationType.VALIDATE.value: {
                "required_config": ["model_selection"],
                "optional_config": ["validation", "output"],
                "prerequisites": ["trained_model"],
                "description": "Requires trained model or checkpoint"
            }
        }
        
        return requirements.get(operation_type, {})
    
    @classmethod
    def create_operation_with_validation(
        cls,
        operation_type: str,
        ui_module: 'TrainingUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> tuple[Any, Dict[str, Any]]:
        """
        Create operation handler with configuration validation.
        
        Args:
            operation_type: The type of operation to create handler for
            ui_module: Reference to the TrainingUIModule instance
            config: Configuration dictionary for the operation
            callbacks: Optional callbacks for operation events
            
        Returns:
            Tuple of (operation_handler, validation_result)
        """
        # Validate operation type
        if not cls.validate_operation_type(operation_type):
            return None, {
                'success': False,
                'message': f'Invalid operation type: {operation_type}',
                'available_operations': list(cls.get_available_operations().keys())
            }
        
        # Get operation requirements
        requirements = cls.get_operation_requirements(operation_type)
        
        # Validate required configuration sections
        missing_config = []
        for required_section in requirements.get('required_config', []):
            if required_section not in config:
                missing_config.append(required_section)
        
        if missing_config:
            return None, {
                'success': False,
                'message': f'Missing required configuration sections: {missing_config}',
                'requirements': requirements
            }
        
        # Validate prerequisites
        validation_result = cls._validate_prerequisites(operation_type, config, ui_module)
        if not validation_result['success']:
            return None, validation_result
        
        # Create operation handler
        try:
            handler = cls.create_operation_handler(operation_type, ui_module, config, callbacks)
            return handler, {'success': True, 'message': 'Operation handler created successfully'}
        except Exception as e:
            return None, {
                'success': False,
                'message': f'Failed to create operation handler: {str(e)}'
            }
    
    @classmethod
    def _validate_prerequisites(
        cls,
        operation_type: str,
        config: Dict[str, Any],
        ui_module: 'TrainingUIModule'
    ) -> Dict[str, Any]:
        """
        Validate operation-specific prerequisites.
        
        Args:
            operation_type: Operation type to validate
            config: Configuration dictionary
            ui_module: UI module instance
            
        Returns:
            Validation result dictionary
        """
        try:
            if operation_type == TrainingOperationType.START.value:
                # Validate backbone configuration is available
                model_selection = config.get('model_selection', {})
                if model_selection.get('source') == 'backbone':
                    # Check if backbone configuration is available
                    if not model_selection.get('backbone_type'):
                        return {
                            'success': False,
                            'message': 'Backbone configuration required for training start'
                        }
                
            elif operation_type == TrainingOperationType.STOP.value:
                # Check if training is actually running
                # This would check actual training state in real implementation
                pass
                
            elif operation_type == TrainingOperationType.RESUME.value:
                # Validate checkpoint file exists
                checkpoint_path = config.get('model_selection', {}).get('checkpoint_path', '')
                if not checkpoint_path:
                    return {
                        'success': False,
                        'message': 'Checkpoint path required for resume operation'
                    }
                
                import os
                if not os.path.exists(checkpoint_path):
                    return {
                        'success': False,
                        'message': f'Checkpoint file not found: {checkpoint_path}'
                    }
                
            elif operation_type == TrainingOperationType.VALIDATE.value:
                # Validate model is available for validation
                model_selection = config.get('model_selection', {})
                if not model_selection.get('backbone_type') and not model_selection.get('checkpoint_path'):
                    return {
                        'success': False,
                        'message': 'No model available for validation'
                    }
            
            return {'success': True, 'message': 'Prerequisites validation passed'}
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Prerequisites validation failed: {str(e)}'
            }
    
    @classmethod
    def get_operation_metadata(cls, operation_type: str) -> Dict[str, Any]:
        """
        Get metadata for a specific operation type.
        
        Args:
            operation_type: Operation type to get metadata for
            
        Returns:
            Dictionary with operation metadata
        """
        metadata = {
            TrainingOperationType.START.value: {
                "display_name": "Start Training",
                "icon": "🚀",
                "color": "success",
                "estimated_duration": "varies",
                "complexity": "high",
                "resource_intensive": True
            },
            TrainingOperationType.STOP.value: {
                "display_name": "Stop Training",
                "icon": "🛑",
                "color": "warning",
                "estimated_duration": "30 seconds",
                "complexity": "low",
                "resource_intensive": False
            },
            TrainingOperationType.RESUME.value: {
                "display_name": "Resume Training",
                "icon": "🔄",
                "color": "info",
                "estimated_duration": "varies",
                "complexity": "medium",
                "resource_intensive": True
            },
            TrainingOperationType.VALIDATE.value: {
                "display_name": "Validate Model",
                "icon": "🔍",
                "color": "primary",
                "estimated_duration": "5-10 minutes",
                "complexity": "medium",
                "resource_intensive": False
            }
        }
        
        return metadata.get(operation_type, {})


# Convenience functions for common operations
def create_start_training_handler(
    ui_module: 'TrainingUIModule',
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> TrainingStartOperationHandler:
    """Create a training start operation handler."""
    return TrainingOperationFactory.create_operation_handler(
        TrainingOperationType.START.value, ui_module, config, callbacks
    )


def create_stop_training_handler(
    ui_module: 'TrainingUIModule',
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> TrainingStopOperationHandler:
    """Create a training stop operation handler."""
    return TrainingOperationFactory.create_operation_handler(
        TrainingOperationType.STOP.value, ui_module, config, callbacks
    )


def create_resume_training_handler(
    ui_module: 'TrainingUIModule',
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> TrainingResumeOperationHandler:
    """Create a training resume operation handler."""
    return TrainingOperationFactory.create_operation_handler(
        TrainingOperationType.RESUME.value, ui_module, config, callbacks
    )


def create_validate_training_handler(
    ui_module: 'TrainingUIModule',
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> TrainingValidateOperationHandler:
    """Create a training validation operation handler."""
    return TrainingOperationFactory.create_operation_handler(
        TrainingOperationType.VALIDATE.value, ui_module, config, callbacks
    )