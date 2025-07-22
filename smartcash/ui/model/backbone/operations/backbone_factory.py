"""
File: smartcash/ui/model/backbone/operations/backbone_factory.py
Description: Factory for creating backbone operation handlers.
"""

from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

from .backbone_validate_operation import BackboneValidateOperationHandler
from .backbone_build_operation import BackboneBuildOperationHandler

if TYPE_CHECKING:
    from smartcash.ui.model.backbone.backbone_uimodule import BackboneUIModule


class BackboneOperationFactory:
    """
    Factory class for creating backbone operation handlers.
    """
    
    @staticmethod
    def create_validate_handler(
        ui_module: 'BackboneUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> BackboneValidateOperationHandler:
        """
        Create a validation operation handler.
        
        Args:
            ui_module: The backbone UI module instance
            config: Configuration dictionary
            callbacks: Optional callbacks dict
            
        Returns:
            BackboneValidateOperationHandler instance
        """
        return BackboneValidateOperationHandler(ui_module, config, callbacks)
    
    @staticmethod
    def create_build_handler(
        ui_module: 'BackboneUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> BackboneBuildOperationHandler:
        """
        Create a build operation handler.
        
        Args:
            ui_module: The backbone UI module instance
            config: Configuration dictionary
            callbacks: Optional callbacks dict
            
        Returns:
            BackboneBuildOperationHandler instance
        """
        return BackboneBuildOperationHandler(ui_module, config, callbacks)
    
    @staticmethod
    def get_available_operations() -> Dict[str, str]:
        """
        Get a dictionary of available operations and their descriptions.
        
        Returns:
            Dict mapping operation names to descriptions
        """
        return {
            'validate': 'Validate backbone configuration and model compatibility',
            'build': 'Build and initialize the backbone model'
        }