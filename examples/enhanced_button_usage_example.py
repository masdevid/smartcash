"""
Example: Enhanced Button State Management Usage

This example demonstrates how any UI module can use the enhanced button
state management functionality from the button handler mixin.
"""

from typing import Dict, Any
from smartcash.ui.core.base_ui_module import BaseUIModule


class ExampleUIModule(BaseUIModule):
    """
    Example UI module demonstrating enhanced button state management.
    
    This shows how any module can leverage the button handler mixin
    for consistent, dependency-based button state management.
    """
    
    def __init__(self):
        super().__init__(module_name='example', parent_module='demo')
        self._data_loaded = False
        self._model_trained = False
        self._results_available = False
    
    def _setup_button_dependencies(self) -> None:
        """
        Setup button dependencies using the enhanced mixin functionality.
        
        This method shows how to define when buttons should be enabled/disabled
        based on module state and conditions.
        """
        # Load data button is always available (no dependencies)
        
        # Train button depends on having data loaded
        self.set_button_dependency('train', self._check_train_dependency)
        
        # Evaluate button depends on having a trained model
        self.set_button_dependency('evaluate', self._check_evaluate_dependency)
        
        # Export button depends on having results
        self.set_button_dependency('export', self._check_export_dependency)
        
        # Reset button depends on having any state to reset
        self.set_button_dependency('reset', self._check_reset_dependency)
    
    def _check_train_dependency(self) -> bool:
        """Check if training can be started."""
        return self._data_loaded
    
    def _check_evaluate_dependency(self) -> bool:
        """Check if evaluation can be started."""
        return self._model_trained
    
    def _check_export_dependency(self) -> bool:
        """Check if results can be exported."""
        return self._results_available
    
    def _check_reset_dependency(self) -> bool:
        """Check if there's state to reset."""
        return self._data_loaded or self._model_trained or self._results_available
    
    def _operation_load_data(self, button=None) -> Dict[str, Any]:
        """Load data operation example."""
        try:
            self.log("🔄 Loading data...", 'info')
            
            # Simulate data loading
            self._data_loaded = True
            
            # Update all button states based on new conditions
            self._update_button_states()
            
            self.log("✅ Data loaded successfully", 'success')
            return {'success': True, 'message': 'Data loaded'}
            
        except Exception as e:
            self.log(f"❌ Failed to load data: {e}", 'error')
            return {'success': False, 'message': str(e)}
    
    def _operation_train_model(self, button=None) -> Dict[str, Any]:
        """Train model operation example."""
        try:
            self.log("🔄 Training model...", 'info')
            
            # Simulate model training
            self._model_trained = True
            
            # Update button states
            self._update_button_states()
            
            self.log("✅ Model trained successfully", 'success')
            return {'success': True, 'message': 'Model trained'}
            
        except Exception as e:
            self.log(f"❌ Failed to train model: {e}", 'error')
            return {'success': False, 'message': str(e)}
    
    def _operation_evaluate_model(self, button=None) -> Dict[str, Any]:
        """Evaluate model operation example."""
        try:
            self.log("🔄 Evaluating model...", 'info')
            
            # Simulate model evaluation
            self._results_available = True
            
            # Update button states
            self._update_button_states()
            
            self.log("✅ Model evaluated successfully", 'success')
            return {'success': True, 'message': 'Model evaluated'}
            
        except Exception as e:
            self.log(f"❌ Failed to evaluate model: {e}", 'error')
            return {'success': False, 'message': str(e)}
    
    def _operation_export_results(self, button=None) -> Dict[str, Any]:
        """Export results operation example."""
        try:
            self.log("🔄 Exporting results...", 'info')
            
            # Simulate results export
            # Results remain available after export
            
            self.log("✅ Results exported successfully", 'success')
            return {'success': True, 'message': 'Results exported'}
            
        except Exception as e:
            self.log(f"❌ Failed to export results: {e}", 'error')
            return {'success': False, 'message': str(e)}
    
    def _operation_reset(self, button=None) -> Dict[str, Any]:
        """Reset operation example."""
        try:
            self.log("🔄 Resetting module state...", 'info')
            
            # Reset all state
            self._data_loaded = False
            self._model_trained = False
            self._results_available = False
            
            # Update button states
            self._update_button_states()
            
            self.log("✅ Module reset successfully", 'success')
            return {'success': True, 'message': 'Module reset'}
            
        except Exception as e:
            self.log(f"❌ Failed to reset: {e}", 'error')
            return {'success': False, 'message': str(e)}
    
    def _update_button_states(self) -> None:
        """
        Update all button states based on current module conditions.
        
        This demonstrates the enhanced button state management in action.
        """
        try:
            # Define button conditions based on current state
            button_conditions = {
                'load_data': True,  # Always available
                'train': self._data_loaded,
                'evaluate': self._model_trained, 
                'export': self._results_available,
                'reset': (self._data_loaded or self._model_trained or self._results_available)
            }
            
            # Define reasons for disabled buttons
            button_reasons = {
                'train': "Load data first" if not self._data_loaded else None,
                'evaluate': "Train model first" if not self._model_trained else None,
                'export': "Evaluate model first" if not self._results_available else None,
                'reset': "No state to reset" if not (self._data_loaded or self._model_trained or self._results_available) else None
            }
            
            # Update all button states in one call
            self.update_button_states_based_on_condition(button_conditions, button_reasons)
            
            self.log_debug("Button states updated based on current conditions")
            
        except Exception as e:
            self.log_error(f"Failed to update button states: {e}")
    
    def get_module_status(self) -> Dict[str, Any]:
        """
        Get current module status including button states.
        
        Returns:
            Dictionary with module status and button information
        """
        return {
            'data_loaded': self._data_loaded,
            'model_trained': self._model_trained,
            'results_available': self._results_available,
            'button_states': self.get_button_states(),
            'disabled_buttons': {
                button_id: self.get_button_disable_reason(button_id)
                for button_id in ['load_data', 'train', 'evaluate', 'export', 'reset']
                if not self.is_button_enabled(button_id)
            }
        }
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get module-specific button handlers."""
        handlers = super()._get_module_button_handlers()
        
        # Add example-specific handlers
        example_handlers = {
            'load_data': self._operation_load_data,
            'train': self._operation_train_model,
            'evaluate': self._operation_evaluate_model,
            'export': self._operation_export_results,
            'reset': self._operation_reset,
        }
        
        handlers.update(example_handlers)
        return handlers
    
    def initialize(self, config=None, **kwargs) -> bool:
        """Initialize the example module."""
        success = super().initialize(config, **kwargs)
        
        if success:
            # Setup button dependencies after initialization
            self._setup_button_dependencies()
            
            # Set initial button states
            self._update_button_states()
            
            self.log("🎯 Example module initialized with enhanced button state management", 'info')
        
        return success


# Usage example
if __name__ == "__main__":
    # Create and initialize the example module
    example_module = ExampleUIModule()
    example_module.initialize()
    
    # Check initial status
    status = example_module.get_module_status()
    print("Initial status:", status)
    
    # Simulate button clicks and state changes
    example_module._operation_load_data()
    example_module._operation_train_model()
    example_module._operation_evaluate_model()
    
    # Check final status
    final_status = example_module.get_module_status()
    print("Final status:", final_status)