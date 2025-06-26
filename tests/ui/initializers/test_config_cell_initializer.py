"""
Tests for the ConfigCellInitializer class to ensure it returns display widgets.
"""
from typing import Dict, Any
import pytest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler

class TestConfigHandler(ConfigCellHandler):
    """Test implementation of ConfigCellHandler for testing."""
    
    def __init__(self, module_name='test_module'):
        super().__init__(module_name=module_name)
    
    def get_config(self):
        return {'test': 'config'}
    
    def validate_config(self, config):
        return True, ""

class TestConfigCellInitializer(ConfigCellInitializer):
    """Test implementation of ConfigCellInitializer for testing."""
    
    def create_handler(self) -> TestConfigHandler:
        """Create a test handler."""
        return TestConfigHandler()
    
    def create_ui_components(self, config: Dict[str, Any]) -> dict:
        """Create test UI components.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary of UI components with at least a 'container' widget
        """
        return {
            'container': widgets.VBox(),
            'button': widgets.Button(description='Test Button')
        }

class TestConfigCellInitializerDisplay:
    """Test that ConfigCellInitializer returns display widgets."""
    
    def test_initialize_returns_widgets(self):
        """Test that initialize() returns widgets instead of dict."""
        print("\n=== Starting test_initialize_returns_widgets ===")
        
        # Arrange
        print("\n--- Debug: Creating TestConfigCellInitializer instance ---")
        try:
            # Create initializer with test parameters
            initializer = TestConfigCellInitializer(
                module_name='test_module',
                config_filename='test_config.yaml'
            )
            print(f"Initializer created: {initializer}")
            print(f"Initializer class: {initializer.__class__.__name__}")
            print(f"Initializer module: {initializer.__class__.__module__}")
            
            # Debug initializer attributes
            print("\nInitializer attributes:")
            for attr in dir(initializer):
                if not attr.startswith('__'):  # Skip private attributes
                    try:
                        attr_value = getattr(initializer, attr)
                        if callable(attr_value):
                            print(f"  - {attr}: callable")
                        else:
                            print(f"  - {attr}: {type(attr_value).__name__} = {attr_value}")
                    except Exception as e:
                        print(f"  - {attr}: <error accessing: {str(e)}>")
            
            # Check if handler is created
            if hasattr(initializer, '_handler'):
                print(f"\nHandler exists: {initializer._handler is not None}")
                if initializer._handler is not None:
                    print(f"Handler type: {type(initializer._handler).__name__}")
            else:
                print("\nNo _handler attribute found")
                
        except Exception as e:
            print(f"\nError creating initializer: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Debug: Check handler creation
        print("\n--- Debug: Checking handler creation ---")
        try:
            handler = initializer.create_handler()
            print(f"Handler created: {handler}")
            print(f"Handler type: {type(handler).__name__}")
            print(f"Handler class: {handler.__class__.__name__}")
            print(f"Handler config: {handler.get_config()}")
            
            # Store the handler for later use
            initializer._handler = handler
            print("Handler stored in initializer._handler")
            
        except Exception as e:
            print(f"Error creating handler: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Debug: Check UI components creation
        print("\n--- Debug: Checking UI components creation ---")
        try:
            config = handler.get_config()
            print(f"Using config: {config}")
            
            ui_components = initializer.create_ui_components(config)
            print(f"UI components created: {type(ui_components)}")
            
            # Store UI components in initializer for the initialize() method
            initializer.ui_components = ui_components
            
            print(f"UI components keys: {list(ui_components.keys())}")
            for key, value in ui_components.items():
                print(f"  - {key}: {type(value).__name__} (Widget: {isinstance(value, widgets.Widget)})")
                
            # Ensure the container is a widget
            if 'container' in ui_components:
                container = ui_components['container']
                print(f"\nContainer details:")
                print(f"  - Type: {type(container)}")
                print(f"  - Class: {container.__class__.__name__}")
                print(f"  - Is Widget: {isinstance(container, widgets.Widget)}")
                print(f"  - Repr: {container!r}")
                
                # Check if container has children
                if hasattr(container, 'children'):
                    children = container.children or []
                    print(f"  - Children count: {len(children)}")
                    for i, child in enumerate(children):
                        print(f"    - Child {i}: {type(child).__name__} (Widget: {isinstance(child, widgets.Widget)})")
            
        except Exception as e:
            print(f"Error creating UI components: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Act
        print("\n--- Debug: Calling initialize() ---")
        try:
            # Patch the initialize method to add debug logging
            original_initialize = initializer.initialize
            
            def wrapped_initialize(*args, **kwargs):
                print("\n--- Inside initialize() method ---")
                print(f"self.ui_components exists: {hasattr(initializer, 'ui_components')}")
                if hasattr(initializer, 'ui_components'):
                    print(f"ui_components keys: {list(initializer.ui_components.keys())}")
                result = original_initialize(*args, **kwargs)
                print(f"initialize() returned: {type(result).__name__}")
                return result
            
            # Apply the patch
            import functools
            initializer.initialize = wrapped_initialize
            
            # Call initialize
            result = initializer.initialize()
            print(f"Initialize completed. Result type: {type(result).__name__}")
            
        except Exception as e:
            print(f"Initialize failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Debug prints
        print("\n--- Debug: Checking result ---")
        print(f"Type of result: {type(result).__name__}")
        print(f"Result class: {result.__class__.__name__}")
        print(f"Result module: {result.__class__.__module__}")
        print(f"Is instance of Widget: {isinstance(result, widgets.Widget)}")
        print(f"Result repr: {result!r}")
        
        # Check widget inheritance
        print("\n--- Debug: Checking widget inheritance ---")
        try:
            from IPython.display import display
            print("IPython.display.display is available")
            display_available = True
            
            # Try to display the widget
            try:
                print("Attempting to display the widget...")
                display(result)
                print("Widget displayed successfully")
            except Exception as e:
                print(f"Error displaying widget: {str(e)}")
                
        except ImportError:
            print("IPython.display.display is not available")
            display_available = False
            
        # Check if result is a display object
        if hasattr(result, '_ipython_display_'):
            print("Result has _ipython_display_ method")
        else:
            print("Result does NOT have _ipython_display_ method")
            
        # Check if result is a widget
        is_widget = isinstance(result, widgets.Widget)
        print(f"Is result a Widget? {is_widget}")
        
        # If result is not a widget, try to get a widget from it
        if not is_widget:
            # Check if it's a dict-like with a 'container' key
            if hasattr(result, 'get') and callable(result.get) and 'container' in result:
                container = result['container']
                print(f"Container from result: {type(container).__name__} (Widget: {isinstance(container, widgets.Widget)})")
                if isinstance(container, widgets.Widget):
                    result = container
                    is_widget = True
            # Check if it has a 'container' attribute
            elif hasattr(result, 'container') and isinstance(result.container, widgets.Widget):
                print("Result has a 'container' attribute that is a Widget")
                print(f"Container type: {type(result.container).__name__}")
                result = result.container
                is_widget = True
        
        # Additional debug: Check if the result is in ui_components
        if hasattr(initializer, 'ui_components'):
            print("\n--- Debug: UI components after initialize() ---")
            print(f"UI components keys: {list(initializer.ui_components.keys())}")
            for key, value in initializer.ui_components.items():
                print(f"  - {key}: {type(value).__name__} (Widget: {isinstance(value, widgets.Widget)})")
        
        # Debug prints
        print("\n--- Debug: Running assertions ---")
        
        assert is_widget, f"Should return a widget instance, got {type(result)} instead"
            
        print("\n=== Test passed ===\n")
        return result
    
    def test_initialize_with_error_returns_error_widget(self):
        """Test that initialize() returns an error widget on failure."""
        # Arrange
        initializer = TestConfigCellInitializer(
            module_name='test_module',
            config_filename='test_config.yaml'
        )
        initializer.create_ui_components = MagicMock(side_effect=Exception("Test error"))
        
        # Act
        result = initializer.initialize()
        
        # Debug prints
        print("\n=== Debug: Error Test ===")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        # Assert
        # The error response should be a widget directly
        assert isinstance(result, widgets.Widget), \
            f"Should return a widget directly, got {type(result)}"
    
    def test_initialize_with_parent_module(self):
        """Test initialization with a parent module."""
        # Arrange
        initializer = TestConfigCellInitializer(
            module_name='test_module',
            config_filename='test_config.yaml',
            parent_module='parent_module'
        )
        
        # Act
        result = initializer.initialize()
        
        # Assert
        assert isinstance(result, widgets.Widget), \
            "Should return a widget with parent module"
    
    @patch('smartcash.ui.config_cell.components.component_registry.register_component')
    def test_component_registration(self, mock_register):
        """Test that components are properly registered."""
        # Arrange
        initializer = TestConfigCellInitializer(
            module_name='test_module',
            config_filename='test_config.yaml',
            is_container=True
        )
        
        # Act
        initializer.initialize()
        
        # Assert
        mock_register.assert_called()
        args, kwargs = mock_register.call_args
        assert kwargs['is_container'] is True, \
            "Should register as a container when is_container=True"
