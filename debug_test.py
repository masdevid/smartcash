"""Debug script for test_parent_child_relationship."""
import sys
import io
import pytest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer

class TestConfigCellInitializer(ConfigCellInitializer):
    """Test implementation of ConfigCellInitializer with required abstract methods."""
    
    def __init__(self, *args, **kwargs):
        # Initialize handler property
        self._handler = None
        # Extract container and content_area from kwargs if provided
        self.container = kwargs.pop('container', None)
        self.content_area = kwargs.pop('content_area', None)
        # Initialize children list before parent init to avoid attribute errors
        self.children = []
        super().__init__(*args, **kwargs)
    
    @property
    def handler(self):
        """Get the handler instance."""
        return self._handler
        
    @handler.setter
    def handler(self, value):
        """Set the handler instance."""
        self._handler = value
    
    def create_handler(self):
        """Create a mock handler for testing."""
        return MagicMock()
    
    def create_ui_components(self, config):
        """Create mock UI components for testing."""
        # Create a widget if container/content_area is not provided
        if not hasattr(self, 'container') or not self.container:
            self.container = widgets.VBox()
        if not hasattr(self, 'content_area') or not self.content_area:
            self.content_area = widgets.VBox()
            
        return {
            'container': self.container,
            'content_area': self.content_area
        }
    
    def setup_handlers(self):
        """Set up mock handlers for testing."""
        pass
        
    def add_child(self, child):
        """Add a child component."""
        print(f"\n--- Adding child: {child.component_id} to parent: {self.component_id} ---")
        self.children.append(child)
        
        # Initialize the child if not already initialized
        if not hasattr(child, '_is_initialized') or not child._is_initialized:
            print(f"Initializing child: {child.component_id}")
            try:
                child.initialize()
            except Exception as e:
                print(f"Error initializing child {child.component_id}: {str(e)}")
                # Create a simple error display
                error_display = widgets.HTML(f"<div style='color: red; padding: 10px;'>Error: {str(e)}</div>")
                self.content_area.children = (error_display,) + self.content_area.children
                return
            
        # Register the child component
        if hasattr(child, '_register_component'):
            print(f"Registering child component: {child.component_id}")
            try:
                child._register_component()
            except Exception as e:
                print(f"Error registering child {child.component_id}: {str(e)}")
                # Create a simple error display
                error_display = widgets.HTML(f"<div style='color: red; padding: 10px;'>Registration Error: {str(e)}</div>")
                self.content_area.children = (error_display,) + self.content_area.children

# Redirect stdout to capture all output
original_stdout = sys.stdout
sys.stdout = io.StringIO()

try:
    print("=== Starting debug script ===")
    
    # Import the test module and patch ConfigCellInitializer with our test implementation
    sys.path.insert(0, "/Users/masdevid/Projects/smartcash")
    
    # Patch the original ConfigCellInitializer with our test implementation
    with patch('smartcash.ui.initializers.config_cell_initializer.ConfigCellInitializer', TestConfigCellInitializer):
        from tests.ui.dataset.split.test_split_init import test_parent_child_relationship
        
        # Run the test directly
        print("\n=== Running test_parent_child_relationship ===")
        test_parent_child_relationship()
    print("\n=== Test completed successfully ===")
    
except Exception as e:
    print(f"\n=== Error running test: {str(e)} ===")
    import traceback
    traceback.print_exc()
    
finally:
    # Restore stdout and print captured output
    captured_output = sys.stdout.getvalue()
    sys.stdout = original_stdout
    
    # Write to a file
    with open("debug_output.log", "w") as f:
        f.write(captured_output)
    
    print("Debug output written to debug_output.log")
