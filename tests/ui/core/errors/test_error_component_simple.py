"""
Simple test suite for the ErrorComponent class.
This version focuses on testing the component's behavior through its public API.
"""
import pytest
from unittest.mock import MagicMock, patch

# Define the module path for patching
MODULE_PATH = 'smartcash.ui.core.errors.error_component'

@pytest.fixture
def mock_ipywidgets():
    """Fixture to mock ipywidgets."""
    with patch('ipywidgets.VBox') as mock_vbox, \
         patch('ipywidgets.HTML') as mock_html, \
         patch('ipywidgets.Button') as mock_button, \
         patch('ipywidgets.Output') as mock_output:
        
        # Configure the mocks
        mock_vbox.return_value = MagicMock()
        mock_html.return_value = MagicMock()
        mock_button.return_value = MagicMock()
        mock_output.return_value = MagicMock()
        
        yield {
            'VBox': mock_vbox,
            'HTML': mock_html,
            'Button': mock_button,
            'Output': mock_output
        }

@pytest.fixture
def mock_display():
    """Fixture to mock IPython display."""
    with patch('IPython.display.HTML') as mock_html, \
         patch('IPython.display.display') as mock_display:
        
        mock_html.return_value = MagicMock()
        
        yield {
            'HTML': mock_html,
            'display': mock_display
        }

def test_error_component_initialization(mock_ipywidgets, mock_display):
    """Test that the error component initializes correctly."""
    # Import inside the test to ensure proper patching
    from smartcash.ui.core.errors.error_component import ErrorComponent
    
    # Initialize the component
    component = ErrorComponent(title="Test Error", width="500px")
    
    # Verify initialization
    assert component.title == "Test Error"
    assert component.width == "500px"
    assert component._components == {}

def test_get_styles():
    """Test that get_styles returns the expected structure."""
    # Import inside the test to ensure proper patching
    from smartcash.ui.core.errors.error_component import ErrorComponent
    
    # Get the styles
    styles = ErrorComponent._get_styles()
    
    # Verify the structure
    assert isinstance(styles, dict)
    assert set(styles.keys()) == {"error", "warning", "info", "success"}
    
    # Check each style has required keys
    required_keys = {"bg", "border", "color", "hover_bg", "dark_mode", "aria_label", "icon"}
    for style_name, style in styles.items():
        assert set(style.keys()) == required_keys, f"Missing keys in {style_name} style"
        assert isinstance(style["dark_mode"], dict), "dark_mode should be a dictionary"

def test_create_method(mock_ipywidgets, mock_display):
    """Test the create method with different parameters."""
    # Import inside the test to ensure proper patching
    from smartcash.ui.core.errors.error_component import ErrorComponent
    
    # Initialize the component
    component = ErrorComponent()
    
    # Test creating without traceback
    result = component.create(
        error_message="Test error",
        error_type="error",
        show_traceback=False
    )
    
    # Verify the result structure
    assert isinstance(result, dict)
    assert "widget" in result
    assert "container" in result
    assert "error_widget" in result
    assert result["traceback_widget"] is None
    
    # Test creating with traceback
    result = component.create(
        error_message="Test error with traceback",
        error_type="error",
        show_traceback=True,
        traceback="Traceback..."
    )
    assert result["traceback_widget"] is not None
