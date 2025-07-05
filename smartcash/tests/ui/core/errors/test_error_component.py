"""
Test suite for the enhanced ErrorComponent class.
"""
import pytest
import sys
import types
from unittest.mock import MagicMock, patch, ANY

# Create a test-specific environment with all necessary mocks
class MockErrorHandler:
    @staticmethod
    def handle_error(*args, **kwargs):
        return {"status": "error", "message": "Test error"}

class MockBaseHandler:
    pass

# Set up mock modules
mock_modules = {
    'smartcash.ui.core.shared.error_handler': types.ModuleType('error_handler'),
    'smartcash.ui.core.shared': types.ModuleType('shared'),
    'smartcash.ui.core.handlers': types.ModuleType('handlers'),
    'smartcash.ui.core.handlers.base_handler': types.ModuleType('base_handler'),
    'smartcash.ui.handlers': types.ModuleType('ui_handlers'),
    'smartcash.ui.core.handlers.config_handler': types.ModuleType('config_handler'),
    'smartcash.ui.core.shared.shared_config_manager': types.ModuleType('shared_config_manager')
}

# Populate the module attributes
mock_modules['smartcash.ui.core.shared.error_handler'].CoreErrorHandler = MockErrorHandler
mock_modules['smartcash.ui.core.shared.error_handler'].handle_error = MockErrorHandler.handle_error
mock_modules['smartcash.ui.core.handlers.base_handler'].BaseHandler = MockBaseHandler

# Add mock modules to sys.modules
for name, module in mock_modules.items():
    sys.modules[name] = module

# Mock ipywidgets
sys.modules['ipywidgets'] = MagicMock()
sys.modules['IPython.display'] = MagicMock()

# Now import the component under test
with patch.dict('sys.modules', sys.modules):
    try:
        from smartcash.ui.core.errors.error_component import ErrorComponent, create_error_component
        from IPython.display import HTML
    except ImportError as e:
        print(f"Import error: {e}")
        raise

class TestEnhancedErrorComponent:
    """Test suite for the enhanced ErrorComponent class."""
    
    @pytest.fixture
    def error_component(self):
        """Create an ErrorComponent instance for testing with default values."""
        return ErrorComponent(title="Test Error", width="500px")
    
    @pytest.fixture
    def sample_traceback(self):
        """Return a sample traceback for testing."""
        return """
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        ValueError: Test error message
        """.strip()
    
    def test_initialization(self, error_component):
        """Test that ErrorComponent initializes with provided values."""
        assert error_component.title == "Test Error"
        assert error_component.width == "500px"
        assert isinstance(error_component._components, dict)
        assert not error_component._components  # Should be empty initially
    
    def test_get_styles(self):
        """Test that get_styles returns the expected style dictionary with all required keys."""
        styles = ErrorComponent._get_styles()
        assert isinstance(styles, dict)
        assert set(styles.keys()) == {"error", "warning", "info", "success"}
        
        # Check each style has required keys
        required_keys = {"bg", "border", "color", "icon", "hover_bg", "dark_mode", "aria_label"}
        for style_name, style in styles.items():
            assert set(style.keys()) == required_keys, f"Missing keys in {style_name} style"
            assert isinstance(style["dark_mode"], dict), "dark_mode should be a dictionary"
    
    def test_create_without_traceback(self, error_component):
        """Test creating an error component without a traceback."""
        with patch.object(ErrorComponent, '_create_main_error_display') as mock_create_display, \
             patch('ipywidgets.VBox') as mock_vbox:
            
            # Setup mocks
            mock_display = MagicMock()
            mock_create_display.return_value = mock_display
            mock_vbox.return_value = MagicMock()
            
            # Call the method without traceback
            result = error_component.create(
                error_message="Test error message",
                error_type="error",
                show_traceback=False
            )
            
            # Assertions
            mock_create_display.assert_called_once_with(
                "Test error message",
                ANY,  # style dict
                False  # has_traceback
            )
            mock_vbox.assert_called_once()
            assert isinstance(result, dict)
            assert "widget" in result
            assert "container" in result
            assert "error_widget" in result
    
    def test_create_with_traceback(self, error_component, sample_traceback):
        """Test creating an error component with a traceback."""
        with patch.object(ErrorComponent, '_create_main_error_display') as mock_create_display, \
             patch.object(ErrorComponent, '_create_traceback_section') as mock_traceback, \
             patch('ipywidgets.VBox') as mock_vbox:
            
            # Setup mocks
            mock_display = MagicMock()
            mock_trace = MagicMock()
            mock_create_display.return_value = mock_display
            mock_traceback.return_value = mock_trace
            mock_vbox.return_value = MagicMock()
            
            # Call the method with traceback
            result = error_component.create(
                error_message="Test error message",
                traceback=sample_traceback,
                error_type="error",
                show_traceback=True
            )
            
            # Assertions
            mock_create_display.assert_called_once()
            mock_traceback.assert_called_once_with(sample_traceback)
            assert mock_vbox.call_count == 2  # One for content, one for container
            assert isinstance(result, dict)
            assert result["traceback_widget"] is not None
    
    def test_create_error_component_factory(self, sample_traceback):
        """Test the create_error_component factory function."""
        with patch('smartcash.ui.core.errors.error_component.ErrorComponent') as mock_component_class:
            # Setup mock
            mock_instance = MagicMock()
            mock_component_class.return_value = mock_instance
            mock_instance.create.return_value = {"widget": MagicMock()}
            
            # Call the factory function
            result = create_error_component(
                error_message="Test error",
                traceback=sample_traceback,
                title="Test Error",
                error_type="error",
                width="500px"
            )
            
            # Assertions
            mock_component_class.assert_called_once_with(title="Test Error", width="500px")
            mock_instance.create.assert_called_once_with(
                error_message="Test error",
                traceback=sample_traceback,
                error_type="error",
                show_traceback=True
            )
            assert result == mock_instance.create.return_value
    
    def test_dark_mode_styling(self, error_component):
        """Test that dark mode styles are properly included in the output."""
        with patch.object(ErrorComponent, '_create_toggle_button') as mock_toggle, \
             patch('ipywidgets.HTML') as mock_html:
            
            # Setup mocks
            mock_toggle.return_value = "<button>Toggle</button>"
            mock_html.return_value = MagicMock()
            
            # Call the method
            error_component._create_main_error_display(
                message="Test message",
                style={
                    "bg": "light-bg",
                    "border": "light-border",
                    "color": "light-color",
                    "icon": "🚨",
                    "dark_mode": {
                        "bg": "dark-bg",
                        "color": "dark-color"
                    }
                },
                has_traceback=True
            )
            
            # Get the HTML content that was passed to the HTML widget
            html_content = mock_html.call_args[0][0]
            
            # Assert dark mode styles are included
            assert "@media (prefers-color-scheme: dark)" in html_content
            assert "dark-bg" in html_content
            assert "dark-color" in html_content
    
    def test_accessibility_attributes(self, error_component):
        """Test that accessibility attributes are included in the output."""
        with patch.object(ErrorComponent, '_create_toggle_button') as mock_toggle, \
             patch('ipywidgets.HTML') as mock_html:
            
            # Setup mocks
            mock_toggle.return_value = "<button>Toggle</button>"
            mock_html.return_value = MagicMock()
            
            # Call the method
            error_component._create_main_error_display(
                message="Test message",
                style={
                    "bg": "light-bg",
                    "border": "light-border",
                    "color": "light-color",
                    "icon": "🚨",
                    "aria_label": "Error Notification"
                },
                has_traceback=False
            )
            
            # Get the HTML content that was passed to the HTML widget
            html_content = mock_html.call_args[0][0]
            
            # Assert accessibility attributes are included
            assert 'role="alert"' in html_content
            assert 'aria-label="Error Notification"' in html_content
    
    def test_responsive_design(self, error_component):
        """Test that responsive design styles are included in the output."""
        with patch.object(ErrorComponent, '_create_toggle_button') as mock_toggle, \
             patch('ipywidgets.HTML') as mock_html:
            
            # Setup mocks
            mock_toggle.return_value = "<button>Toggle</button>"
            mock_html.return_value = MagicMock()
            
            # Call the method
            error_component._create_main_error_display(
                message="Test message",
                style={
                    "bg": "light-bg",
                    "border": "light-border",
                    "color": "light-color",
                    "icon": "🚨"
                },
                has_traceback=True
            )
            
            # Get the HTML content that was passed to the HTML widget
            html_content = mock_html.call_args[0][0]
            
            # Assert responsive design styles are included
            assert "@media (max-width: 600px)" in html_content
