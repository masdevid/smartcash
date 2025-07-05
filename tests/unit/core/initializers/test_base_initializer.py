"""
Tests for the base initializer functionality in the core module.
"""
import pytest
from unittest.mock import MagicMock, patch, ANY

# Import the mock errors module first to set up mocks
import smartcash.tests.unit.core.mock_core_errors

# Now import the initializer we want to test
from smartcash.ui.core.initializers.base_initializer import BaseInitializer

# Create a concrete implementation of BaseInitializer for testing
class ConcreteInitializer(BaseInitializer):
    """Concrete implementation of BaseInitializer for testing."""
    def __init__(self, module_name="test_module", parent_module=None):
        super().__init__(module_name, parent_module)
        self.initialized = False
        self.reset_called = False
        self.saved_configs = []
    
    def _initialize_impl(self, *args, **kwargs) -> dict:
        """Implementation of initialization logic."""
        self.initialized = True
        return {"status": "initialized"}
    
    def reset_config(self) -> None:
        """Reset the initializer configuration."""
        try:
            self.reset_called = True
            self._is_initialized = False
            self.initialized = False
            self.saved_configs = []
            self._initialization_result = None
            self._error_count = 0  # Reset error count to zero
            self._last_error = None  # Clear any previous error
        except Exception as e:
            self._last_error = e
            raise
    
    def save_config(self, config: dict) -> None:
        """Save the configuration.
        
        Args:
            config: Configuration dictionary to save
        """
        self.saved_configs.append(config)
        # Call the error handler's save_config if it exists
        if hasattr(self, '_error_handler') and hasattr(self._error_handler, 'save_config'):
            self._error_handler.save_config(
                config=config,
                component=self.__class__.__name__,
                context=getattr(self, '_error_context', None)
            )

class TestBaseInitializer:
    """Test cases for the BaseInitializer class."""
    
    @pytest.fixture
    def mock_error_handler(self):
        """Create a mock error handler."""
        mock = MagicMock()
        mock.handle_error.return_value = None
        return mock
    
    @pytest.fixture
    def base_initializer(self, mock_error_handler):
        """Create a concrete initializer instance with a mock error handler."""
        with patch('smartcash.ui.core.initializers.base_initializer.get_error_handler', 
                  return_value=mock_error_handler):
            initializer = ConcreteInitializer()
            initializer._error_handler = mock_error_handler
            return initializer
    
    def test_initialization(self, base_initializer, mock_error_handler):
        """Test that BaseInitializer initializes correctly."""
        assert hasattr(base_initializer, '_error_handler')
        assert base_initializer._error_handler == mock_error_handler
    
    def test_initialize_calls_implementation(self, base_initializer):
        """Test that initialize calls the implementation method."""
        with patch.object(base_initializer, '_initialize_impl', return_value={"status": "test"}) as mock_impl:
            result = base_initializer.initialize()
            mock_impl.assert_called_once()
            assert result == {"status": "test"}
    
    def test_reset_config(self, base_initializer):
        """Test that reset_config resets the initializer state."""
        base_initializer._is_initialized = True
        base_initializer._initialization_result = {'test': 'data'}
        base_initializer._error_count = 5
        base_initializer._last_error = Exception('Test error')
        
        base_initializer.reset_config()
        
        assert base_initializer._is_initialized is False
        assert base_initializer._initialization_result is None
        assert base_initializer._error_count == 0
        assert base_initializer._last_error is None
    
    def test_save_config(self, base_initializer):
        """Test that save_config calls the error handler's save_config method."""
        test_config = {'test': 'config'}
        base_initializer.save_config(test_config)
        base_initializer._error_handler.save_config.assert_called_once_with(
            config=test_config,
            component=base_initializer.__class__.__name__,
            context=base_initializer._error_context
        )
