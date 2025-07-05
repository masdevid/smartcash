"""
Tests for error handlers in smartcash.ui.core.errors.handlers
"""
import pytest
from unittest.mock import MagicMock, patch
from smartcash.ui.core.errors.handlers import CoreErrorHandler
from smartcash.ui.core.errors.exceptions import SmartCashUIError
from smartcash.ui.core.errors.context import ErrorContext

class TestCoreErrorHandler:
    """Test cases for CoreErrorHandler"""
    
    @pytest.fixture
    def error_handler(self):
        """Create a CoreErrorHandler instance for testing"""
        return CoreErrorHandler()
    
    def test_initialization(self, error_handler):
        """Test handler initialization"""
        assert error_handler is not None
        assert error_handler.get_logger("test") is not None
        
    def test_handle_exception(self, error_handler):
        """Test exception handling"""
        mock_logger = MagicMock()
        error_handler.get_logger = MagicMock(return_value=mock_logger)
        
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            details={"key": "value"}
        )
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            error_handler.handle_exception(
                "Test error",
                context=context,
                error_level="ERROR"
            )
            
        mock_logger.error.assert_called_once()
        assert "Test error" in str(mock_logger.error.call_args[0][0])
        
    def test_log_error(self, error_handler):
        """Test error logging"""
        mock_logger = MagicMock()
        error_handler.get_logger = MagicMock(return_value=mock_logger)
        
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation"
        )
        
        error_handler.log_error(
            "Test warning",
            error_level="WARNING",
            context=context
        )
        
        mock_logger.warning.assert_called_once()
        assert "Test warning" in str(mock_logger.warning.call_args[0][0])
        
    def test_handle_ui_error(self, error_handler):
        """Test UI error handling"""
        mock_ui = MagicMock()
        error_handler.handle_ui_component_error(
            "Test UI error",
            component=mock_ui,
            error_level="ERROR"
        )
        
        # Verify UI component was updated with error
        mock_ui.update.assert_called_once()
        assert "error" in str(mock_ui.update.call_args[0][0]).lower()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
