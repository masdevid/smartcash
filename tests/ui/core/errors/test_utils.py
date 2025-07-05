"""
Tests for error utilities in smartcash.ui.core.errors.utils
"""
import pytest
from unittest.mock import MagicMock, patch
from smartcash.ui.core.errors.utils import (
    create_error_context,
    error_handler_scope,
    with_error_handling,
    log_errors,
    safe_ui_operation
)
from smartcash.ui.core.errors.exceptions import SmartCashUIError
from smartcash.ui.core.errors.context import ErrorContext

class TestErrorUtils:
    """Test cases for error utility functions"""
    
    def test_create_error_context(self):
        """Test create_error_context helper function"""
        context = create_error_context(
            component="TestComponent",
            operation="test_operation",
            details={"key": "value"},
            ui_components={"output": MagicMock()}
        )
        
        assert isinstance(context, ErrorContext)
        assert context.component == "TestComponent"
        assert context.operation == "test_operation"
        assert context.details == {"key": "value"}
        assert "output" in context.ui_components
        
    def test_error_handler_scope_success(self):
        """Test error_handler_scope context manager with success"""
        mock_logger = MagicMock()
        
        with error_handler_scope(
            component="TestComponent",
            operation="test_operation",
            logger=mock_logger
        ) as result:
            result.value = "success"
            
        assert result.value == "success"
        
    def test_error_handler_scope_error(self):
        """Test error_handler_scope context manager with error"""
        mock_logger = MagicMock()
        
        with pytest.raises(SmartCashUIError):
            with error_handler_scope(
                component="TestComponent",
                operation="test_operation",
                logger=mock_logger
            ):
                raise ValueError("Test error")
                
        mock_logger.error.assert_called_once()
        
    def test_with_error_handling_decorator(self):
        """Test with_error_handling decorator"""
        mock_logger = MagicMock()
        
        @with_error_handling(
            component="TestComponent",
            operation="test_operation",
            logger=mock_logger
        )
        def test_func():
            return "success"
            
        assert test_func() == "success"
        
    def test_log_errors_decorator(self):
        """Test log_errors decorator"""
        mock_logger = MagicMock()
        
        @log_errors(
            logger=mock_logger,
            level="error",
            component="TestComponent",
            operation="test_operation"
        )
        def test_func():
            raise ValueError("Test error")
            
        with pytest.raises(ValueError):
            test_func()
            
        mock_logger.error.assert_called_once()
        
    def test_safe_ui_operation_decorator(self):
        """Test safe_ui_operation decorator"""
        mock_ui = MagicMock()
        
        @safe_ui_operation("TestComponent", "test_operation")
        def test_func(ui_component):
            return "success"
            
        assert test_func(mock_ui) == "success"
        
    def test_safe_ui_operation_with_error(self):
        """Test safe_ui_operation decorator with error"""
        mock_ui = MagicMock()
        
        @safe_ui_operation("TestComponent", "test_operation")
        def test_func(ui_component):
            raise ValueError("Test error")
            
        with pytest.raises(SmartCashUIError):
            test_func(mock_ui)
            
        # Verify UI was updated with error
        assert mock_ui.update.called

if __name__ == "__main__":
    pytest.main(["-v", __file__])
