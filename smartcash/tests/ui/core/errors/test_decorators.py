"""
Tests for error decorators in smartcash.ui.core.errors.decorators
"""
import pytest
from unittest.mock import MagicMock, patch
from smartcash.ui.core.errors.decorators import (
    handle_errors,
    safe_component_operation,
    log_errors,
    with_error_handling,
    wrap_async,
    wrap_sync
)
from smartcash.ui.core.errors.exceptions import SmartCashUIError

class TestErrorDecorators:
    """Test cases for error handling decorators"""
    
    def test_handle_errors_success(self):
        """Test handle_errors decorator with successful execution"""
        @handle_errors()
        def test_func():
            return "success"
            
        assert test_func() == "success"
        
    def test_handle_errors_exception(self):
        """Test handle_errors decorator with exception"""
        @handle_errors()
        def test_func():
            raise ValueError("Test error")
            
        with pytest.raises(SmartCashUIError) as exc_info:
            test_func()
            
        assert "Test error" in str(exc_info.value)
        
    def test_safe_component_operation_success(self):
        """Test safe_component_operation with successful execution"""
        mock_component = MagicMock()
        
        @safe_component_operation("test_operation")
        def test_func(comp):
            return "success"
            
        assert test_func(mock_component) == "success"
        
    def test_safe_component_operation_error(self):
        """Test safe_component_operation with error"""
        mock_component = MagicMock()
        
        @safe_component_operation("test_operation")
        def test_func(comp):
            raise ValueError("Test error")
            
        with pytest.raises(SmartCashUIError):
            test_func(mock_component)
            
    def test_log_errors_decorator(self):
        """Test log_errors decorator"""
        mock_logger = MagicMock()
        
        @log_errors(logger=mock_logger, level="error")
        def test_func():
            raise ValueError("Test error")
            
        with pytest.raises(ValueError):
            test_func()
            
        mock_logger.error.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_wrap_async(self):
        """Test wrap_async decorator"""
        mock_handler = MagicMock()
        
        @wrap_async(mock_handler)
        async def async_func():
            return "success"
            
        result = await async_func()
        assert result == "success"
        
    def test_wrap_sync(self):
        """Test wrap_sync decorator"""
        mock_handler = MagicMock()
        
        @wrap_sync(mock_handler)
        def sync_func():
            return "success"
            
        assert sync_func() == "success"
        
    def test_with_error_handling_success(self):
        """Test with_error_handling context manager success"""
        with with_error_handling("test_component", "test_operation") as result:
            result.value = "success"
            
        assert result.value == "success"
        
    def test_with_error_handling_error(self):
        """Test with_error_handling context manager with error"""
        with pytest.raises(SmartCashUIError):
            with with_error_handling("test_component", "test_operation"):
                raise ValueError("Test error")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
