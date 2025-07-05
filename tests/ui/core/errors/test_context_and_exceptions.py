"""
Tests for error context and exceptions in smartcash.ui.core.errors
"""
import pytest
from smartcash.ui.core.errors.context import ErrorContext
from smartcash.ui.core.errors.exceptions import (
    SmartCashUIError,
    ValidationError,
    ConfigurationError,
    NetworkError,
    AuthenticationError,
    AuthorizationError,
    ResourceNotFoundError,
    ResourceExistsError,
    TimeoutError,
    OperationNotSupportedError
)

class TestErrorContext:
    """Test cases for ErrorContext"""
    
    def test_error_context_creation(self):
        """Test basic ErrorContext creation"""
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            details={"key": "value"}
        )
        
        assert context.component == "TestComponent"
        assert context.operation == "test_operation"
        assert context.details == {"key": "value"}
        
    def test_error_context_update(self):
        """Test updating ErrorContext"""
        context = ErrorContext("TestComponent", "test_operation")
        context.update_details({"new_key": "new_value"})
        
        assert context.details == {"new_key": "new_value"}
        
    def test_error_context_str(self):
        """Test string representation of ErrorContext"""
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            details={"key": "value"}
        )
        
        assert "TestComponent" in str(context)
        assert "test_operation" in str(context)
        assert "key" in str(context)

class TestExceptions:
    """Test cases for custom exceptions"""
    
    def test_smartcash_ui_error(self):
        """Test base SmartCashUIError"""
        context = ErrorContext("TestComponent", "test_operation")
        error = SmartCashUIError("Test error", context=context)
        
        assert str(error) == "Test error"
        assert error.context == context
        assert error.error_code == "UNKNOWN_ERROR"
        
    def test_validation_error(self):
        """Test ValidationError"""
        error = ValidationError("Invalid input", field="username")
        assert error.error_code == "VALIDATION_ERROR"
        assert error.field == "username"
        
    def test_configuration_error(self):
        """Test ConfigurationError"""
        error = ConfigurationError("Missing config")
        assert error.error_code == "CONFIGURATION_ERROR"
        
    def test_network_error(self):
        """Test NetworkError"""
        error = NetworkError("Connection failed", url="http://example.com")
        assert error.error_code == "NETWORK_ERROR"
        assert error.url == "http://example.com"
        
    def test_authentication_error(self):
        """Test AuthenticationError"""
        error = AuthenticationError("Invalid credentials")
        assert error.error_code == "AUTHENTICATION_ERROR"
        
    def test_authorization_error(self):
        """Test AuthorizationError"""
        error = AuthorizationError("Permission denied")
        assert error.error_code == "AUTHORIZATION_ERROR"
        
    def test_resource_not_found_error(self):
        """Test ResourceNotFoundError"""
        error = ResourceNotFoundError("User not found", resource_type="user", resource_id=123)
        assert error.error_code == "RESOURCE_NOT_FOUND"
        assert error.resource_type == "user"
        assert error.resource_id == 123
        
    def test_resource_exists_error(self):
        """Test ResourceExistsError"""
        error = ResourceExistsError("User exists", resource_type="user", resource_id=123)
        assert error.error_code == "RESOURCE_EXISTS"
        assert error.resource_type == "user"
        assert error.resource_id == 123
        
    def test_timeout_error(self):
        """Test TimeoutError"""
        error = TimeoutError("Request timed out", timeout=30)
        assert error.error_code == "TIMEOUT_ERROR"
        assert error.timeout == 30
        
    def test_operation_not_supported_error(self):
        """Test OperationNotSupportedError"""
        error = OperationNotSupportedError("Operation not supported", operation="delete")
        assert error.error_code == "OPERATION_NOT_SUPPORTED"
        assert error.operation == "delete"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
