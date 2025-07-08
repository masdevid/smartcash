"""
Comprehensive tests for Core Initializers module.

This module provides complete test coverage for the core initializer infrastructure,
including BaseInitializer functionality, error handling, and integration tests.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from smartcash.ui.core.initializers.base_initializer import BaseInitializer
from smartcash.ui.core.errors import ErrorContext, SmartCashUIError


class TestBaseInitializer:
    """Test suite for BaseInitializer functionality."""
    
    @pytest.fixture
    def concrete_initializer(self):
        """Create a concrete implementation of BaseInitializer for testing."""
        
        class TestInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                return {"status": "success", "data": kwargs}
        
        return TestInitializer("test_module", "test_parent")
    
    def test_base_initializer_creation(self, concrete_initializer):
        """Test basic creation of BaseInitializer."""
        assert concrete_initializer.module_name == "test_module"
        assert concrete_initializer.parent_module == "test_parent"
        assert concrete_initializer.full_module_name == "test_parent.test_module"
        assert hasattr(concrete_initializer, 'logger')
        assert hasattr(concrete_initializer, '_error_context')
        assert not concrete_initializer.is_initialized
    
    def test_base_initializer_no_parent(self):
        """Test BaseInitializer creation without parent module."""
        
        class TestInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                return {"status": "success"}
        
        initializer = TestInitializer("standalone_module")
        assert initializer.module_name == "standalone_module"
        assert initializer.parent_module is None
        assert initializer.full_module_name == "standalone_module"
    
    def test_error_context_creation(self, concrete_initializer):
        """Test that ErrorContext is properly created with details attribute."""
        assert hasattr(concrete_initializer._error_context, 'details')
        assert isinstance(concrete_initializer._error_context.details, dict)
        assert 'module_name' in concrete_initializer._error_context.details
        assert 'parent_module' in concrete_initializer._error_context.details
    
    def test_initialization_success(self, concrete_initializer):
        """Test successful initialization."""
        result = concrete_initializer.initialize(test_param="test_value")
        
        assert concrete_initializer.is_initialized
        assert result["status"] == "success"
        assert result["data"]["test_param"] == "test_value"
        assert concrete_initializer.initialization_result == result
    
    def test_double_initialization(self, concrete_initializer):
        """Test that double initialization returns cached result."""
        result1 = concrete_initializer.initialize(param1="value1")
        result2 = concrete_initializer.initialize(param2="value2")
        
        # Second call should return the same result
        assert result1 == result2
        assert result2["data"]["param1"] == "value1"
        assert "param2" not in result2["data"]
    
    def test_initialization_error_handling(self):
        """Test error handling during initialization."""
        
        class FailingInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                raise ValueError("Test error")
        
        initializer = FailingInitializer("failing_module")
        
        with pytest.raises(SmartCashUIError):
            initializer.initialize()
        
        assert not initializer.is_initialized
        assert initializer.error_count > 0
        assert "Test error" in str(initializer.last_error)
    
    def test_handle_error_method(self, concrete_initializer):
        """Test the handle_error method."""
        with pytest.raises(SmartCashUIError):
            concrete_initializer.handle_error("Test error message", test_context="test_value")
        
        assert concrete_initializer.error_count == 1
        assert concrete_initializer.last_error == "Test error message"
    
    def test_properties(self, concrete_initializer):
        """Test all properties of BaseInitializer."""
        # Before initialization
        assert concrete_initializer.is_initialized is False
        assert concrete_initializer.initialization_result is None
        assert concrete_initializer.error_count == 0
        assert concrete_initializer.last_error is None
        
        # After initialization
        concrete_initializer.initialize()
        assert concrete_initializer.is_initialized is True
        assert concrete_initializer.initialization_result is not None
    
    def test_error_context_updates(self, concrete_initializer):
        """Test that error context gets updated properly."""
        with pytest.raises(SmartCashUIError):
            concrete_initializer.handle_error("Error", extra_info="test_info")
        
        # Check that the error context details were updated
        assert hasattr(concrete_initializer._error_context, 'details')
        assert isinstance(concrete_initializer._error_context.details, dict)
    
    def test_logger_functionality(self, concrete_initializer):
        """Test that logger is properly configured."""
        assert hasattr(concrete_initializer, 'logger')
        assert concrete_initializer.logger is not None
        
        # Test that we can call logger methods without error
        concrete_initializer.logger.debug("Test debug message")
        concrete_initializer.logger.info("Test info message")


class TestConcreteInitializerImplementations:
    """Test various concrete implementations of BaseInitializer."""
    
    def test_minimal_implementation(self):
        """Test minimal concrete implementation."""
        
        class MinimalInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                return {"status": "minimal_success"}
        
        initializer = MinimalInitializer("minimal")
        result = initializer.initialize()
        
        assert result["status"] == "minimal_success"
        assert initializer.is_initialized
    
    def test_complex_implementation(self):
        """Test more complex concrete implementation."""
        
        class ComplexInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                # Simulate complex initialization
                self.custom_data = kwargs.get('data', {})
                self.initialized_components = ['component1', 'component2']
                
                return {
                    "status": "complex_success",
                    "components": self.initialized_components,
                    "data_keys": list(self.custom_data.keys())
                }
        
        test_data = {"key1": "value1", "key2": "value2"}
        initializer = ComplexInitializer("complex", "parent")
        result = initializer.initialize(data=test_data)
        
        assert result["status"] == "complex_success"
        assert result["components"] == ['component1', 'component2']
        assert result["data_keys"] == ['key1', 'key2']
        assert hasattr(initializer, 'custom_data')
        assert hasattr(initializer, 'initialized_components')
    
    def test_initialization_with_dependencies(self):
        """Test initialization that depends on external resources."""
        
        class DependentInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                required_service = kwargs.get('service')
                if not required_service:
                    raise ValueError("Service dependency required")
                
                return {
                    "status": "dependent_success",
                    "service_name": required_service.name
                }
        
        # Test with missing dependency
        initializer = DependentInitializer("dependent")
        with pytest.raises(SmartCashUIError):
            initializer.initialize()
        
        # Test with provided dependency
        mock_service = Mock()
        mock_service.name = "test_service"
        
        result = initializer.initialize(service=mock_service)
        assert result["status"] == "dependent_success"
        assert result["service_name"] == "test_service"


class TestErrorContextIntegration:
    """Test ErrorContext integration with BaseInitializer."""
    
    def test_error_context_attributes(self):
        """Test that ErrorContext has all required attributes."""
        context = ErrorContext(component="test", operation="test", details={"key": "value"})
        
        assert hasattr(context, 'details')
        assert isinstance(context.details, dict)
        assert context.details.get('key') == 'value'
    
    def test_error_context_in_initializer(self):
        """Test ErrorContext usage within BaseInitializer."""
        
        class TestInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                # Access error context details to ensure it works
                self._error_context.details.update({"test_key": "test_value"})
                return {"status": "success"}
        
        initializer = TestInitializer("test_module")
        result = initializer.initialize()
        
        assert result["status"] == "success"
        assert initializer._error_context.details.get("test_key") == "test_value"
    
    def test_error_context_thread_local(self):
        """Test that ErrorContext works with thread-local storage."""
        context1 = ErrorContext(test_data="data1")
        context2 = ErrorContext(test_data="data2")
        
        # Both contexts should maintain their own details
        assert context1.details.get('test_data') == 'data1'
        assert context2.details.get('test_data') == 'data2'


class TestInitializerPerformance:
    """Test performance aspects of BaseInitializer."""
    
    def test_initialization_performance(self):
        """Test that initialization completes within reasonable time."""
        import time
        
        class PerformanceInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                # Simulate some work
                time.sleep(0.001)  # 1ms
                return {"status": "performance_success"}
        
        initializer = PerformanceInitializer("performance_test")
        
        start_time = time.time()
        result = initializer.initialize()
        end_time = time.time()
        
        assert result["status"] == "performance_success"
        assert (end_time - start_time) < 1.0  # Should complete within 1 second
    
    def test_multiple_initializer_performance(self):
        """Test performance with multiple initializers."""
        import time
        
        class FastInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                return {"status": "fast_success", "id": kwargs.get('id', 0)}
        
        initializers = []
        start_time = time.time()
        
        # Create and initialize 10 initializers
        for i in range(10):
            initializer = FastInitializer(f"fast_{i}")
            result = initializer.initialize(id=i)
            initializers.append(initializer)
            assert result["status"] == "fast_success"
            assert result["id"] == i
        
        end_time = time.time()
        
        assert len(initializers) == 10
        assert all(init.is_initialized for init in initializers)
        assert (end_time - start_time) < 5.0  # Should complete within 5 seconds


class TestInitializerEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_module_name(self):
        """Test behavior with empty module name."""
        
        class EmptyNameInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                return {"status": "empty_name_success"}
        
        initializer = EmptyNameInitializer("")
        assert initializer.module_name == ""
        
        result = initializer.initialize()
        assert result["status"] == "empty_name_success"
    
    def test_special_characters_in_module_name(self):
        """Test behavior with special characters in module name."""
        
        class SpecialCharInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                return {"status": "special_char_success"}
        
        initializer = SpecialCharInitializer("module-with_special.chars")
        assert initializer.module_name == "module-with_special.chars"
        
        result = initializer.initialize()
        assert result["status"] == "special_char_success"
    
    def test_very_long_module_name(self):
        """Test behavior with very long module name."""
        
        class LongNameInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                return {"status": "long_name_success"}
        
        long_name = "very_long_module_name" * 10  # 200+ characters
        initializer = LongNameInitializer(long_name)
        assert initializer.module_name == long_name
        
        result = initializer.initialize()
        assert result["status"] == "long_name_success"
    
    def test_none_parameters_handling(self):
        """Test handling of None parameters."""
        
        class NoneParamInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                return {
                    "status": "none_param_success",
                    "none_values": [k for k, v in kwargs.items() if v is None]
                }
        
        initializer = NoneParamInitializer("none_test")
        result = initializer.initialize(param1=None, param2="value", param3=None)
        
        assert result["status"] == "none_param_success"
        assert "param1" in result["none_values"]
        assert "param3" in result["none_values"]
        assert "param2" not in result["none_values"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])