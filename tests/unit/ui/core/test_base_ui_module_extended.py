"""
Extended comprehensive tests for smartcash.ui.core.base_ui_module.BaseUIModule

This test suite provides additional coverage for edge cases, error conditions,
and advanced functionality not covered in the main test suite.
"""
import os
import sys
import pytest
import threading
import time
import json
from unittest.mock import MagicMock, patch, ANY, call, PropertyMock
from typing import Dict, Any, Optional
from collections import defaultdict

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Import the module under test
from smartcash.ui.core.base_ui_module import BaseUIModule


# ============================================================================
# Advanced Test Fixtures
# ============================================================================

class AdvancedTestUIModule(BaseUIModule):
    """Advanced test implementation with configurable behavior."""
    
    def __init__(self, 
                 module_name: str = "advanced_test", 
                 parent_module: str = "test",
                 fail_config: bool = False,
                 fail_ui_creation: bool = False,
                 custom_config: Dict[str, Any] = None,
                 **kwargs):
        self.fail_config = fail_config
        self.fail_ui_creation = fail_ui_creation
        self.custom_config = custom_config or {"test": "value", "nested": {"key": "value"}}
        self.operation_calls = []
        self.button_clicks = []
        super().__init__(module_name=module_name, parent_module=parent_module, **kwargs)
    
    def get_default_config(self) -> Dict[str, Any]:
        if self.fail_config:
            raise ValueError("Config creation failed")
        return self.custom_config
    
    def create_config_handler(self, config: Dict[str, Any]) -> Any:
        if self.fail_config:
            raise RuntimeError("Config handler creation failed")
        
        handler = MagicMock()
        handler.get_current_config.return_value = config
        handler.initialize = MagicMock()
        handler.save_config = MagicMock(return_value={'success': True})
        handler.validate_config = MagicMock(return_value={'valid': True, 'errors': []})
        return handler
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if self.fail_ui_creation:
            raise ConnectionError("UI component creation failed")
        
        return {
            "main_container": MagicMock(),
            "header_container": MagicMock(),
            "form_container": MagicMock(),
            "action_container": MagicMock(),
            "operation_container": MagicMock(),
            "footer_container": MagicMock()
        }
    
    def test_operation(self, **kwargs):
        """Test operation for tracking calls."""
        self.operation_calls.append(kwargs)
        return {"success": True, "data": "test_result"}
    
    def test_button_handler(self, button):
        """Test button handler for tracking clicks."""
        self.button_clicks.append(button.description if hasattr(button, 'description') else str(button))
        return {"success": True, "button": str(button)}


@pytest.fixture
def advanced_module():
    """Fixture providing an advanced test module."""
    module = AdvancedTestUIModule(enable_environment=False)
    
    # Mock logging methods
    module.log = MagicMock()
    module.log_debug = MagicMock()
    module.log_info = MagicMock()
    module.log_warning = MagicMock()
    module.log_error = MagicMock()
    
    # Mock progress methods
    module.start_progress = MagicMock()
    module.update_progress = MagicMock()
    module.complete_progress = MagicMock()
    module.error_progress = MagicMock()
    
    # Mock button methods
    module.disable_all_buttons = MagicMock()
    module.enable_all_buttons = MagicMock()
    
    return module


@pytest.fixture
def thread_safe_module():
    """Fixture providing a module for thread safety testing."""
    return AdvancedTestUIModule(
        module_name="thread_test",
        enable_environment=False,
        custom_config={"thread_safe": True, "counter": 0}
    )


# ============================================================================
# Extended Test Cases
# ============================================================================

def test_initialization_edge_cases():
    """Test initialization with various edge case parameters."""
    # Test with minimal parameters
    module = AdvancedTestUIModule()
    assert module.module_name == "advanced_test"
    assert module.parent_module == "test"
    
    # Test with None parent module
    module = AdvancedTestUIModule(parent_module=None)
    assert module.parent_module is None
    assert module.full_module_name == "advanced_test"
    
    # Test with empty strings
    module = AdvancedTestUIModule(module_name="", parent_module="")
    assert module.module_name == ""
    assert module.parent_module == ""
    assert module.full_module_name == ""
    
    # Test with special characters
    module = AdvancedTestUIModule(
        module_name="test-module_v2.0",
        parent_module="parent.module"
    )
    assert module.module_name == "test-module_v2.0"
    assert module.parent_module == "parent.module"


def test_config_handler_failure_scenarios(advanced_module):
    """Test various config handler failure scenarios."""
    # Test config creation failure
    failing_module = AdvancedTestUIModule(fail_config=True, enable_environment=False)
    failing_module.log_error = MagicMock()
    
    with pytest.raises(ValueError):
        failing_module.get_default_config()
    
    # Test config handler initialization failure
    failing_module = AdvancedTestUIModule(fail_config=True, enable_environment=False)
    failing_module.log_error = MagicMock()
    
    with pytest.raises(RuntimeError):
        failing_module.create_config_handler({})


def test_ui_component_creation_failure(advanced_module):
    """Test UI component creation failure scenarios."""
    failing_module = AdvancedTestUIModule(fail_ui_creation=True, enable_environment=False)
    failing_module.log_error = MagicMock()
    
    # Mock other required methods to avoid side effects
    failing_module._initialize_config_handler = MagicMock()
    failing_module._initialize_progress_display = MagicMock()
    failing_module._register_default_operations = MagicMock()
    failing_module._register_dynamic_button_handlers = MagicMock()
    failing_module._validate_button_handler_integrity = MagicMock()
    failing_module._link_action_container = MagicMock()
    failing_module._log_initialization_complete = MagicMock()
    
    # Initialization should fail due to UI creation failure
    result = failing_module.initialize()
    assert result is False
    assert failing_module._is_initialized is False


def test_configuration_validation_edge_cases(advanced_module):
    """Test configuration validation with edge cases."""
    # Initialize the module
    advanced_module.initialize()
    
    # Test validation with various data types
    test_configs = [
        None,
        {},
        {"empty": None},
        {"boolean": True, "number": 42, "string": "test"},
        {"nested": {"deep": {"value": "test"}}},
        {"list": [1, 2, 3], "tuple": (4, 5, 6)},
        {"unicode": "тест", "special_chars": "!@#$%^&*()"}
    ]
    
    for config in test_configs:
        # Should handle all config types gracefully
        try:
            result = advanced_module.validate_all()
            assert isinstance(result, dict)
            assert 'is_valid' in result
        except Exception as e:
            pytest.fail(f"Validation failed for config {config}: {e}")


def test_button_handler_advanced_scenarios(advanced_module):
    """Test advanced button handler scenarios."""
    # Initialize the module
    advanced_module.initialize()
    
    # Test registering multiple handlers for the same button
    handler1 = MagicMock(return_value="result1")
    handler2 = MagicMock(return_value="result2")
    
    advanced_module.register_button_handler("test_btn", handler1)
    advanced_module.register_button_handler("test_btn", handler2)  # Should override
    
    # The second handler should be registered (overriding the first)
    assert "test_btn" in advanced_module._button_handlers
    
    # Test button handler with exception
    def failing_handler(button):
        raise ValueError("Handler failed")
    
    advanced_module.register_button_handler("failing_btn", failing_handler)
    wrapped_handler = advanced_module._wrap_button_handler("failing_btn", failing_handler)
    
    button = MagicMock()
    button.description = "Failing Button"
    
    # Handler should catch exception and handle it gracefully
    result = wrapped_handler(button)
    
    # Verify error was logged
    advanced_module.log_error.assert_called()


def test_operation_execution_advanced_scenarios(advanced_module):
    """Test advanced operation execution scenarios."""
    # Initialize the module
    advanced_module.initialize()
    
    # Test operation with custom validation function
    def custom_validator():
        return {'valid': True, 'message': 'Validation passed'}
    
    def test_operation():
        return {'success': True, 'data': 'test'}
    
    # Test operation execution with validation
    result = advanced_module._execute_operation_with_wrapper(
        operation_name="validated operation",
        operation_func=test_operation,
        validation_func=custom_validator,
        success_message="Operation completed"
    )
    
    assert result['success'] is True
    assert 'Operation completed' in result['message']
    
    # Test operation with failing validation
    def failing_validator():
        return {'valid': False, 'message': 'Validation failed'}
    
    result = advanced_module._execute_operation_with_wrapper(
        operation_name="invalid operation",
        operation_func=test_operation,
        validation_func=failing_validator
    )
    
    assert result['success'] is False
    assert 'Validation failed' in result['message']


def test_environment_detection_scenarios():
    """Test environment detection in various scenarios."""
    # Test with different environment settings
    test_cases = [
        (True, "Environment enabled"),
        (False, "Environment disabled"),
    ]
    
    for enable_env, description in test_cases:
        module = AdvancedTestUIModule(enable_environment=enable_env)
        
        # Environment detection should work regardless of enable_environment
        env = module._detect_environment()
        assert env in ['colab', 'jupyter', 'local']
        
        # Environment info should always be available
        info = module.get_environment_info()
        assert isinstance(info, dict)
        assert 'environment_type' in info


def test_thread_safety(thread_safe_module):
    """Test thread safety of the module."""
    # Initialize the module
    thread_safe_module.log = MagicMock()
    thread_safe_module.log_debug = MagicMock()
    thread_safe_module.log_error = MagicMock()
    
    results = []
    errors = []
    
    def worker_thread(thread_id):
        try:
            # Each thread performs operations
            config = thread_safe_module.get_current_config()
            config['thread_id'] = thread_id
            config['counter'] = config.get('counter', 0) + 1
            
            # Simulate some work
            time.sleep(0.01)
            
            # Store result
            results.append({
                'thread_id': thread_id,
                'config': config.copy()
            })
            
        except Exception as e:
            errors.append({'thread_id': thread_id, 'error': e})
    
    # Run multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker_thread, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify results
    assert len(errors) == 0, f"Thread errors: {errors}"
    assert len(results) == 5
    
    # Verify each thread got a result
    thread_ids = [r['thread_id'] for r in results]
    assert set(thread_ids) == {0, 1, 2, 3, 4}


def test_memory_management(advanced_module):
    """Test memory management and cleanup."""
    # Initialize with many components
    advanced_module.initialize()
    
    # Add many button handlers and operations
    for i in range(100):
        handler = MagicMock()
        advanced_module.register_button_handler(f"btn_{i}", handler)
        advanced_module.register_operation_handler(f"op_{i}", handler)
    
    # Verify they were registered
    assert len(advanced_module._button_handlers) >= 100
    assert len(advanced_module._operation_handlers) >= 100
    
    # Cleanup should clear everything
    advanced_module.cleanup()
    
    # Verify cleanup
    assert not advanced_module._is_initialized
    assert not advanced_module._ui_components or len(advanced_module._ui_components) == 0


def test_serialization_compatibility(advanced_module):
    """Test that module state can be serialized."""
    # Initialize the module
    advanced_module.initialize()
    
    # Get module info (this should be serializable)
    info = advanced_module.get_module_info()
    
    # Test JSON serialization
    try:
        json_str = json.dumps(info, default=str)  # Use default=str for non-serializable objects
        restored_info = json.loads(json_str)
        
        # Verify basic structure is preserved
        assert restored_info['module_name'] == info['module_name']
        assert restored_info['parent_module'] == info['parent_module']
        assert restored_info['full_module_name'] == info['full_module_name']
        
    except TypeError as e:
        pytest.fail(f"Module info should be serializable: {e}")


def test_unicode_and_special_characters(advanced_module):
    """Test handling of unicode and special characters."""
    # Test with unicode config
    unicode_config = {
        'name': 'тестовый модуль',
        'description': '这是一个测试模块',
        'emoji': '🚀💻🔧',
        'special_chars': '!@#$%^&*()[]{}|\\:";\'<>?,./',
        'nested': {
            'unicode_key': 'مرحبا بالعالم',
            'mixed': 'English + Русский + 中文'
        }
    }
    
    # Module should handle unicode config gracefully
    unicode_module = AdvancedTestUIModule(
        module_name="unicode_test",
        custom_config=unicode_config,
        enable_environment=False
    )
    unicode_module.log = MagicMock()
    unicode_module.log_error = MagicMock()
    
    # Should initialize without issues
    result = unicode_module.initialize()
    assert result is True
    
    # Should be able to retrieve config
    config = unicode_module.get_current_config()
    assert config['name'] == 'тестовый модуль'
    assert config['emoji'] == '🚀💻🔧'


def test_error_propagation_and_handling(advanced_module):
    """Test error propagation and handling mechanisms."""
    # Test with various exception types
    exception_types = [
        ValueError("Value error"),
        RuntimeError("Runtime error"),
        ConnectionError("Connection error"),
        TimeoutError("Timeout error"),
        KeyError("key_error"),
        AttributeError("Attribute not found")
    ]
    
    for exception in exception_types:
        # Create operation that raises this exception
        def failing_operation():
            raise exception
        
        # Execute operation and verify error handling
        result = advanced_module._execute_operation_with_wrapper(
            operation_name=f"test {type(exception).__name__}",
            operation_func=failing_operation
        )
        
        # Should handle all exception types gracefully
        assert result['success'] is False
        assert str(exception) in result['message']


def test_state_consistency_after_errors(advanced_module):
    """Test that module state remains consistent after errors."""
    # Initialize successfully first
    advanced_module.initialize()
    initial_state = advanced_module._is_initialized
    
    # Cause various errors
    try:
        advanced_module._execute_operation_with_wrapper(
            operation_name="failing operation",
            operation_func=lambda: 1/0  # ZeroDivisionError
        )
    except:
        pass  # Ignore the error
    
    # State should remain consistent
    assert advanced_module._is_initialized == initial_state
    
    # Should still be able to perform operations
    result = advanced_module._execute_operation_with_wrapper(
        operation_name="recovery operation",
        operation_func=lambda: {"success": True, "recovered": True}
    )
    
    assert result['success'] is True


def test_performance_with_many_operations(advanced_module):
    """Test performance with many registered operations."""
    # Initialize the module
    advanced_module.initialize()
    
    # Register many operations
    start_time = time.time()
    
    for i in range(1000):
        def operation_func(index=i):
            return {"success": True, "index": index}
        
        advanced_module.register_operation_handler(f"op_{i}", operation_func)
    
    registration_time = time.time() - start_time
    
    # Should register operations efficiently (less than 1 second for 1000 operations)
    assert registration_time < 1.0, f"Registration took too long: {registration_time:.2f}s"
    
    # Should be able to execute operations efficiently
    start_time = time.time()
    
    for i in range(0, 100, 10):  # Test every 10th operation
        result = advanced_module.execute_operation(f"op_{i}")
        assert result['success'] is True
        assert result['index'] == i
    
    execution_time = time.time() - start_time
    
    # Should execute operations efficiently
    assert execution_time < 1.0, f"Execution took too long: {execution_time:.2f}s"


def test_resource_cleanup_on_failure():
    """Test that resources are properly cleaned up when initialization fails."""
    # Create module that will fail during initialization
    failing_module = AdvancedTestUIModule(
        fail_ui_creation=True,
        enable_environment=False
    )
    failing_module.log_error = MagicMock()
    
    # Mock successful steps before the failure
    failing_module._initialize_config_handler = MagicMock()
    
    # Attempt initialization (should fail)
    result = failing_module.initialize()
    assert result is False
    
    # Resources should be cleaned up
    assert failing_module._is_initialized is False
    
    # Should be able to try initialization again after fixing the issue
    failing_module.fail_ui_creation = False
    result = failing_module.initialize()
    assert result is True


def test_module_inheritance_patterns():
    """Test various inheritance patterns and their behavior."""
    
    class SpecializedModule(AdvancedTestUIModule):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.specialized_data = {"type": "specialized"}
        
        def get_default_config(self):
            base_config = super().get_default_config()
            base_config.update({"specialized": True})
            return base_config
        
        def specialized_operation(self):
            return {"success": True, "specialized": True}
    
    # Test specialized module
    module = SpecializedModule(
        module_name="specialized",
        enable_environment=False
    )
    module.log = MagicMock()
    
    # Should inherit all base functionality
    assert hasattr(module, 'initialize')
    assert hasattr(module, 'get_current_config')
    
    # Should have specialized functionality
    assert hasattr(module, 'specialized_operation')
    assert module.specialized_data['type'] == "specialized"
    
    # Configuration should be merged
    module.initialize()
    config = module.get_current_config()
    assert config.get('specialized') is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])