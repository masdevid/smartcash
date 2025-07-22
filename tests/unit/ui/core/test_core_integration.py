"""
Integration tests for smartcash.ui.core components

This test suite covers integration scenarios between:
- BaseUIModule and its mixins
- UI factory and module creation
- Logging integration
- Validation integration
- Error handling integration
"""
import os
import sys
import pytest
import threading
import time
from unittest.mock import MagicMock, patch, ANY, call, PropertyMock
from typing import Dict, Any, Optional

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Import core components
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.mixins import (
    ConfigurationMixin,
    OperationMixin,
    LoggingMixin,
    ButtonHandlerMixin,
    ValidationMixin,
    DisplayMixin,
    ColabSecretsMixin
)

try:
    from smartcash.ui.core import ui_factory, ui_utils
    from smartcash.ui.core.logging import ui_logging_manager
    from smartcash.ui.core.validation import button_validator
except ImportError as e:
    print(f"Optional imports not available: {e}")


# ============================================================================
# Integration Test Fixtures
# ============================================================================

class FullIntegrationModule(BaseUIModule):
    """Full integration test module with all functionality."""
    
    def __init__(self, 
                 module_name: str = "integration_test",
                 parent_module: str = "core_tests",
                 test_config: Optional[Dict] = None,
                 **kwargs):
        self.test_config = test_config or {
            "module": {
                "name": module_name,
                "version": "1.0.0",
                "enabled": True
            },
            "ui": {
                "theme": "default",
                "components": ["header", "form", "action", "operation"]
            },
            "operations": {
                "test_op": {"timeout": 30, "retry": 3},
                "long_op": {"timeout": 300, "retry": 1}
            },
            "validation": {
                "required_components": ["operation_container", "action_container"],
                "strict_mode": False
            }
        }
        self.operation_results = []
        self.button_events = []
        super().__init__(module_name=module_name, parent_module=parent_module, **kwargs)
    
    def get_default_config(self) -> Dict[str, Any]:
        return self.test_config
    
    def create_config_handler(self, config: Dict[str, Any]) -> Any:
        handler = MagicMock()
        handler.get_current_config.return_value = config
        handler.save_config.return_value = {"success": True, "path": "/test/config.yaml"}
        handler.validate_config.return_value = {"valid": True, "errors": []}
        handler.initialize = MagicMock()
        return handler
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Create comprehensive mock UI components
        components = {}
        
        for component_name in config.get("ui", {}).get("components", []):
            component = MagicMock()
            component.name = f"{component_name}_container"
            component.visible = True
            component.enabled = True
            component.update = MagicMock()
            components[f"{component_name}_container"] = component
        
        # Add required components
        if "operation_container" not in components:
            operation_container = MagicMock()
            operation_container.progress_tracker = MagicMock()
            operation_container.log = MagicMock()
            operation_container.show_dialog = MagicMock()
            components["operation_container"] = operation_container
        
        if "action_container" not in components:
            action_container = MagicMock()
            action_container.disable_all_buttons = MagicMock()
            action_container.enable_all_buttons = MagicMock()
            components["action_container"] = action_container
        
        return components
    
    def test_operation_sync(self, **kwargs):
        """Synchronous test operation."""
        result = {
            "success": True,
            "operation": "test_operation_sync",
            "kwargs": kwargs,
            "timestamp": time.time()
        }
        self.operation_results.append(result)
        return result
    
    def test_operation_with_progress(self, progress_callback=None, **kwargs):
        """Test operation with progress reporting."""
        if progress_callback:
            progress_callback(0.0, "Starting operation")
            time.sleep(0.01)  # Simulate some work
            progress_callback(0.5, "Halfway done")
            time.sleep(0.01)
            progress_callback(1.0, "Operation complete")
        
        result = {
            "success": True,
            "operation": "test_operation_with_progress",
            "progress_reported": progress_callback is not None
        }
        self.operation_results.append(result)
        return result
    
    def test_operation_failing(self, **kwargs):
        """Test operation that fails."""
        error_msg = kwargs.get("error_msg", "Test operation failed")
        raise RuntimeError(error_msg)
    
    def test_button_handler(self, button):
        """Test button handler."""
        event = {
            "button_id": getattr(button, 'description', str(button)),
            "timestamp": time.time(),
            "enabled": getattr(button, 'enabled', True)
        }
        self.button_events.append(event)
        return {"success": True, "button_event": event}


@pytest.fixture
def integration_module():
    """Fixture providing a full integration test module."""
    module = FullIntegrationModule(enable_environment=False)
    
    # Mock all logging methods to avoid real logging during tests
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


# ============================================================================
# Core Integration Tests
# ============================================================================

class TestBaseUIModuleIntegration:
    """Integration tests for BaseUIModule with all its components."""
    
    def test_full_initialization_lifecycle(self, integration_module):
        """Test complete initialization lifecycle."""
        # Module should start uninitialized
        assert not integration_module._is_initialized
        
        # Initialize should succeed
        result = integration_module.initialize()
        assert result is True
        assert integration_module._is_initialized
        
        # All components should be available
        assert integration_module._ui_components is not None
        assert len(integration_module._ui_components) > 0
        
        # Config handler should be initialized
        assert integration_module._config_handler is not None
        
        # Should be able to get current config
        config = integration_module.get_current_config()
        assert isinstance(config, dict)
        assert "module" in config
    
    def test_configuration_integration(self, integration_module):
        """Test configuration system integration."""
        integration_module.initialize()
        
        # Test configuration retrieval
        config = integration_module.get_current_config()
        assert config["module"]["name"] == "integration_test"
        assert config["ui"]["theme"] == "default"
        
        # Test configuration validation
        validation_result = integration_module.validate_current_config()
        assert validation_result["valid"] is True
        
        # Test configuration saving
        save_result = integration_module.save_config()
        assert save_result["success"] is True
    
    def test_operation_system_integration(self, integration_module):
        """Test operation system integration."""
        integration_module.initialize()
        
        # Register operations
        integration_module.register_operation_handler(
            "sync_op", integration_module.test_operation_sync
        )
        integration_module.register_operation_handler(
            "progress_op", integration_module.test_operation_with_progress
        )
        integration_module.register_operation_handler(
            "failing_op", integration_module.test_operation_failing
        )
        
        # Test successful operation
        result = integration_module.execute_operation("sync_op", test_param="value")
        assert result["success"] is True
        assert result["operation"] == "test_operation_sync"
        assert result["kwargs"]["test_param"] == "value"
        
        # Test operation with progress
        result = integration_module.execute_operation("progress_op")
        assert result["success"] is True
        assert result["progress_reported"] is True
        
        # Test failing operation
        result = integration_module.execute_operation("failing_op", error_msg="Custom error")
        assert result["success"] is False
        assert "Custom error" in result["error"]
    
    def test_button_handler_integration(self, integration_module):
        """Test button handler system integration."""
        integration_module.initialize()
        
        # Register button handler
        integration_module.register_button_handler(
            "test_btn", integration_module.test_button_handler
        )
        
        # Simulate button click
        mock_button = MagicMock()
        mock_button.description = "Test Button"
        mock_button.enabled = True
        
        wrapped_handler = integration_module._wrap_button_handler(
            "test_btn", integration_module.test_button_handler
        )
        
        result = wrapped_handler(mock_button)
        
        # Verify handler was called and event recorded
        assert len(integration_module.button_events) == 1
        assert integration_module.button_events[0]["button_id"] == "Test Button"
        
        # Verify button state was updated
        assert "test_btn" in integration_module._button_states
    
    def test_validation_integration(self, integration_module):
        """Test validation system integration."""
        integration_module.initialize()
        
        # Test component validation
        validation_result = integration_module.validate_all()
        assert isinstance(validation_result, dict)
        assert "is_valid" in validation_result
        assert "missing_components" in validation_result
        
        # Test component readiness
        readiness_result = integration_module.ensure_components_ready()
        assert readiness_result is True  # Should be ready after initialization
    
    def test_logging_integration(self, integration_module):
        """Test logging system integration."""
        integration_module.initialize()
        
        # Test different log levels
        integration_module.log("Info message", "info")
        integration_module.log("Debug message", "debug")
        integration_module.log("Warning message", "warning")
        integration_module.log("Error message", "error")
        
        # Verify logging methods were called
        assert integration_module.log.call_count >= 4
    
    def test_colab_secrets_integration(self, integration_module):
        """Test colab secrets system integration."""
        integration_module.initialize()
        
        # Test environment detection (basic functionality)
        env_type = integration_module._detect_environment()
        assert env_type in ['colab', 'jupyter', 'local']
        
        # Test colab secrets functionality if available
        if hasattr(integration_module, 'get_colab_secret'):
            try:
                # Test getting a secret (will fail gracefully in test environment)
                result = integration_module.get_colab_secret('TEST_KEY')
                assert result is None or isinstance(result, str)
            except ImportError:
                # Expected when google.colab is not available
                pass


# ============================================================================
# Multi-Module Integration Tests
# ============================================================================

class TestMultiModuleIntegration:
    """Test integration between multiple modules."""
    
    def test_multiple_module_initialization(self):
        """Test initializing multiple modules simultaneously."""
        modules = []
        
        # Create multiple modules
        for i in range(3):
            module = FullIntegrationModule(
                module_name=f"module_{i}",
                parent_module="multi_test",
                enable_environment=False
            )
            
            # Mock logging for each module
            module.log = MagicMock()
            module.log_debug = MagicMock()
            module.log_error = MagicMock()
            
            modules.append(module)
        
        # Initialize all modules
        for module in modules:
            result = module.initialize()
            assert result is True
            assert module._is_initialized
        
        # Verify each module has its own state
        for i, module in enumerate(modules):
            assert module.module_name == f"module_{i}"
            assert module.parent_module == "multi_test"
            assert module._ui_components is not None
            
            # Modules should not interfere with each other
            config = module.get_current_config()
            assert config["module"]["name"] == f"module_{i}"
    
    def test_module_communication_patterns(self):
        """Test communication patterns between modules."""
        # Create two modules
        sender_module = FullIntegrationModule(
            module_name="sender",
            enable_environment=False
        )
        receiver_module = FullIntegrationModule(
            module_name="receiver", 
            enable_environment=False
        )
        
        # Mock logging
        for module in [sender_module, receiver_module]:
            module.log = MagicMock()
            module.log_debug = MagicMock()
            module.log_error = MagicMock()
        
        # Initialize both
        sender_module.initialize()
        receiver_module.initialize()
        
        # Test shared data pattern (simulate shared state)
        shared_data = {"messages": []}
        
        def send_message(message, **kwargs):
            shared_data["messages"].append({
                "from": "sender",
                "message": message,
                "timestamp": time.time()
            })
            return {"success": True, "sent": message}
        
        def receive_messages(**kwargs):
            messages = shared_data["messages"].copy()
            return {"success": True, "messages": messages}
        
        # Register operations
        sender_module.register_operation_handler("send", send_message)
        receiver_module.register_operation_handler("receive", receive_messages)
        
        # Test communication
        send_result = sender_module.execute_operation("send", message="Hello")
        assert send_result["success"] is True
        
        receive_result = receiver_module.execute_operation("receive")
        assert receive_result["success"] is True
        assert len(receive_result["messages"]) == 1
        assert receive_result["messages"][0]["message"] == "Hello"


# ============================================================================
# Performance Integration Tests
# ============================================================================

class TestPerformanceIntegration:
    """Performance integration tests."""
    
    def test_module_initialization_performance(self):
        """Test module initialization performance."""
        start_time = time.time()
        
        # Initialize multiple modules
        modules = []
        for i in range(10):
            module = FullIntegrationModule(
                module_name=f"perf_module_{i}",
                enable_environment=False
            )
            module.log = MagicMock()
            module.log_debug = MagicMock()
            module.log_error = MagicMock()
            
            init_result = module.initialize()
            assert init_result is True
            modules.append(module)
        
        initialization_time = time.time() - start_time
        
        # Should initialize reasonably quickly (less than 5 seconds for 10 modules)
        assert initialization_time < 5.0, f"Initialization took too long: {initialization_time:.2f}s"
        
        # Clean up
        for module in modules:
            module.cleanup()
    
    def test_operation_execution_performance(self, integration_module):
        """Test operation execution performance."""
        integration_module.initialize()
        
        # Register a simple operation
        def fast_operation(**kwargs):
            return {"success": True, "data": kwargs}
        
        integration_module.register_operation_handler("fast_op", fast_operation)
        
        # Execute many operations
        start_time = time.time()
        
        for i in range(100):
            result = integration_module.execute_operation("fast_op", iteration=i)
            assert result["success"] is True
        
        execution_time = time.time() - start_time
        
        # Should execute operations efficiently (less than 2 seconds for 100 operations)
        assert execution_time < 2.0, f"Operations took too long: {execution_time:.2f}s"
    
    def test_concurrent_operations(self, integration_module):
        """Test concurrent operation execution."""
        integration_module.initialize()
        
        results = []
        errors = []
        
        def concurrent_operation(thread_id, **kwargs):
            time.sleep(0.01)  # Simulate some work
            return {"success": True, "thread_id": thread_id, "timestamp": time.time()}
        
        integration_module.register_operation_handler("concurrent_op", concurrent_operation)
        
        def worker_thread(thread_id):
            try:
                result = integration_module.execute_operation("concurrent_op", thread_id=thread_id)
                results.append(result)
            except Exception as e:
                errors.append({"thread_id": thread_id, "error": e})
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 5
        
        # Verify all threads completed successfully
        thread_ids = [r["thread_id"] for r in results]
        assert set(thread_ids) == {0, 1, 2, 3, 4}


# ============================================================================
# Error Handling Integration Tests
# ============================================================================

class TestErrorHandlingIntegration:
    """Error handling integration tests."""
    
    def test_error_propagation_through_stack(self, integration_module):
        """Test error propagation through the module stack."""
        integration_module.initialize()
        
        def deep_failing_operation(**kwargs):
            raise ValueError("Deep error")
        
        def middle_operation(**kwargs):
            return deep_failing_operation(**kwargs)
        
        integration_module.register_operation_handler("deep_op", deep_failing_operation)
        integration_module.register_operation_handler("middle_op", middle_operation)
        
        # Test direct error
        result = integration_module.execute_operation("deep_op")
        assert result["success"] is False
        assert "Deep error" in result["error"]
        
        # Test error through middle layer
        result = integration_module.execute_operation("middle_op")
        assert result["success"] is False
        assert "Deep error" in result["error"]
    
    def test_error_recovery_mechanisms(self, integration_module):
        """Test error recovery mechanisms."""
        integration_module.initialize()
        
        failure_count = 0
        
        def unreliable_operation(**kwargs):
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= 2:
                raise ConnectionError(f"Failure {failure_count}")
            
            return {"success": True, "attempt": failure_count}
        
        integration_module.register_operation_handler("unreliable_op", unreliable_operation)
        
        # First two calls should fail
        result1 = integration_module.execute_operation("unreliable_op")
        assert result1["success"] is False
        
        result2 = integration_module.execute_operation("unreliable_op")
        assert result2["success"] is False
        
        # Third call should succeed
        result3 = integration_module.execute_operation("unreliable_op")
        assert result3["success"] is True
        assert result3["attempt"] == 3
    
    def test_error_context_preservation(self, integration_module):
        """Test that error context is preserved."""
        integration_module.initialize()
        
        def contextual_error_operation(context="default", **kwargs):
            raise RuntimeError(f"Error in context: {context}")
        
        integration_module.register_operation_handler("context_error", contextual_error_operation)
        
        # Test with different contexts
        contexts = ["user_action", "background_task", "initialization"]
        
        for context in contexts:
            result = integration_module.execute_operation("context_error", context=context)
            assert result["success"] is False
            assert f"Error in context: {context}" in result["error"]


# ============================================================================
# State Management Integration Tests  
# ============================================================================

class TestStateManagementIntegration:
    """State management integration tests."""
    
    def test_module_state_consistency(self, integration_module):
        """Test module state consistency."""
        # Initial state
        assert not integration_module._is_initialized
        assert integration_module._ui_components is None or len(integration_module._ui_components) == 0
        
        # Initialize
        integration_module.initialize()
        assert integration_module._is_initialized
        assert integration_module._ui_components is not None
        
        # Perform operations
        integration_module.register_operation_handler(
            "state_op", integration_module.test_operation_sync
        )
        
        result = integration_module.execute_operation("state_op", test="data")
        assert result["success"] is True
        
        # State should remain consistent
        assert integration_module._is_initialized
        assert integration_module._ui_components is not None
        
        # Clean up
        integration_module.cleanup()
        assert not integration_module._is_initialized
    
    def test_configuration_state_persistence(self, integration_module):
        """Test configuration state persistence."""
        integration_module.initialize()
        
        # Get initial config
        initial_config = integration_module.get_current_config()
        
        # Simulate config change
        initial_config["test_key"] = "test_value"
        
        # Config handler should maintain state
        current_config = integration_module.get_current_config()
        # The exact behavior depends on how the mock is set up
        assert isinstance(current_config, dict)
    
    def test_operation_state_isolation(self, integration_module):
        """Test that operations maintain isolated state."""
        integration_module.initialize()
        
        # Register operations that modify internal state
        def stateful_operation_1(value, **kwargs):
            integration_module.test_state_1 = value
            return {"success": True, "set": "state_1", "value": value}
        
        def stateful_operation_2(value, **kwargs):
            integration_module.test_state_2 = value
            return {"success": True, "set": "state_2", "value": value}
        
        integration_module.register_operation_handler("state_1", stateful_operation_1)
        integration_module.register_operation_handler("state_2", stateful_operation_2)
        
        # Execute operations
        result1 = integration_module.execute_operation("state_1", value="value1")
        result2 = integration_module.execute_operation("state_2", value="value2")
        
        assert result1["success"] is True
        assert result2["success"] is True
        
        # States should be isolated
        assert integration_module.test_state_1 == "value1"
        assert integration_module.test_state_2 == "value2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])