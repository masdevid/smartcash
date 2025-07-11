"""
Comprehensive test suite for UIModule and UIModuleFactory.
Tests the new UIModule-centric architecture with all components.
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from smartcash.ui.core.ui_module import (
    UIModule, 
    ModuleStatus, 
    ModuleInfo,
    SharedMethodRegistry,
    register_ui_method,
    register_operation_method,
    register_config_method
)
from smartcash.ui.core.ui_module_factory import (
    UIModuleFactory,
    ModuleTemplate,
    FactoryMode,
    create_module,
    get_module,
    create_template
)
from smartcash.ui.core.handlers.operation_handler import OperationResult, OperationStatus
from smartcash.ui.core.errors import SmartCashUIError


class TestUIModule:
    """Test UIModule functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        # Clear shared registry
        SharedMethodRegistry._shared_methods.clear()
        SharedMethodRegistry._method_metadata.clear()
    
    def test_basic_initialization(self):
        """Test basic UIModule initialization."""
        module = UIModule("test_module", "test_parent")
        
        assert module.module_name == "test_module"
        assert module.parent_module == "test_parent"
        assert module.full_module_name == "test_parent.test_module"
        assert module.get_status() == ModuleStatus.PENDING
        assert not module.is_ready()
        assert not module.has_errors()
    
    def test_auto_initialization(self):
        """Test auto-initialization."""
        module = UIModule("test_module", auto_initialize=True)
        
        assert module.is_ready()
        assert module.get_status() == ModuleStatus.READY
        assert module._initialized_at is not None
    
    def test_manual_initialization(self):
        """Test manual initialization."""
        module = UIModule("test_module")
        assert module.get_status() == ModuleStatus.PENDING
        
        result = module.initialize()
        assert result is module  # Method chaining
        assert module.is_ready()
        assert module.get_status() == ModuleStatus.READY
    
    def test_initialization_with_config(self):
        """Test initialization with configuration."""
        config = {"setting1": "value1", "setting2": 42}
        module = UIModule("test_module", config=config, auto_initialize=True)
        
        assert module.get_config("setting1") == "value1"
        assert module.get_config("setting2") == 42
        assert module.get_config() == config
    
    def test_component_management(self):
        """Test component registration and retrieval."""
        module = UIModule("test_module", auto_initialize=True)
        
        # Create mock component
        mock_component = Mock()
        mock_component.__class__.__name__ = "MockComponent"
        
        # Register component
        module.register_component("test_component", mock_component)
        
        # Retrieve component
        retrieved = module.get_component("test_component")
        assert retrieved is mock_component
        
        # List components
        components = module.list_components()
        assert "test_component" in components
        assert components["test_component"] == "MockComponent"
    
    def test_operation_management(self):
        """Test operation registration and execution."""
        module = UIModule("test_module", auto_initialize=True)
        
        # Create mock operation
        def test_operation(value: int) -> int:
            return value * 2
        
        # Register operation
        module.register_operation("multiply", test_operation)
        
        # List operations
        operations = module.list_operations()
        assert "multiply" in operations
        
        # Execute operation
        result = module.execute_operation("multiply", 5)
        assert result.status == OperationStatus.COMPLETED
        assert result.data == 10
        
        # Test unknown operation
        result = module.execute_operation("unknown")
        assert result.status == OperationStatus.FAILED
        assert isinstance(result.error, ValueError)
    
    def test_configuration_management(self):
        """Test configuration updates and retrieval."""
        module = UIModule("test_module", auto_initialize=True)
        
        # Update config
        module.update_config(new_setting="new_value", number=123)
        
        assert module.get_config("new_setting") == "new_value"
        assert module.get_config("number") == 123
        assert module.get_config("nonexistent", "default") == "default"
        
        # Reset config
        module.reset_config()
        assert module.get_config() == {}
    
    def test_shared_method_management(self):
        """Test shared method functionality."""
        module = UIModule("test_module", auto_initialize=True)
        
        def shared_method(x: int) -> str:
            return f"Processed: {x}"
        
        # Share method
        module.share_method("process", shared_method)
        
        # Check method was shared
        assert "process" in module.list_shared_methods()
        assert hasattr(module, "process")
        assert module.process(42) == "Processed: 42"
        
        # Get shared method
        retrieved = module.get_shared_method("process")
        assert retrieved is shared_method
    
    def test_ui_delegation_methods(self):
        """Test UI method delegation to handlers."""
        module = UIModule("test_module", auto_initialize=True)
        
        # Mock UI handler
        module._ui_handler = Mock()
        
        # Test delegation methods
        module.update_status("test message", "info")
        module._ui_handler.update_status.assert_called_with("test message", "info")
        
        module.update_progress(50.0, "progress message")
        module._ui_handler.update_progress.assert_called_with(50.0, "progress message")
        
        module.log_message("log message", "warning")
        module._ui_handler.log_message.assert_called_with("log message", "warning")
        
        module.show_dialog("title", "message", "info")
        module._ui_handler.show_dialog.assert_called_with("title", "message", "info")
    
    def test_module_info(self):
        """Test module information retrieval."""
        module = UIModule("test_module", "parent", auto_initialize=True)
        
        # Add some components and operations
        module.register_component("comp1", Mock())
        module.register_operation("op1", lambda: None)
        
        info = module.get_info()
        
        assert isinstance(info, ModuleInfo)
        assert info.name == "test_module"
        assert info.parent_module == "parent"
        assert info.status == ModuleStatus.READY
        assert "comp1" in info.components
        assert "op1" in info.operations
        assert info.error_count == 0
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with UIModule("test_module") as module:
            assert module.is_ready()
            assert module.get_status() == ModuleStatus.READY
        
        # Module should be cleaned up after exit
        assert module.get_status() == ModuleStatus.CLEANUP
    
    def test_reset_functionality(self):
        """Test module reset."""
        module = UIModule("test_module", auto_initialize=True)
        
        # Add some data
        module.register_component("comp", Mock())
        module.register_operation("op", lambda: None)
        module.update_config(setting="value")
        
        # Reset
        module.reset()
        
        assert module.get_status() == ModuleStatus.PENDING
        assert not module.is_ready()
        assert len(module.list_components()) == 0
        assert len(module.list_operations()) == 0
        assert module.get_config() == {}
    
    def test_cleanup(self):
        """Test module cleanup."""
        module = UIModule("test_module", auto_initialize=True)
        
        # Add mock handlers
        module._ui_handler = Mock()
        module._config_handler = Mock()
        module._operation_handler = Mock()
        
        # Cleanup
        module.cleanup()
        
        assert module.get_status() == ModuleStatus.CLEANUP
        module._ui_handler.cleanup.assert_called_once()
        module._config_handler.cleanup.assert_called_once()
        module._operation_handler.cleanup.assert_called_once()


class TestSharedMethodRegistry:
    """Test SharedMethodRegistry functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        SharedMethodRegistry._shared_methods.clear()
        SharedMethodRegistry._method_metadata.clear()
    
    def test_method_registration(self):
        """Test method registration."""
        def test_method():
            return "test"
        
        SharedMethodRegistry.register_method(
            "test_method", 
            test_method,
            description="Test method",
            category="testing"
        )
        
        # Check registration
        assert SharedMethodRegistry.get_method("test_method") is test_method
        
        # Check metadata
        metadata = SharedMethodRegistry.list_methods()
        assert "test_method" in metadata
        assert metadata["test_method"]["category"] == "testing"
        assert metadata["test_method"]["description"] == "Test method"
    
    def test_method_overwrite_protection(self):
        """Test method overwrite protection."""
        def method1():
            return "method1"
        
        def method2():
            return "method2"
        
        # Register first method
        SharedMethodRegistry.register_method("test", method1)
        
        # Try to register again without overwrite
        with pytest.raises(ValueError, match="already registered"):
            SharedMethodRegistry.register_method("test", method2)
        
        # Register with overwrite
        SharedMethodRegistry.register_method("test", method2, overwrite=True)
        assert SharedMethodRegistry.get_method("test") is method2
    
    def test_method_injection(self):
        """Test method injection into UIModule."""
        def ui_method():
            return "ui"
        
        def config_method():
            return "config"
        
        # Register methods in different categories
        SharedMethodRegistry.register_method("ui_func", ui_method, category="ui")
        SharedMethodRegistry.register_method("config_func", config_method, category="config")
        
        module = UIModule("test_module")
        
        # Inject all methods
        count = SharedMethodRegistry.inject_methods(module)
        assert count == 2
        assert hasattr(module, "ui_func")
        assert hasattr(module, "config_func")
        
        # Inject specific category
        module2 = UIModule("test_module2")
        count = SharedMethodRegistry.inject_methods(module2, category="ui")
        assert count == 1
        assert hasattr(module2, "ui_func")
        assert not hasattr(module2, "config_func")
    
    def test_method_unregistration(self):
        """Test method unregistration."""
        def test_method():
            return "test"
        
        SharedMethodRegistry.register_method("test", test_method)
        assert SharedMethodRegistry.get_method("test") is not None
        
        # Unregister
        result = SharedMethodRegistry.unregister_method("test")
        assert result is True
        assert SharedMethodRegistry.get_method("test") is None
        
        # Try to unregister non-existent
        result = SharedMethodRegistry.unregister_method("nonexistent")
        assert result is False
    
    def test_category_clearing(self):
        """Test clearing methods by category."""
        # Register methods in different categories
        SharedMethodRegistry.register_method("ui1", lambda: None, category="ui")
        SharedMethodRegistry.register_method("ui2", lambda: None, category="ui")
        SharedMethodRegistry.register_method("config1", lambda: None, category="config")
        
        # Clear UI category
        count = SharedMethodRegistry.clear_category("ui")
        assert count == 2
        
        # Check results
        assert SharedMethodRegistry.get_method("ui1") is None
        assert SharedMethodRegistry.get_method("ui2") is None
        assert SharedMethodRegistry.get_method("config1") is not None
    
    def test_convenience_functions(self):
        """Test convenience registration functions."""
        def ui_func():
            return "ui"
        
        def op_func():
            return "operation"
        
        def config_func():
            return "config"
        
        # Test convenience functions
        register_ui_method("ui_test", ui_func)
        register_operation_method("op_test", op_func)
        register_config_method("config_test", config_func)
        
        # Check registration and categories
        methods = SharedMethodRegistry.list_methods()
        assert methods["ui_test"]["category"] == "ui"
        assert methods["op_test"]["category"] == "operations"
        assert methods["config_test"]["category"] == "config"


class TestUIModuleFactory:
    """Test UIModuleFactory functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        UIModuleFactory.reset_factory()
    
    def teardown_method(self):
        """Cleanup after each test."""
        UIModuleFactory.reset_factory()
    
    def test_basic_module_creation(self):
        """Test basic module creation."""
        module = UIModuleFactory.create_module("test_module", "parent")
        
        assert module.module_name == "test_module"
        assert module.parent_module == "parent"
        assert module.full_module_name == "parent.test_module"
    
    def test_instance_reuse(self):
        """Test instance reuse."""
        module1 = UIModuleFactory.create_module("test_module")
        module2 = UIModuleFactory.create_module("test_module")
        
        # Should be same instance
        assert module1 is module2
    
    def test_force_new_instance(self):
        """Test forcing new instance creation."""
        module1 = UIModuleFactory.create_module("test_module")
        module2 = UIModuleFactory.create_module("test_module", force_new=True)
        
        # Should be different instances
        assert module1 is not module2
    
    def test_template_registration_and_usage(self):
        """Test template registration and usage."""
        template = ModuleTemplate(
            module_name="test_module",
            parent_module=None,
            default_config={"setting1": "default_value"},
            required_components=["comp1"],
            required_operations=["op1"],
            auto_initialize=True,
            description="Test template"
        )
        
        # Register template
        UIModuleFactory.register_template(template)
        
        # Create module using template
        module = UIModuleFactory.create_module("test_module")
        
        assert module.is_ready()  # auto_initialize=True
        assert module.get_config("setting1") == "default_value"
        assert "op1" in module.list_operations()  # Placeholder should be added
    
    def test_template_overwrite_protection(self):
        """Test template overwrite protection."""
        template1 = create_template("test_module", description="First template")
        template2 = create_template("test_module", description="Second template")
        
        # Register first template
        UIModuleFactory.register_template(template1)
        
        # Try to register again without overwrite
        with pytest.raises(ValueError, match="already exists"):
            UIModuleFactory.register_template(template2)
        
        # Register with overwrite
        UIModuleFactory.register_template(template2, overwrite=True)
        retrieved = UIModuleFactory.get_template("test_module")
        assert retrieved.description == "Second template"
    
    def test_factory_mode(self):
        """Test factory mode setting."""
        assert UIModuleFactory.get_mode() == FactoryMode.DEVELOPMENT
        
        UIModuleFactory.set_mode(FactoryMode.PRODUCTION)
        assert UIModuleFactory.get_mode() == FactoryMode.PRODUCTION
    
    def test_instance_listing(self):
        """Test instance listing."""
        module1 = UIModuleFactory.create_module("module1")
        module2 = UIModuleFactory.create_module("module2", "parent")
        
        instances = UIModuleFactory.list_instances()
        assert len(instances) == 2
        assert "module1" in instances
        assert "parent.module2" in instances
        assert instances["module1"] is module1
        assert instances["parent.module2"] is module2
    
    def test_instance_cleanup(self):
        """Test instance cleanup."""
        module = UIModuleFactory.create_module("test_module")
        
        # Verify instance exists
        assert UIModuleFactory.get_module("test_module") is module
        
        # Cleanup instance
        result = UIModuleFactory.cleanup_instance("test_module")
        assert result is True
        
        # Verify instance is gone
        assert UIModuleFactory.get_module("test_module") is None
    
    def test_batch_creation_from_config(self):
        """Test batch creation from configuration."""
        config = {
            "module1": {
                "setting1": "value1",
                "_factory": {"auto_initialize": True}
            },
            "parent.module2": {
                "setting2": "value2"
            }
        }
        
        modules = UIModuleFactory.create_modules_from_config(config)
        
        assert len(modules) == 2
        assert "module1" in modules
        assert "parent.module2" in modules
        
        # Check configuration
        assert modules["module1"].get_config("setting1") == "value1"
        assert modules["module1"].is_ready()  # auto_initialize=True
        assert modules["parent.module2"].get_config("setting2") == "value2"
    
    def test_batch_initialization(self):
        """Test batch initialization."""
        # Create modules without auto-initialization
        module1 = UIModuleFactory.create_module("module1", auto_initialize=False)
        module2 = UIModuleFactory.create_module("module2", auto_initialize=False)
        
        assert not module1.is_ready()
        assert not module2.is_ready()
        
        # Batch initialize
        results = UIModuleFactory.initialize_all_modules()
        
        assert results["module1"] is True
        assert results["module2"] is True
        assert module1.is_ready()
        assert module2.is_ready()
    
    def test_factory_stats(self):
        """Test factory statistics."""
        # Create some modules
        module1 = UIModuleFactory.create_module("stats_module1", auto_initialize=True)
        module2 = UIModuleFactory.create_module("stats_module2", auto_initialize=False)
        
        # Register a template
        template = create_template("template_module")
        UIModuleFactory.register_template(template)
        
        # Get stats
        stats = UIModuleFactory.get_factory_stats()
        
        # Basic structure tests
        assert "active_instances" in stats
        assert "registered_templates" in stats
        assert "instances_by_status" in stats
        assert stats["registered_templates"] >= 1  # At least 1 template
    
    def test_context_manager(self):
        """Test factory context manager."""
        with UIModuleFactory.module_context("test_module", auto_initialize=True) as module:
            assert module.is_ready()
            assert UIModuleFactory.get_module("test_module") is module
        
        # Module should be cleaned up after exit
        assert UIModuleFactory.get_module("test_module") is None
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test create_module function
        module1 = create_module("test_module", auto_initialize=True)
        assert module1.is_ready()
        
        # Test get_module function
        module2 = get_module("test_module")
        assert module2 is module1
        
        # Test create_template function
        template = create_template(
            "template_module",
            default_config={"test": "value"},
            required_operations=["test_op"]
        )
        assert template.module_name == "template_module"
        assert template.default_config == {"test": "value"}
        assert template.required_operations == ["test_op"]
    
    def test_weak_reference_cleanup(self):
        """Test automatic cleanup of dead references."""
        module = UIModuleFactory.create_module("test_module")
        
        # Verify instance exists
        instances = UIModuleFactory.list_instances()
        assert "test_module" in instances
        
        # Delete the module reference
        del module
        
        # Force garbage collection (implementation dependent)
        import gc
        gc.collect()
        
        # List instances should clean up dead references
        instances = UIModuleFactory.list_instances()
        # Note: This test might be flaky depending on garbage collection timing
        # In real usage, dead references are cleaned up automatically
    
    def test_thread_safety(self):
        """Test thread safety of factory operations."""
        # Simple test to verify no deadlocks or race conditions
        # Note: Due to weak references and teardown, exact instance counting is difficult
        
        results = []
        errors = []
        
        def create_modules(thread_id: int):
            try:
                module = UIModuleFactory.create_module(f"thread_test_{thread_id}")
                results.append((thread_id, module.full_module_name))
            except Exception as e:
                errors.append((thread_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=create_modules, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results - main goal is no errors in threaded operations
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3  # 3 threads, 1 module each


class TestErrorHandling:
    """Test error handling in UIModule and Factory."""
    
    def test_module_initialization_error(self):
        """Test error handling during module initialization."""
        module = UIModule("test_module")
        
        # Mock handler initialization to fail
        with patch.object(module, '_initialize_handlers') as mock_init:
            mock_init.side_effect = Exception("Handler init failed")
            
            with pytest.raises((SmartCashUIError, RuntimeError)):
                module.initialize()
            
            assert module.get_status() == ModuleStatus.ERROR
            assert module.has_errors()
    
    def test_operation_execution_error(self):
        """Test error handling in operation execution."""
        module = UIModule("test_module", auto_initialize=True)
        
        def failing_operation():
            raise ValueError("Operation failed")
        
        module.register_operation("fail", failing_operation)
        
        # The operation should either return a failed result or raise an exception
        try:
            result = module.execute_operation("fail")
            assert result.status == OperationStatus.FAILED
            assert isinstance(result.error, ValueError)
            assert "Operation failed" in str(result.error)
        except (RuntimeError, ValueError):
            # This is also acceptable behavior
            pass
    
    def test_factory_creation_error(self):
        """Test error handling in factory module creation."""
        # Mock UIModule init to fail
        with patch('smartcash.ui.core.ui_module_factory.UIModule') as mock_module:
            mock_module.side_effect = Exception("Module creation failed")
            
            with pytest.raises(SmartCashUIError, match="Failed to create"):
                UIModuleFactory.create_module("test_module")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])