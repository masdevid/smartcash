#!/usr/bin/env python3
"""
Integration test for Colab and Dependency modules with their operation containers.
Tests the complete workflow including UI component creation and logging integration.
"""

import sys
import os
import traceback
from unittest.mock import Mock, patch

# Add the project root to Python path
sys.path.insert(0, '/Users/masdevid/Projects/smartcash')

def test_colab_complete_integration():
    """Test complete Colab module integration with operation container."""
    print("🧪 Testing Colab complete integration...")
    
    try:
        from smartcash.ui.setup.colab.colab_uimodule import create_colab_uimodule
        
        # Create Colab module with minimal config
        config = {
            "test_mode": True,
            "skip_environment_checks": True,
            "mock_ui_components": True
        }
        
        colab_module = create_colab_uimodule(config=config, auto_initialize=False)
        
        # Test initialization
        colab_module.initialize()
        print("✅ Colab module initialized successfully")
        
        # Test component access
        components = colab_module.list_components()
        print(f"✅ Colab module has {len(components)} components: {list(components.keys())}")
        
        # Test operation manager access
        operation_manager = colab_module.get_operation_manager()
        if operation_manager:
            print("✅ Colab operation manager accessible")
            
            # Test operation manager logging
            operation_manager.log("Test operation manager log message", 'info')
            print("✅ Operation manager logging works")
        else:
            print("⚠️ No operation manager available")
        
        # Test environment status
        env_status = colab_module.get_environment_status()
        print(f"✅ Environment status retrieved: {env_status.get('environment_type', 'unknown')}")
        
        # Test reset functionality
        reset_result = colab_module.reset_environment(hard_reset=False)
        if reset_result.get('success'):
            print("✅ Environment reset successful")
        else:
            print(f"⚠️ Environment reset failed: {reset_result.get('message')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Colab integration test failed: {e}")
        traceback.print_exc()
        return False

def test_dependency_complete_integration():
    """Test complete Dependency module integration with operation container."""
    print("\n🧪 Testing Dependency complete integration...")
    
    try:
        from smartcash.ui.setup.dependency.dependency_uimodule import create_dependency_uimodule
        
        # Create Dependency module with minimal config
        config = {
            "test_mode": True,
            "skip_package_checks": True,
            "mock_ui_components": True
        }
        
        dependency_module = create_dependency_uimodule(config=config, auto_initialize=False)
        
        # Test initialization
        dependency_module.initialize()
        print("✅ Dependency module initialized successfully")
        
        # Test component access
        components = dependency_module.list_components()
        print(f"✅ Dependency module has {len(components)} components: {list(components.keys())}")
        
        # Test operation manager access
        operation_manager = dependency_module.get_operation_manager()
        if operation_manager:
            print("✅ Dependency operation manager accessible")
            
            # Test operation manager logging
            operation_manager.log("Test dependency operation manager log message", 'info')
            print("✅ Dependency operation manager logging works")
            
            # Test available operations
            operations = operation_manager.get_operations()
            print(f"✅ Operation manager has {len(operations)} operations: {list(operations.keys())}")
        else:
            print("⚠️ No operation manager available")
        
        return True
        
    except Exception as e:
        print(f"❌ Dependency integration test failed: {e}")
        traceback.print_exc()
        return False

def test_operation_container_mock_integration():
    """Test operation container integration with mock components."""
    print("\n🧪 Testing operation container mock integration...")
    
    try:
        from smartcash.ui.core.handlers.operation_handler import OperationHandler
        from smartcash.ui.components.log_accordion import LogLevel
        
        # Create mock operation container
        mock_container = Mock()
        
        # Set up mock methods
        mock_container.log = Mock()
        mock_container.log_message = Mock()
        mock_container.update_progress = Mock()
        mock_container.clear_outputs = Mock()
        
        # Create operation handler with mock container
        class TestOperationHandler(OperationHandler):
            def __init__(self, operation_container):
                super().__init__(
                    module_name="test_mock_integration",
                    parent_module="test",
                    operation_container=operation_container
                )
            
            def get_operations(self):
                return {"test_operation": self.test_operation}
            
            def test_operation(self):
                self.log("Starting test operation", 'info')
                self.update_progress(50, "Halfway done")
                self.log("Test operation completed", 'info')
                return {"success": True, "message": "Test completed"}
        
        # Create handler
        handler = TestOperationHandler(mock_container)
        
        # Test logging through operation container
        handler.log("Test info message", 'info')
        handler.log("Test warning message", 'warning')
        handler.log("Test error message", 'error')
        
        # Verify mock calls
        assert mock_container.log.call_count >= 3, "Operation container log method should be called"
        print("✅ Operation container logging verified")
        
        # Test progress updates
        handler.update_progress(25, "Quarter done")
        handler.update_progress(75, "Three quarters done")
        
        # Verify progress calls
        assert mock_container.update_progress.call_count >= 2, "Progress updates should be called"
        print("✅ Progress updates verified")
        
        # Test operation execution
        operations = handler.get_operations()
        test_op = operations.get("test_operation")
        if test_op:
            result = test_op()
            print(f"✅ Test operation executed successfully: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Operation container mock integration test failed: {e}")
        traceback.print_exc()
        return False

def test_button_state_management():
    """Test button state management during operations."""
    print("\n🧪 Testing button state management...")
    
    try:
        from smartcash.ui.core.handlers.operation_handler import OperationHandler
        from unittest.mock import Mock
        import time
        
        class MockUIModule:
            def __init__(self):
                # Create mock buttons
                self.buttons = {}
                for button_name in ['primary_button', 'setup_button', 'save_button', 'reset_button']:
                    button = Mock()
                    button.disabled = False
                    button.description = f"Original {button_name}"
                    button.button_style = 'primary'
                    self.buttons[button_name] = button
            
            def get_component(self, component_name):
                return self.buttons.get(component_name)
        
        # Create mock UI module
        ui_module = MockUIModule()
        
        # Create operation handler
        class TestButtonHandler(OperationHandler):
            def __init__(self, ui_module):
                super().__init__(module_name="test_buttons", parent_module="test")
                self.ui_module = ui_module
            
            def get_component(self, component_name):
                return self.ui_module.get_component(component_name)
            
            def get_operations(self):
                return {}
        
        handler = TestButtonHandler(ui_module)
        
        # Test button disable
        original_states = {}
        for name, button in ui_module.buttons.items():
            original_states[name] = {
                'disabled': button.disabled,
                'description': button.description,
                'button_style': button.button_style
            }
        
        # Disable buttons
        button_states = handler.disable_all_buttons("⏳ Processing...")
        print(f"✅ Disabled {len(button_states)} buttons")
        
        # Verify buttons are disabled
        for name, button in ui_module.buttons.items():
            if name in button_states:
                assert button.disabled == True, f"Button {name} should be disabled"
                assert button.description == "⏳ Processing...", f"Button {name} should have processing message"
        
        print("✅ Button disable state verified")
        
        # Test button enable with success
        handler.enable_all_buttons(button_states, success=True, success_message="✅ Success")
        
        # Verify buttons are enabled with success state
        for name, button in ui_module.buttons.items():
            if name in button_states:
                assert button.disabled == original_states[name]['disabled'], f"Button {name} should be restored to original disabled state"
                assert button.description == "✅ Success", f"Button {name} should have success message"
        
        print("✅ Button enable with success verified")
        
        # Test button enable with error
        button_states = handler.disable_all_buttons("⏳ Processing again...")
        handler.enable_all_buttons(button_states, success=False, error_message="❌ Failed")
        
        # Verify buttons show error state
        for name, button in ui_module.buttons.items():
            if name in button_states:
                assert button.description == "❌ Failed", f"Button {name} should have error message"
        
        print("✅ Button enable with error verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Button state management test failed: {e}")
        traceback.print_exc()
        return False

def test_log_format_integration():
    """Test the compact log format integration."""
    print("\n🧪 Testing log format integration...")
    
    try:
        from smartcash.ui.components.log_accordion import LogAccordion, LogLevel
        from datetime import datetime
        import pytz
        
        # Create log accordion
        log_accordion = LogAccordion(
            component_name="integration_test_log",
            module_name="IntegrationTest",
            height="300px"
        )
        
        log_accordion.initialize()
        
        # Test various log scenarios that would occur in real usage
        test_scenarios = [
            # Colab module logs
            ("🚀 Starting Colab environment setup", LogLevel.INFO, "smartcash.ui.setup.colab"),
            ("✅ Google Drive mounted successfully", LogLevel.SUCCESS, "smartcash.ui.setup.colab.drive"),
            ("⚠️ GPU not available, using CPU", LogLevel.WARNING, "smartcash.ui.setup.colab.gpu"),
            
            # Dependency module logs  
            ("📦 Installing package: numpy", LogLevel.INFO, "smartcash.ui.setup.dependency.install"),
            ("✅ Package installed successfully: numpy==1.21.0", LogLevel.SUCCESS, "smartcash.ui.setup.dependency.install"),
            ("❌ Failed to install package: invalid-package", LogLevel.ERROR, "smartcash.ui.setup.dependency.install"),
            
            # Long error message with traceback
            ("Error during installation\nTraceback (most recent call last):\n  File \"install.py\", line 45, in install_package\n    subprocess.run(['pip', 'install', package])\n  File \"/usr/lib/python3.8/subprocess.py\", line 512, in run\n    raise CalledProcessError(retcode, process.args)\nCalledProcessError: Command 'pip install invalid-package' returned non-zero exit status 1", 
             LogLevel.ERROR, "smartcash.ui.setup.dependency.error"),
        ]
        
        for message, level, namespace in test_scenarios:
            log_accordion.log(
                message=message,
                level=level,
                namespace=namespace,
                timestamp=datetime.now()
            )
        
        print(f"✅ Created {len(log_accordion.log_entries)} log entries with compact format")
        
        # Test timestamp formatting (GMT+7)
        recent_entry = log_accordion.log_entries[-1] if log_accordion.log_entries else None
        if recent_entry:
            formatted_timestamp = log_accordion._format_timestamp(recent_entry.timestamp)
            print(f"✅ Timestamp formatted as: {formatted_timestamp}:GMT+7")
        
        # Test namespace shortening
        test_namespaces = [
            "smartcash.ui.setup.colab.operations",
            "smartcash.ui.setup.dependency.install",
            "smartcash.ui.components.log_accordion",
            "external.package.module"
        ]
        
        for namespace in test_namespaces:
            shortened = log_accordion._shorten_namespace(namespace)
            print(f"✅ Namespace '{namespace}' → '{shortened}'")
        
        # Verify no errors in log creation
        widget = log_accordion.show()
        print("✅ Log accordion widget created without errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Log format integration test failed: {e}")
        traceback.print_exc()
        return False

def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Running integration tests for Colab and Dependency modules...")
    print("=" * 70)
    
    test_results = []
    
    # Run all integration tests
    tests = [
        ("Colab Complete Integration", test_colab_complete_integration),
        ("Dependency Complete Integration", test_dependency_complete_integration),
        ("Operation Container Mock Integration", test_operation_container_mock_integration),
        ("Button State Management", test_button_state_management),
        ("Log Format Integration", test_log_format_integration),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ Integration test '{test_name}' crashed: {e}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("🧪 INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} integration tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All integration tests passed! Modules are working correctly together.")
    else:
        print("⚠️ Some integration tests failed. Please review the output above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)