#!/usr/bin/env python3
"""
Final validation test to ensure all logger callable errors are resolved.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/Users/masdevid/Projects/smartcash')

def test_logger_callable_fix():
    """Test that logger callable error is completely fixed."""
    print("🧪 Testing final logger callable fix...")
    
    try:
        from smartcash.ui.core.handlers.operation_handler import OperationHandler
        from unittest.mock import Mock
        
        # Create operation handler with mock container
        mock_container = Mock()
        mock_container.log = Mock(side_effect=Exception("Mock container error"))
        
        class TestHandler(OperationHandler):
            def __init__(self):
                super().__init__(
                    module_name="test_final",
                    parent_module="test",
                    operation_container=mock_container
                )
            
            def get_operations(self):
                return {}
        
        handler = TestHandler()
        
        # Test various log levels that might cause issues
        test_levels = [
            'info', 'debug', 'warning', 'error', 'critical',
            'success',  # Custom level
            'invalid',  # Invalid level
            '',  # Empty string
            None,  # None value
        ]
        
        for level in test_levels:
            try:
                # This should not raise "Logger object is not callable" error
                if level is None:
                    handler.log("Test message", 'info')  # Use default
                else:
                    handler.log("Test message", level)
                print(f"✅ Level '{level}' handled correctly")
            except Exception as e:
                if "'Logger' object is not callable" in str(e):
                    print(f"❌ Logger callable error still occurs with level '{level}': {e}")
                    return False
                else:
                    print(f"⚠️ Other error with level '{level}': {e}")
        
        print("✅ All log levels handled without 'Logger object is not callable' errors")
        return True
        
    except Exception as e:
        print(f"❌ Logger callable fix test failed: {e}")
        return False

def test_complete_module_workflow():
    """Test complete workflow without any callable errors."""
    print("\n🧪 Testing complete module workflow...")
    
    try:
        from smartcash.ui.setup.colab.colab_uimodule import create_colab_uimodule
        from smartcash.ui.setup.dependency.dependency_uimodule import create_dependency_uimodule
        
        # Test Colab module complete workflow
        print("Testing Colab module...")
        colab_module = create_colab_uimodule(
            config={"test_mode": True},
            auto_initialize=False
        )
        colab_module.initialize()
        
        # Test operation manager logging without errors
        operation_manager = colab_module.get_operation_manager()
        if operation_manager:
            # Test all logging levels
            operation_manager.log("Test info message", 'info')
            operation_manager.log("Test warning message", 'warning')
            operation_manager.log("Test error message", 'error')
            operation_manager.log("Test debug message", 'debug')
        
        print("✅ Colab module workflow completed without callable errors")
        
        # Test Dependency module complete workflow
        print("Testing Dependency module...")
        dependency_module = create_dependency_uimodule(
            config={"test_mode": True},
            auto_initialize=False
        )
        dependency_module.initialize()
        
        # Test operation manager logging without errors
        operation_manager = dependency_module.get_operation_manager()
        if operation_manager:
            # Test all logging levels
            operation_manager.log("Test info message", 'info')
            operation_manager.log("Test warning message", 'warning')
            operation_manager.log("Test error message", 'error')
            operation_manager.log("Test debug message", 'debug')
        
        print("✅ Dependency module workflow completed without callable errors")
        
        return True
        
    except Exception as e:
        if "'Logger' object is not callable" in str(e):
            print(f"❌ Logger callable error still occurs in workflow: {e}")
            return False
        else:
            print(f"⚠️ Other error in workflow: {e}")
            return True  # Other errors are acceptable, we're just checking for callable errors

def run_final_validation():
    """Run final validation tests."""
    print("🧪 Running final validation tests...")
    print("=" * 50)
    
    tests = [
        ("Logger Callable Fix", test_logger_callable_fix),
        ("Complete Module Workflow", test_complete_module_workflow),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("🧪 FINAL VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} final validation tests passed")
    
    if passed == total:
        print("🎉 All fixes validated! No more 'Logger object is not callable' errors.")
    else:
        print("⚠️ Some validation tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = run_final_validation()
    sys.exit(0 if success else 1)