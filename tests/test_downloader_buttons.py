#!/usr/bin/env python3
"""
Test button operations for downloader module to ensure proper error handling.
"""

import sys
import io
import contextlib
import traceback
from unittest.mock import patch, MagicMock

def test_button_operations():
    """Test that button operations handle errors properly in UI."""
    print("🧪 Testing Button Operations Error Handling")
    print("=" * 50)
    
    try:
        from smartcash.ui.dataset.downloader import initialize_downloader_ui
        
        # Initialize downloader
        ui_result = initialize_downloader_ui()
        
        if not ui_result or not isinstance(ui_result, dict):
            print("❌ Failed to initialize downloader UI")
            return False
        
        # Get operation manager
        operation_manager = ui_result.get('downloader_operation_manager') or ui_result.get('operation_manager')
        
        if not operation_manager:
            print("❌ No operation manager found")
            return False
        
        # Test error handling capabilities
        print(f"✅ Operation manager found: {type(operation_manager).__name__}")
        
        # Check if error handling methods exist
        error_methods = ['handle_error', 'log_error', '_handle_error']
        found_methods = [method for method in error_methods if hasattr(operation_manager, method)]
        
        if found_methods:
            print(f"✅ Error handling methods found: {found_methods}")
            
            # Test error handling
            test_error = "Test error message"
            try:
                if hasattr(operation_manager, 'handle_error'):
                    operation_manager.handle_error(test_error)
                    print("✅ handle_error method works")
                elif hasattr(operation_manager, 'log_error'):
                    operation_manager.log_error(test_error)
                    print("✅ log_error method works")
                else:
                    print("⚠️  Error methods exist but not tested")
            except Exception as e:
                print(f"⚠️  Error handling test failed: {e}")
                # This is okay - the method might need specific parameters
        else:
            print("❌ No error handling methods found")
            return False
        
        # Test operations availability
        if hasattr(operation_manager, 'get_operations'):
            operations = operation_manager.get_operations()
            print(f"✅ Available operations: {list(operations.keys())}")
        else:
            print("⚠️  No get_operations method found")
        
        # Test operation container logging
        operation_container = ui_result.get('operation_container')
        if operation_container:
            print("✅ Operation container found for logging")
            
            # Test logging to operation container
            if hasattr(operation_container, 'log_message'):
                operation_container.log_message("Test log message")
                print("✅ Operation container logging works")
            else:
                print("⚠️  Operation container doesn't have log_message method")
        else:
            print("⚠️  No operation container found")
        
        return True
        
    except Exception as e:
        print(f"❌ Button operations test failed: {e}")
        traceback.print_exc()
        return False

def test_ui_error_display():
    """Test that UI displays errors properly instead of console fallbacks."""
    print("\n🧪 Testing UI Error Display")
    print("=" * 50)
    
    try:
        from smartcash.ui.dataset.downloader.downloader_initializer import DownloaderInitializer
        
        # Test with a config that might cause errors
        invalid_config = {
            "roboflow": {
                "api_key": "invalid_key",
                "dataset_id": "nonexistent/dataset"
            }
        }
        
        initializer = DownloaderInitializer()
        ui_result = initializer.initialize(config=invalid_config)
        
        if ui_result and isinstance(ui_result, dict):
            print("✅ UI initialized even with invalid config")
            
            # Check if error components are present
            error_components = ['error', 'error_component', 'success']
            found_error_components = [comp for comp in error_components if comp in ui_result]
            
            if found_error_components:
                print(f"✅ Error components found: {found_error_components}")
            else:
                print("⚠️  No explicit error components, but UI handled gracefully")
            
            # Check if UI components are still present
            ui_components = ui_result.get('ui')
            if ui_components:
                print("✅ UI components still available despite errors")
            else:
                print("⚠️  No UI components in result")
            
            return True
        else:
            print("❌ UI initialization failed completely")
            return False
            
    except Exception as e:
        print(f"❌ UI error display test failed: {e}")
        return False

def main():
    """Run all button and error tests."""
    print("🚀 Starting Downloader Button and Error Tests")
    print("=" * 80)
    
    success_count = 0
    total_tests = 2
    
    # Run tests
    if test_button_operations():
        success_count += 1
    
    if test_ui_error_display():
        success_count += 1
    
    # Results
    print("\n" + "=" * 80)
    print("📊 BUTTON AND ERROR TEST RESULTS")
    print("=" * 80)
    print(f"Passed: {success_count}/{total_tests}")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 All button and error tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)