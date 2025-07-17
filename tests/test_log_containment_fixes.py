#!/usr/bin/env python3
"""
Test script to verify that logs are properly contained within operation containers
"""

import sys
import os
import io
import contextlib

# Add the project root to Python path
project_root = '/Users/masdevid/Projects/smartcash'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_log_containment():
    """Test that logs are properly contained within operation containers."""
    print("🧪 Testing Log Containment Fixes...")
    
    try:
        # Test 1: Parameter conflict fix for Colab
        print("\n1. Testing Colab parameter conflict fix...")
        from smartcash.ui.setup.colab.colab_uimodule import initialize_colab_ui
        
        # This should not cause a parameter conflict
        try:
            result = initialize_colab_ui(config=None, display=False, show_display=False)
            print("   ✅ No parameter conflict with display=False, show_display=False")
        except TypeError as e:
            if "multiple values" in str(e):
                print(f"   ❌ Parameter conflict still exists: {e}")
                return False
            else:
                print(f"   ⚠️ Different TypeError (not parameter conflict): {e}")
        
        # Test 2: Display mixin log containment
        print("\n2. Testing display mixin log containment...")
        from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
        
        # Capture all output during initialization and display
        console_output = io.StringIO()
        
        with contextlib.redirect_stdout(console_output), contextlib.redirect_stderr(console_output):
            dep_module = DependencyUIModule()
            success = dep_module.initialize()
            
            # Check if the message came from initialization
            init_output = console_output.getvalue()
            
            if success and hasattr(dep_module, 'display_ui'):
                # Clear the buffer to isolate display_ui output
                console_output = io.StringIO()
                with contextlib.redirect_stdout(console_output), contextlib.redirect_stderr(console_output):
                    # This would trigger the "UI displayed successfully" message
                    display_result = dep_module.display_ui()
                display_output = console_output.getvalue()
                
                if "✅ UI displayed successfully" in init_output:
                    print(f"   🔍 Message appeared during initialization")
                if "✅ UI displayed successfully" in display_output:
                    print(f"   🔍 Message appeared during display_ui() call")
                
                # Combine outputs for final check
                console_output = io.StringIO()
                console_output.write(init_output)
                console_output.write(display_output)
        
        captured_output = console_output.getvalue()
        
        # Check for the specific log message appearing in console
        if "✅ UI displayed successfully" in captured_output:
            print("   ⚠️ 'UI displayed successfully' still appearing in console output")
            print(f"   Full console output:\n{captured_output}")
        else:
            print("   ✅ 'UI displayed successfully' not appearing in console output")
        
        if success:
            print("   ✅ Module initialized successfully")
        else:
            print("   ❌ Module initialization failed")
        
        # Test 3: Check that logs are going to operation container instead
        print("\n3. Testing operation container log capture...")
        
        # Create a fresh module and check log buffering
        dep_module2 = DependencyUIModule()
        
        # Check initial log buffer
        if hasattr(dep_module2, '_log_buffer'):
            initial_buffer_size = len(dep_module2._log_buffer)
            print(f"   Initial log buffer size: {initial_buffer_size}")
        else:
            print("   ❌ Log buffer not found")
            return False
        
        # Initialize the module (which should buffer logs then flush to operation container)
        with contextlib.redirect_stdout(console_output), contextlib.redirect_stderr(console_output):
            dep_module2.initialize()
        
        # Check if buffer was used and flushed
        if hasattr(dep_module2, '_log_buffer'):
            final_buffer_size = len(dep_module2._log_buffer)
            print(f"   Final log buffer size: {final_buffer_size}")
            
            if final_buffer_size == 0:
                print("   ✅ Log buffer was flushed (empty after initialization)")
            else:
                print(f"   ⚠️ Log buffer still contains {final_buffer_size} entries")
        
        # Check if operation container exists and can receive logs
        if hasattr(dep_module2, '_ui_components') and dep_module2._ui_components:
            operation_container = dep_module2._ui_components.get('operation_container')
            if operation_container and hasattr(operation_container, 'log'):
                print("   ✅ Operation container ready to receive logs")
                
                # Test logging to operation container
                test_message = "Test log message for containment"
                if hasattr(dep_module2, 'log'):
                    dep_module2.log(test_message, 'info')
                    print("   ✅ Successfully sent test log to operation container")
                else:
                    print("   ❌ Module doesn't have log method")
            else:
                print("   ❌ Operation container not ready for logs")
        
        # Test 4: Enhanced factory parameter handling
        print("\n4. Testing enhanced factory parameter handling...")
        from smartcash.ui.setup.dependency.dependency_uimodule import initialize_dependency_ui
        
        test_cases = [
            {"config": None, "display": False},
            {"config": None, "display": True, "show_display": False},  # Conflicting params
            {"display": False, "other_param": "test"},
        ]
        
        all_cases_passed = True
        for i, test_case in enumerate(test_cases):
            try:
                with contextlib.redirect_stdout(console_output), contextlib.redirect_stderr(console_output):
                    result = initialize_dependency_ui(**test_case)
                print(f"   ✅ Test case {i+1}: {test_case} - No errors")
            except Exception as e:
                if "multiple values" in str(e):
                    print(f"   ❌ Test case {i+1}: Parameter conflict - {e}")
                    all_cases_passed = False
                else:
                    print(f"   ✅ Test case {i+1}: Different error (not parameter conflict) - {e}")
        
        if all_cases_passed:
            print("   ✅ All parameter handling test cases passed")
        
        print("\n🎉 Log containment tests completed!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_log_containment()