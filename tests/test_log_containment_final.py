#!/usr/bin/env python3
"""
Final test script to verify all log containment fixes
"""

import sys
import os
import io
import contextlib

# Add the project root to Python path
project_root = '/Users/masdevid/Projects/smartcash'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_log_containment_fixes():
    """Test that all logs are properly contained within operation containers."""
    print("🧪 Testing Final Log Containment Fixes...")
    
    try:
        # Test 1: Colab module log containment
        print("\n1. Testing Colab module log containment...")
        from smartcash.ui.setup.colab.colab_uimodule import initialize_colab_ui
        
        console_output = io.StringIO()
        with contextlib.redirect_stdout(console_output), contextlib.redirect_stderr(console_output):
            result = initialize_colab_ui(config=None, display=False)
        
        captured_output = console_output.getvalue()
        
        # Check for specific problematic log messages
        problematic_logs = [
            "✅ UI displayed successfully",
            "🔄 Starting",
            "Status update:",
            "[Status]"
        ]
        
        found_issues = []
        for log_msg in problematic_logs:
            if log_msg in captured_output:
                found_issues.append(log_msg)
        
        if found_issues:
            print(f"   ❌ Found problematic logs: {found_issues}")
            print(f"   Console output preview: {captured_output[:500]}...")
        else:
            print("   ✅ No problematic logs in console output")
        
        # Test 2: Dependency module log containment
        print("\n2. Testing Dependency module log containment...")
        from smartcash.ui.setup.dependency.dependency_uimodule import initialize_dependency_ui
        
        console_output = io.StringIO()
        with contextlib.redirect_stdout(console_output), contextlib.redirect_stderr(console_output):
            result = initialize_dependency_ui(config=None, display=False)
        
        captured_output = console_output.getvalue()
        
        found_issues = []
        for log_msg in problematic_logs:
            if log_msg in captured_output:
                found_issues.append(log_msg)
        
        if found_issues:
            print(f"   ❌ Found problematic logs: {found_issues}")
            print(f"   Console output preview: {captured_output[:500]}...")
        else:
            print("   ✅ No problematic logs in console output")
        
        # Test 3: Split module parameter conflict (if available)
        print("\n3. Testing Split module parameter conflict fix...")
        try:
            from smartcash.ui.dataset.split import initialize_split_ui
            
            # This should not cause a parameter conflict
            console_output = io.StringIO()
            with contextlib.redirect_stdout(console_output), contextlib.redirect_stderr(console_output):
                result = initialize_split_ui(display=True)  # This was causing the conflict
            
            print("   ✅ No parameter conflict with display=True")
            
            # Check for logs in split module too
            captured_output = console_output.getvalue()
            found_issues = []
            for log_msg in problematic_logs:
                if log_msg in captured_output:
                    found_issues.append(log_msg)
            
            if found_issues:
                print(f"   ❌ Found problematic logs in split: {found_issues}")
            else:
                print("   ✅ No problematic logs in split module")
                
        except TypeError as e:
            if "multiple values" in str(e):
                print(f"   ❌ Parameter conflict still exists: {e}")
            else:
                print(f"   ⚠️ Different error (not parameter conflict): {e}")
        except Exception as e:
            print(f"   ⚠️ Split module test failed: {e}")
        
        print("\n🎉 Final log containment testing completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_log_containment_fixes()