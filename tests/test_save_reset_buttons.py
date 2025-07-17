#!/usr/bin/env python3
"""
Test script to verify save/reset button functionality and log containment
"""

import sys
import os
import io
import contextlib

# Add the project root to Python path
project_root = '/Users/masdevid/Projects/smartcash'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_save_reset_buttons():
    """Test that save/reset buttons provide proper status updates."""
    print("🧪 Testing Save/Reset Button Functionality...")
    
    try:
        # Test 1: Dependency module save/reset buttons
        print("\n1. Testing Dependency module save/reset buttons...")
        from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
        
        dep_module = DependencyUIModule()
        dep_module.initialize()
        
        # Test save button
        print("   Testing save button...")
        save_result = dep_module._handle_save_config()
        print(f"   Save result: {save_result}")
        
        # Test reset button
        print("   Testing reset button...")
        reset_result = dep_module._handle_reset_config()
        print(f"   Reset result: {reset_result}")
        
        # Test 2: Colab module save/reset buttons
        print("\n2. Testing Colab module save/reset buttons...")
        from smartcash.ui.setup.colab.colab_uimodule import ColabUIModule
        
        colab_module = ColabUIModule()
        colab_module.initialize()
        
        # Test save button
        print("   Testing save button...")
        save_result = colab_module._handle_save_config()
        print(f"   Save result: {save_result}")
        
        # Test reset button
        print("   Testing reset button...")
        reset_result = colab_module._handle_reset_config()
        print(f"   Reset result: {reset_result}")
        
        # Test 3: Split module save/reset buttons
        print("\n3. Testing Split module save/reset buttons...")
        from smartcash.ui.dataset.split.split_uimodule import SplitUIModule
        
        split_module = SplitUIModule()
        split_module.initialize()
        
        # Test save button
        print("   Testing save button...")
        save_result = split_module._handle_save_config()
        print(f"   Save result: {save_result}")
        
        # Test reset button
        print("   Testing reset button...")
        reset_result = split_module._handle_reset_config()
        print(f"   Reset result: {reset_result}")
        
        print("\n🎉 Save/Reset button testing completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_log_containment_dependency():
    """Test that dependency module button clicks don't leak logs."""
    print("\n🧪 Testing Dependency Module Log Containment...")
    
    try:
        from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
        
        console_output = io.StringIO()
        with contextlib.redirect_stdout(console_output), contextlib.redirect_stderr(console_output):
            dep_module = DependencyUIModule()
            dep_module.initialize()
            
            # Try to trigger the install button (this was causing the log)
            dep_module._handle_install_packages()
        
        captured_output = console_output.getvalue()
        
        # Check for problematic log messages
        problematic_logs = [
            "[Status] Tidak ada paket yang dipilih untuk diinstal",
            "Failed to get selected packages"
        ]
        
        found_issues = []
        for log_msg in problematic_logs:
            if log_msg in captured_output:
                found_issues.append(log_msg)
        
        if found_issues:
            print(f"   ❌ Found problematic logs: {found_issues}")
            print(f"   Console output: {captured_output}")
        else:
            print("   ✅ No problematic logs in console output")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_save_reset_buttons()
    test_log_containment_dependency()