#!/usr/bin/env python3
"""
Test script to reproduce the dependency install issue from development_logs.txt
"""

import sys
import os
import io
import contextlib

# Add the project root to Python path
project_root = '/Users/masdevid/Projects/smartcash'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_dependency_install_button():
    """Test dependency install button to reproduce the log issue."""
    print("🧪 Testing Dependency Install Button...")
    
    try:
        from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
        
        # Create and initialize module
        dep_module = DependencyUIModule()
        dep_module.initialize()
        
        print("✅ Module initialized successfully")
        
        # Test install button click (this should trigger the warning about no packages selected)
        print("🔄 Testing install button click...")
        
        console_output = io.StringIO()
        with contextlib.redirect_stdout(console_output), contextlib.redirect_stderr(console_output):
            result = dep_module._handle_install_packages()
        
        captured_output = console_output.getvalue()
        
        print(f"Install result: {result}")
        
        # Check for problematic logs
        problematic_logs = [
            "[Status] Tidak ada paket yang dipilih untuk diinstal",
            "Failed to get selected packages"
        ]
        
        found_issues = []
        for log_msg in problematic_logs:
            if log_msg in captured_output:
                found_issues.append(log_msg)
        
        if found_issues:
            print(f"❌ Found problematic logs in console: {found_issues}")
            print(f"Console output: {captured_output}")
        else:
            print("✅ No problematic logs in console")
            
        # Also test if operation container received the logs
        if hasattr(dep_module, '_ui_components') and dep_module._ui_components:
            operation_container = dep_module._ui_components.get('operation_container')
            if operation_container:
                print("✅ Operation container is available for logs")
            else:
                print("❌ Operation container not found")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dependency_install_button()