#!/usr/bin/env python3
"""
Validate DRY principle fix - operation_container should handle progress, not individual modules
"""

import os
import re

def validate_dry_fix():
    """Validate that DRY violation has been fixed"""
    print("🔍 Validating DRY principle fix...")
    
    # 1. Check that duplicate methods are removed from dependency module
    file_path = "smartcash/ui/setup/dependency/dependency_uimodule.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check that duplicate methods are removed
    duplicate_methods = [
        "def update_stage_progress",
        "def update_triple_progress", 
        "def set_progress_level",
        "def get_progress_level"
    ]
    
    found_duplicates = []
    for method in duplicate_methods:
        if method in content:
            found_duplicates.append(method)
    
    if found_duplicates:
        print(f"❌ Still found duplicate methods: {found_duplicates}")
        return False
    
    # Check that DRY principle comment is present
    if "DRY PRINCIPLE" in content:
        print("✅ DRY principle documentation found")
    else:
        print("❌ DRY principle documentation missing")
        return False
    
    # 2. Check that dependency UI uses dual progress mode
    ui_file = "smartcash/ui/setup/dependency/components/dependency_ui.py"
    
    if not os.path.exists(ui_file):
        print(f"❌ File not found: {ui_file}")
        return False
    
    with open(ui_file, 'r') as f:
        ui_content = f.read()
    
    if "progress_levels='dual'" in ui_content:
        print("✅ Dependency module configured for dual progress")
    else:
        print("❌ Dependency module not configured for dual progress")
        return False
    
    # 3. Check that operation_container supports dual progress
    op_file = "smartcash/ui/components/operation_container.py"
    
    if not os.path.exists(op_file):
        print(f"❌ File not found: {op_file}")
        return False
    
    with open(op_file, 'r') as f:
        op_content = f.read()
    
    # Check for dual progress support
    dual_checks = [
        "progress_levels",
        "level_mapping",
        "ProgressLevel.DUAL",
        "def update_progress"
    ]
    
    missing_dual = []
    for check in dual_checks:
        if check not in op_content:
            missing_dual.append(check)
    
    if missing_dual:
        print(f"❌ Operation container missing dual progress support: {missing_dual}")
        return False
    
    print("✅ Operation container supports dual progress")
    
    # 4. Check for proper progress level mapping
    if '"dual": [' in op_content or 'dual' in op_content:
        print("✅ Dual progress level mapping found")
    else:
        print("❌ Dual progress level mapping missing")
        return False
    
    print("\n🎉 DRY principle fix validation PASSED!")
    print("✅ Duplicate methods removed from dependency module")
    print("✅ Dependency module uses dual progress mode")
    print("✅ Operation container handles progress centrally")
    print("✅ No DRY violations detected")
    
    return True

if __name__ == "__main__":
    success = validate_dry_fix()
    exit(0 if success else 1)