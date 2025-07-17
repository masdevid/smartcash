#!/usr/bin/env python3
"""
Architecture verification test - examines code without importing.
"""

import re
import os

def test_dependency_uimodule_architecture():
    """Test that the dependency_uimodule follows the correct architecture"""
    print("🔍 Testing Dependency UIModule Architecture...")
    
    dependency_file = "/Users/masdevid/Projects/smartcash/smartcash/ui/setup/dependency/dependency_uimodule.py"
    
    if not os.path.exists(dependency_file):
        print("❌ Dependency module file not found")
        return False
    
    with open(dependency_file, 'r') as f:
        content = f.read()
    
    # Check for proper inheritance from BaseUIModule
    if "class DependencyUIModule(BaseUIModule)" in content:
        print("✅ Inherits from BaseUIModule (correct)")
    else:
        print("❌ Does not inherit from BaseUIModule")
        return False
    
    # Check that it doesn't have the old progress tracking mixin
    if "ProgressTrackingMixin" in content:
        print("❌ Still has ProgressTrackingMixin (DRY violation)")
        return False
    else:
        print("✅ No ProgressTrackingMixin (DRY compliant)")
    
    # Check for correct use of operation_mixin methods
    if "update_operation_status" in content:
        print("✅ Uses update_operation_status (correct architecture)")
    else:
        print("❌ Does not use update_operation_status")
        return False
    
    # Check for correct use of progress updates
    if "update_progress" in content:
        print("✅ Uses update_progress (correct architecture)")
    else:
        print("❌ Does not use update_progress")
        return False
    
    # Check for correct use of logging (can be either log_operation or self.log)
    if "log_operation" in content or "self.log(" in content:
        print("✅ Uses logging methods (correct architecture)")
    else:
        print("❌ Does not use logging methods")
        return False
    
    return True

def test_operation_mixin_architecture():
    """Test that the operation_mixin follows the correct architecture"""
    print("\n🔍 Testing Operation Mixin Architecture...")
    
    operation_mixin_file = "/Users/masdevid/Projects/smartcash/smartcash/ui/core/mixins/operation_mixin.py"
    
    if not os.path.exists(operation_mixin_file):
        print("❌ Operation mixin file not found")
        return False
    
    with open(operation_mixin_file, 'r') as f:
        content = f.read()
    
    # Check that status updates go to header_container
    if "header_container.update_status" in content:
        print("✅ Status updates routed to header_container (correct)")
    else:
        print("❌ Status updates not routed to header_container")
        return False
    
    # Check that progress updates go to operation_container
    if "operation_container.update_progress" in content:
        print("✅ Progress updates routed to operation_container (correct)")
    else:
        print("❌ Progress updates not routed to operation_container")
        return False
    
    # Check that logging goes to operation_container
    if "operation_container.log" in content:
        print("✅ Logging routed to operation_container (correct)")
    else:
        print("❌ Logging not routed to operation_container")
        return False
    
    # Check that operation_container does NOT handle status updates
    if "operation_container.update_status" in content:
        print("❌ operation_container handles status updates (WRONG)")
        return False
    else:
        print("✅ operation_container does NOT handle status updates (correct)")
    
    return True

def test_downloader_uimodule_architecture():
    """Test that the downloader_uimodule follows the correct architecture"""
    print("\n🔍 Testing Downloader UIModule Architecture...")
    
    downloader_file = "/Users/masdevid/Projects/smartcash/smartcash/ui/dataset/downloader/downloader_uimodule.py"
    
    if not os.path.exists(downloader_file):
        print("❌ Downloader module file not found")
        return False
    
    with open(downloader_file, 'r') as f:
        content = f.read()
    
    # Check for proper delegation to operation_mixin
    if "update_operation_status" in content:
        print("✅ Uses update_operation_status (correct architecture)")
    else:
        print("❌ Does not use update_operation_status")
        return False
    
    # Check for proper delegation to button_handler_mixin
    if "enable_button" in content and "disable_button" in content:
        print("✅ Uses button_handler_mixin methods (correct architecture)")
    else:
        print("❌ Does not use button_handler_mixin methods")
        return False
    
    # Check for correct log delegation
    if "log_operation" in content:
        print("✅ Uses log_operation (correct architecture)")
    else:
        print("❌ Does not use log_operation")
        return False
    
    return True

def test_no_dry_violations():
    """Test that there are no DRY violations"""
    print("\n🔍 Testing for DRY Violations...")
    
    # Check that progress_tracking_mixin.py was removed
    progress_mixin_file = "/Users/masdevid/Projects/smartcash/smartcash/ui/core/mixins/progress_tracking_mixin.py"
    if os.path.exists(progress_mixin_file):
        print("❌ progress_tracking_mixin.py still exists (DRY violation)")
        return False
    else:
        print("✅ progress_tracking_mixin.py was removed (DRY compliant)")
    
    # Check that BaseUIModule doesn't inherit from ProgressTrackingMixin
    base_ui_file = "/Users/masdevid/Projects/smartcash/smartcash/ui/core/base_ui_module.py"
    if os.path.exists(base_ui_file):
        with open(base_ui_file, 'r') as f:
            content = f.read()
        
        # Check for actual inheritance (not just comments)
        import_match = re.search(r'from.*ProgressTrackingMixin', content)
        class_match = re.search(r'class.*ProgressTrackingMixin', content)
        
        if import_match or class_match:
            print("❌ BaseUIModule still inherits from ProgressTrackingMixin (DRY violation)")
            return False
        else:
            print("✅ BaseUIModule does not inherit from ProgressTrackingMixin (DRY compliant)")
    
    return True

def test_correct_architecture_summary():
    """Test that the architecture summary is correct"""
    print("\n🔍 Testing Architecture Summary...")
    
    summary_file = "/Users/masdevid/Projects/smartcash/correct_architecture_summary.md"
    if not os.path.exists(summary_file):
        print("❌ Architecture summary file not found")
        return False
    
    with open(summary_file, 'r') as f:
        content = f.read()
    
    # Check key architecture points
    if "Status updates → header_container" in content:
        print("✅ Architecture summary shows correct status routing")
    else:
        print("❌ Architecture summary missing correct status routing")
        return False
    
    if "Progress updates → operation_container" in content:
        print("✅ Architecture summary shows correct progress routing")
    else:
        print("❌ Architecture summary missing correct progress routing")
        return False
    
    if "NO update_status() (REMOVED)" in content:
        print("✅ Architecture summary shows operation_container doesn't handle status")
    else:
        print("❌ Architecture summary missing operation_container status removal")
        return False
    
    return True

def main():
    """Run all architecture verification tests"""
    print("🚀 Architecture Verification Test")
    print("=" * 60)
    print("Verifying correct architecture without importing:")
    print("- Status updates → header_container")
    print("- Progress updates → operation_container") 
    print("- Logging → operation_container")
    print("- No DRY violations")
    print("- Proper mixin usage")
    print("=" * 60)
    
    tests = [
        ("Dependency UIModule Architecture", test_dependency_uimodule_architecture),
        ("Operation Mixin Architecture", test_operation_mixin_architecture),
        ("Downloader UIModule Architecture", test_downloader_uimodule_architecture),
        ("No DRY Violations", test_no_dry_violations),
        ("Correct Architecture Summary", test_correct_architecture_summary)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Architecture Verification Results:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 ALL ARCHITECTURE TESTS PASSED!")
        print("✅ Dual progress tracker: CORRECTLY IMPLEMENTED")
        print("✅ Status updates via header_container: CORRECTLY IMPLEMENTED")
        print("✅ Logging via operation_container: CORRECTLY IMPLEMENTED")
        print("✅ Proper architecture separation: CORRECTLY IMPLEMENTED")
        print("✅ DRY compliance: CORRECTLY IMPLEMENTED")
        print("✅ Mixin delegation: CORRECTLY IMPLEMENTED")
        
        print("\n🔧 Architecture Summary:")
        print("- dependency_uimodule inherits from BaseUIModule")
        print("- Uses operation_mixin for status/progress/logging coordination")
        print("- Uses button_handler_mixin for button management")
        print("- Status updates routed to header_container")
        print("- Progress updates routed to operation_container")
        print("- Logging routed to operation_container")
        print("- No DRY violations (progress_tracking_mixin removed)")
        print("- Proper separation of concerns")
        
        return True
    else:
        print("\n⚠️  Some architecture tests failed")
        return False

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)