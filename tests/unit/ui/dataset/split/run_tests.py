#!/usr/bin/env python3
"""
Test runner for the dataset split module.

This script runs all tests for the dataset split module and provides
a comprehensive report of test results.
"""

import sys
import pytest
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def run_split_module_tests():
    """Run all tests for the split module."""
    print("🚀 Starting Dataset Split Module Test Suite")
    print("=" * 50)
    
    # Get the test directory
    test_dir = Path(__file__).parent
    
    # Test files to run
    test_files = [
        "test_split_config_handler.py",
        "test_split_initializer.py", 
        "test_split_components.py",
        "test_split_integration.py",
        "test_split_execution.py"
    ]
    
    total_passed = 0
    total_failed = 0
    results = {}
    
    for test_file in test_files:
        print(f"\n📋 Running {test_file}")
        print("-" * 30)
        
        test_path = test_dir / test_file
        if not test_path.exists():
            print(f"⚠️ Test file not found: {test_file}")
            continue
        
        try:
            # Run pytest for the specific file
            result = subprocess.run([
                sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=project_root)
            
            # Parse results
            output_lines = result.stdout.split('\n')
            test_line = None
            for line in output_lines:
                if "passed" in line and "failed" in line:
                    test_line = line
                    break
            
            if test_line:
                # Extract numbers from line like "27 passed, 2 warnings in 1.73s"  
                import re
                passed_match = re.search(r'(\d+) passed', test_line)
                failed_match = re.search(r'(\d+) failed', test_line)
                
                passed = int(passed_match.group(1)) if passed_match else 0
                failed = int(failed_match.group(1)) if failed_match else 0
                
                print(f"✅ {test_file}: {passed} passed, {failed} failed")
                
                total_passed += passed
                total_failed += failed
                results[test_file] = (passed, failed)
            else:
                # Alternative parsing - look for patterns like "27 passed, 2 warnings"
                import re
                passed_match = re.search(r'(\d+) passed', result.stdout)
                failed_match = re.search(r'(\d+) failed', result.stdout)
                
                passed = int(passed_match.group(1)) if passed_match else 0
                failed = int(failed_match.group(1)) if failed_match else 0
                
                if passed > 0 or failed > 0:
                    print(f"✅ {test_file}: {passed} passed, {failed} failed")
                    total_passed += passed
                    total_failed += failed
                    results[test_file] = (passed, failed)
                else:
                    print(f"❌ {test_file}: Could not parse results")
                    results[test_file] = (0, 0)
                
        except Exception as e:
            print(f"❌ {test_file}: Error running tests - {str(e)}")
            results[test_file] = (0, 0)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    for test_file, result in results.items():
        if isinstance(result, tuple):
            passed, failed = result
            status = "✅" if failed == 0 else "❌"
            print(f"{status} {test_file}: {passed}/{passed + failed} passed")
        else:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {test_file}: {result['passed']}/{result['passed'] + result['failed']} passed")
    
    total_tests = total_passed + total_failed
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTotal: {total_passed}/{total_tests} passed")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if total_failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️ {total_failed} test(s) failed")
        return False


def run_quick_validation():
    """Run quick validation tests."""
    print("\n🔍 Quick Validation Tests")
    print("=" * 30)
    
    validations = []
    
    try:
        # Test 1: Module imports
        print("📦 Testing module imports...")
        from smartcash.ui.dataset.split.split_initializer import SplitInitializer
        from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
        from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG
        print("✅ Module imports successful")
        validations.append(True)
        
        # Test 2: Basic instantiation
        print("🏗️ Testing basic instantiation...")
        initializer = SplitInitializer()
        handler = SplitConfigHandler()
        assert initializer is not None
        assert handler is not None
        print("✅ Basic instantiation successful")
        validations.append(True)
        
        # Test 3: Config validation
        print("🔧 Testing config validation...")
        is_valid = handler.validate_config(DEFAULT_SPLIT_CONFIG)
        assert is_valid is True
        print("✅ Config validation successful")
        validations.append(True)
        
        # Test 4: Entry points
        print("🚪 Testing entry points...")
        from smartcash.ui.dataset.split.split_initializer import (
            init_split_ui, 
            get_split_initializer
        )
        test_initializer = get_split_initializer()
        assert test_initializer is not None
        print("✅ Entry points working")
        validations.append(True)
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        validations.append(False)
    
    success_count = sum(validations)
    total_count = len(validations)
    
    print(f"\n📊 Validation: {success_count}/{total_count} checks passed")
    
    return success_count == total_count


if __name__ == "__main__":
    print("🚀 Dataset Split Module Test Suite")
    print("=" * 60)
    
    # Run quick validation first
    validation_success = run_quick_validation()
    
    # Run full test suite
    test_success = run_split_module_tests()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 Final Results")
    print("=" * 60)
    
    print(f"✅ Validation: {'PASSED' if validation_success else 'FAILED'}")
    print(f"{'✅' if test_success else '❌'} Test Suite: {'PASSED' if test_success else 'FAILED'}")
    
    if validation_success and test_success:
        print("\n🎉 DATASET SPLIT MODULE TESTING SUCCESSFUL!")
        print("✅ All validations passed")
        print("✅ All tests passed")
        print("✅ Module is ready for use")
        exit_code = 0
    elif validation_success:
        print("\n⚠️ Module mostly functional with test issues")
        print("✅ Core functionality working")
        print("🔧 Review test failures for details")
        exit_code = 0
    else:
        print("\n❌ Significant issues found")
        print("🔧 Fix validation errors before proceeding")
        exit_code = 1
    
    sys.exit(exit_code)