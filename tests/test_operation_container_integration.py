#!/usr/bin/env python3
"""
Comprehensive Test Suite for Operation Container Integration

This test suite validates the enhanced integration between operation container
and its components: progress tracker, log accordion, and dialog system.
"""

import sys
import os
import time
import traceback
from typing import Dict, Any, List
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_operation_container_basic_creation():
    """Test basic operation container creation and initialization."""
    print("🧪 Testing basic operation container creation...")
    
    try:
        from smartcash.ui.components.operation_container import create_operation_container
        
        # Test basic creation
        container = create_operation_container(
            component_name="test_container",
            show_progress=True,
            show_logs=True,
            show_dialog=True
        )
        
        assert container is not None, "Container should be created"
        assert 'container' in container, "Container should have 'container' key"
        assert 'progress_tracker' in container, "Container should have progress tracker"
        assert 'log_accordion' in container, "Container should have log accordion"
        assert 'update_progress' in container, "Container should have update_progress method"
        assert 'log_message' in container, "Container should have log_message method"
        
        print("✅ Basic creation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic creation test failed: {e}")
        traceback.print_exc()
        return False

def test_progress_tracker_integration():
    """Test progress tracker integration and functionality."""
    print("🧪 Testing progress tracker integration...")
    
    try:
        from smartcash.ui.components.operation_container import create_operation_container
        
        # Create container with progress tracking
        container = create_operation_container(
            component_name="progress_test",
            show_progress=True,
            show_logs=True,
            progress_levels='triple'
        )
        
        # Test progress updates
        update_progress = container['update_progress']
        
        # Test primary progress
        update_progress(25, "Testing primary progress", "primary")
        time.sleep(0.1)  # Small delay to allow UI updates
        
        # Test secondary progress
        update_progress(50, "Testing secondary progress", "secondary") 
        time.sleep(0.1)
        
        # Test tertiary progress
        update_progress(75, "Testing tertiary progress", "tertiary")
        time.sleep(0.1)
        
        # Test completion
        update_progress(100, "Testing completion", "primary")
        time.sleep(0.1)
        
        print("✅ Progress tracker integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Progress tracker integration test failed: {e}")
        traceback.print_exc()
        return False

def test_log_accordion_integration():
    """Test log accordion integration and functionality."""
    print("🧪 Testing log accordion integration...")
    
    try:
        from smartcash.ui.components.operation_container import create_operation_container
        from smartcash.ui.components.log_accordion import LogLevel
        
        # Create container with logging
        container = create_operation_container(
            component_name="log_test",
            show_logs=True,
            log_module_name="test_module",
            log_height="300px"
        )
        
        # Test logging methods
        log_message = container['log_message']
        
        # Test different log levels
        log_message("Debug message for testing", LogLevel.DEBUG)
        log_message("Info message for testing", LogLevel.INFO)
        log_message("Warning message for testing", LogLevel.WARNING)
        log_message("Error message for testing", LogLevel.ERROR)
        
        print("✅ Log accordion integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Log accordion integration test failed: {e}")
        traceback.print_exc()
        return False

def test_dialog_system_integration():
    """Test dialog system integration and functionality."""
    print("🧪 Testing dialog system integration...")
    
    try:
        from smartcash.ui.components.operation_container import create_operation_container
        
        # Create container with dialogs
        container = create_operation_container(
            component_name="dialog_test",
            show_dialog=True,
            show_logs=True
        )
        
        # Test dialog availability
        show_dialog = container.get('show_dialog')
        show_info_dialog = container.get('show_info_dialog')
        clear_dialog = container.get('clear_dialog')
        
        assert show_dialog is not None, "show_dialog method should be available"
        assert show_info_dialog is not None, "show_info_dialog method should be available" 
        assert clear_dialog is not None, "clear_dialog method should be available"
        
        print("✅ Dialog system integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Dialog system integration test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all operation container integration tests."""
    print("🚀 Starting comprehensive operation container integration tests...")
    print("=" * 80)
    
    tests = [
        ("Basic Creation", test_operation_container_basic_creation),
        ("Progress Tracker Integration", test_progress_tracker_integration),
        ("Log Accordion Integration", test_log_accordion_integration),
        ("Dialog System Integration", test_dialog_system_integration)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} tests...")
        print("-" * 60)
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name}: CRASHED - {e}")
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Operation container integration is working perfectly.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    print("🧪 Operation Container Integration Test Suite")
    print("=" * 80)
    print("This test suite validates the enhanced integration between")
    print("operation container and its components.")
    print("")
    
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        traceback.print_exc()
        sys.exit(1)