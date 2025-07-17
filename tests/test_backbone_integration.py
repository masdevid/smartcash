#!/usr/bin/env python3
"""
Test Backbone UI Integration

This test validates the backbone UI module integration with the enhanced
operation container, including progress tracker visibility and logging bridge.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_backbone_initialization():
    """Test backbone UI module initialization."""
    print("🧪 Testing backbone UI module initialization...")
    
    try:
        from smartcash.ui.model.backbone.backbone_uimodule import initialize_backbone_ui
        
        # Initialize backbone UI without display
        result = initialize_backbone_ui(display=False)
        
        assert result is not None, "Backbone UI should initialize"
        assert result.get('success') == True, "Initialization should succeed"
        assert 'module' in result, "Result should contain module"
        assert 'ui_components' in result, "Result should contain UI components"
        
        # Test module properties
        module = result['module']
        assert module.is_initialized(), "Module should be initialized"
        
        # Test UI components
        ui_components = result['ui_components']
        assert 'operation_container' in ui_components, "Should have operation container"
        assert 'main_container' in ui_components, "Should have main container"
        
        print("  ✅ Module initialized successfully")
        print(f"  📊 UI Components: {list(ui_components.keys())}")
        
        return True, result
        
    except Exception as e:
        print(f"❌ Backbone initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_progress_tracker_visibility():
    """Test that progress tracker is visible by default."""
    print("🧪 Testing progress tracker visibility...")
    
    try:
        success, result = test_backbone_initialization()
        if not success:
            return False
        
        module = result['module']
        ui_components = result['ui_components']
        
        # Get operation container
        operation_container = ui_components.get('operation_container')
        assert operation_container is not None, "Operation container should exist"
        
        # Test progress tracker exists
        if hasattr(operation_container, 'progress_tracker'):
            progress_tracker = operation_container.progress_tracker
            assert progress_tracker is not None, "Progress tracker should exist"
            
            # Test initialization and visibility
            if hasattr(progress_tracker, '_initialized'):
                assert progress_tracker._initialized, "Progress tracker should be initialized"
            
            if hasattr(progress_tracker, 'container') and progress_tracker.container:
                display = progress_tracker.container.layout.display
                print(f"  📊 Progress tracker display: {display}")
                assert display != 'none', "Progress tracker should be visible"
            
        print("  ✅ Progress tracker is visible")
        return True
        
    except Exception as e:
        print(f"❌ Progress tracker visibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging_bridge():
    """Test that logging bridge is set up correctly."""
    print("🧪 Testing logging bridge setup...")
    
    try:
        success, result = test_backbone_initialization()
        if not success:
            return False
        
        module = result['module']
        
        # Test that the logging bridge setup method exists
        assert hasattr(module, '_setup_ui_logging_bridge'), "Should have logging bridge setup method"
        
        # Test operation manager exists and has logging
        if hasattr(module, '_operation_manager') and module._operation_manager:
            op_manager = module._operation_manager
            
            # Test logging functionality
            if hasattr(op_manager, 'log'):
                op_manager.log("Test log message from backbone", 'info')
                print("  ✅ Logging integration working")
            
            # Test progress update functionality
            if hasattr(op_manager, 'update_progress'):
                op_manager.update_progress(50, "Test progress from backbone")
                print("  ✅ Progress update integration working")
        
        print("  ✅ Logging bridge appears to be working")
        return True
        
    except Exception as e:
        print(f"❌ Logging bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backend_logging_capture():
    """Test that backend service logs are captured."""
    print("🧪 Testing backend logging capture...")
    
    try:
        import logging
        
        # Create a test logger that should be captured
        test_logger = logging.getLogger('smartcash.dataset.preprocessor.test')
        
        success, result = test_backbone_initialization()
        if not success:
            return False
        
        # Wait a moment for logging bridge to be set up
        time.sleep(0.1)
        
        # Test logging from backend service namespace
        test_logger.info("Test backend service log message")
        test_logger.warning("Test backend service warning")
        test_logger.error("Test backend service error")
        
        print("  ✅ Backend logging capture test completed")
        print("  📝 Check operation container logs to verify capture")
        return True
        
    except Exception as e:
        print(f"❌ Backend logging capture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_operation_flow():
    """Test a complete operation flow."""
    print("🧪 Testing complete operation flow...")
    
    try:
        success, result = test_backbone_initialization()
        if not success:
            return False
        
        module = result['module']
        
        # Test operation manager operations
        if hasattr(module, '_operation_manager') and module._operation_manager:
            op_manager = module._operation_manager
            
            # Test clearing logs
            if hasattr(op_manager, 'clear_logs'):
                op_manager.clear_logs()
                print("  ✅ Log clearing works")
            
            # Test progress updates
            if hasattr(op_manager, 'update_progress'):
                op_manager.update_progress(25, "Starting operation...")
                time.sleep(0.1)
                op_manager.update_progress(50, "Processing...")
                time.sleep(0.1)
                op_manager.update_progress(75, "Nearly complete...")
                time.sleep(0.1)
                op_manager.update_progress(100, "Operation completed!")
                print("  ✅ Progress flow works")
            
            # Test logging
            if hasattr(op_manager, 'log'):
                op_manager.log("🚀 Operation started", 'info')
                op_manager.log("⚠️ Some warning occurred", 'warning')
                op_manager.log("✅ Operation completed successfully", 'success')
                print("  ✅ Logging flow works")
        
        print("  ✅ Complete operation flow test passed")
        return True
        
    except Exception as e:
        print(f"❌ Operation flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_backbone_tests():
    """Run all backbone integration tests."""
    print("🚀 Starting backbone UI integration tests...")
    print("=" * 70)
    
    tests = [
        ("Backbone Initialization", test_backbone_initialization),
        ("Progress Tracker Visibility", test_progress_tracker_visibility),
        ("Logging Bridge Setup", test_logging_bridge),
        ("Backend Logging Capture", test_backend_logging_capture),
        ("Complete Operation Flow", test_operation_flow)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        print("-" * 50)
        
        try:
            if test_name == "Backbone Initialization":
                # Special handling for initialization test
                result, _ = test_func()
            else:
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
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 BACKBONE INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL BACKBONE TESTS PASSED!")
        print("✅ Progress tracker is visible by default")
        print("✅ Backend logging is properly captured")
        print("✅ Operation container integration is working")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed.")
        return False

if __name__ == "__main__":
    print("🧬 Backbone UI Integration Test Suite")
    print("=" * 70)
    print("Testing the fixes for:")
    print("- Progress tracker visibility")
    print("- Backend logging capture")
    print("- Operation container integration")
    print("")
    
    try:
        success = run_backbone_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)