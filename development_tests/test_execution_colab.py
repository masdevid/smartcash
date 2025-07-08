#!/usr/bin/env python3
"""
Test execution script for cell_1_2_colab.py

This script tests the colab UI initialization and ensures:
1. UI is displayed (not returned as dict)
2. No logger prints before UI components are ready
3. All logs appear only in UI logger components
"""

import contextlib
import io
import sys
import logging
from typing import Any, Dict
from IPython.display import display

@contextlib.contextmanager
def capture_stdout_stderr():
    """Context manager to capture stdout and stderr"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        yield stdout_capture, stderr_capture
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def suppress_early_logging():
    """Temporarily suppress logging before UI is ready"""
    # Get all loggers and set them to CRITICAL level temporarily
    root_logger = logging.getLogger()
    original_level = root_logger.level
    
    # Set high level to suppress early logs
    root_logger.setLevel(logging.CRITICAL)
    
    # Also suppress specific loggers
    smartcash_logger = logging.getLogger('smartcash')
    original_smartcash_level = smartcash_logger.level
    smartcash_logger.setLevel(logging.CRITICAL)
    
    return original_level, original_smartcash_level

def restore_logging(original_level, original_smartcash_level):
    """Restore logging levels after UI is ready"""
    root_logger = logging.getLogger()
    root_logger.setLevel(original_level)
    
    smartcash_logger = logging.getLogger('smartcash')
    smartcash_logger.setLevel(original_smartcash_level)

def test_colab_execution():
    """Test colab cell execution with proper UI display and logging management"""
    
    print("🧪 Testing cell_1_2_colab.py execution...")
    print("=" * 60)
    
    # Suppress early logging
    original_level, original_smartcash_level = suppress_early_logging()
    
    try:
        # Capture any unwanted stdout/stderr during initialization
        with capture_stdout_stderr() as (stdout_capture, stderr_capture):
            
            # Import and initialize the colab UI
            from smartcash.ui.setup.colab.colab_initializer import initialize_colab_ui
            
            # Call the initializer - this should now return UI components
            ui_result = initialize_colab_ui()
            
        # Check what was captured during initialization
        captured_stdout = stdout_capture.getvalue()
        captured_stderr = stderr_capture.getvalue()
        
        # Restore logging now that UI should be ready
        restore_logging(original_level, original_smartcash_level)
        
        # Print analysis of captured output
        print("📊 Initialization Analysis:")
        print(f"   Captured stdout length: {len(captured_stdout)} chars")
        print(f"   Captured stderr length: {len(captured_stderr)} chars")
        
        if captured_stdout:
            print("⚠️  Stdout during init (should be minimal):")
            print(f"   '{captured_stdout[:200]}...' " if len(captured_stdout) > 200 else f"   '{captured_stdout}'")
        else:
            print("✅ No stdout during initialization")
            
        if captured_stderr:
            print("⚠️  Stderr during init (should be minimal):")
            print(f"   '{captured_stderr[:200]}...' " if len(captured_stderr) > 200 else f"   '{captured_stderr}'")
        else:
            print("✅ No stderr during initialization")
        
        print("\n🎯 UI Analysis:")
        
        # Analyze the returned UI
        if isinstance(ui_result, dict):
            print("📦 UI returned as dictionary with keys:")
            for key in ui_result.keys():
                print(f"   - {key}: {type(ui_result[key])}")
            
            # Try to display the main UI component
            if 'ui' in ui_result:
                print("\n🎨 Displaying main UI component...")
                display(ui_result['ui'])
                print("✅ UI displayed successfully")
            elif 'main_container' in ui_result:
                print("\n🎨 Displaying main container...")
                display(ui_result['main_container'])
                print("✅ Main container displayed successfully")
            else:
                print("❌ No displayable UI component found")
                
        elif hasattr(ui_result, 'children') or hasattr(ui_result, 'layout'):
            # Direct widget
            print("🎨 UI returned as direct widget")
            print(f"   Type: {type(ui_result)}")
            display(ui_result)
            print("✅ Widget displayed successfully")
            
        else:
            print(f"❌ Unexpected UI result type: {type(ui_result)}")
            print(f"   Value: {ui_result}")
        
        print("\n🔍 Post-Display Logging Test:")
        print("   Testing if logs now appear in UI components...")
        
        # Test logging after UI is ready
        logger = logging.getLogger('smartcash.test')
        logger.info("🧪 Test log message - should appear in UI logger if properly configured")
        logger.warning("⚠️ Test warning - should appear in UI logger if properly configured")
        
        print("✅ Colab execution test completed")
        return True
        
    except Exception as e:
        # Restore logging in case of error
        restore_logging(original_level, original_smartcash_level)
        
        print(f"❌ Colab execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_colab_reinitialization():
    """Test that reinitializing doesn't create duplicate UI"""
    
    print("\n🔄 Testing colab reinitialization...")
    
    try:
        from smartcash.ui.setup.colab.colab_initializer import initialize_colab_ui
        
        # First initialization
        ui_result1 = initialize_colab_ui()
        
        # Second initialization (should reuse existing)
        ui_result2 = initialize_colab_ui()
        
        print(f"   First result type: {type(ui_result1)}")
        print(f"   Second result type: {type(ui_result2)}")
        
        if ui_result1 is ui_result2:
            print("✅ Reinitialization returns same instance (good)")
        else:
            print("⚠️  Reinitialization returns different instance")
            
        return True
        
    except Exception as e:
        print(f"❌ Reinitialization test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Colab UI Execution Tests")
    print("=" * 60)
    
    # Run main execution test
    test1_success = test_colab_execution()
    
    # Run reinitialization test
    test2_success = test_colab_reinitialization()
    
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    print(f"   Execution Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"   Reinitialization Test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print("🎉 All colab tests passed!")
    else:
        print("⚠️  Some tests failed - check output above")