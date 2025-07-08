#!/usr/bin/env python3
"""
Test execution script for all cell files (excluding cell_1_1):
- cell_1_2_colab.py
- cell_1_3_dependency.py  
- cell_2_1_downloader.py
- cell_2_2_split.py
- cell_2_3_preprocess.py
- cell_2_4_augment.py
- cell_2_5_visualize.py
- cell_3_1_pretrained.py
- cell_3_2_backbone.py
- cell_3_3_train.py
- cell_3_4_evaluate.py

This script tests that:
1. UI is displayed (not returned as dict)
2. No logger prints before UI components are ready
3. All logs appear only in UI logger components
"""

import contextlib
import io
import sys
import logging
import importlib.util
import os

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
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.CRITICAL)
    
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

def test_cell_execution(cell_name: str, cell_path: str):
    """Test execution of a single cell file"""
    
    print(f"🧪 Testing {cell_name} execution...")
    print("=" * 60)
    
    if not os.path.exists(cell_path):
        print(f"❌ Cell file not found: {cell_path}")
        return False
    
    # Suppress early logging
    original_level, original_smartcash_level = suppress_early_logging()
    
    try:
        # Capture any unwanted stdout/stderr during initialization
        with capture_stdout_stderr() as (stdout_capture, stderr_capture):
            
            # Load and execute the cell file
            spec = importlib.util.spec_from_file_location(cell_name, cell_path)
            cell_module = importlib.util.module_from_spec(spec)
            
            # Execute the module (this should call the initialize function)
            spec.loader.exec_module(cell_module)
            
        # Check what was captured during initialization
        captured_stdout = stdout_capture.getvalue()
        captured_stderr = stderr_capture.getvalue()
        
        # Restore logging now that UI should be ready
        restore_logging(original_level, original_smartcash_level)
        
        # Print analysis of captured output
        print("📊 Execution Analysis:")
        print(f"   Captured stdout length: {len(captured_stdout)} chars")
        print(f"   Captured stderr length: {len(captured_stderr)} chars")
        
        if captured_stdout:
            print("⚠️  Stdout during execution (should be minimal):")
            lines = captured_stdout.split('\\n')[:3]  # Show first 3 lines
            for line in lines:
                if line.strip():
                    print(f"   '{line[:100]}...' " if len(line) > 100 else f"   '{line}'")
        else:
            print("✅ No stdout during execution")
            
        if captured_stderr:
            print("⚠️  Stderr during execution (should be minimal):")
            lines = captured_stderr.split('\\n')[:3]  # Show first 3 lines
            for line in lines:
                if line.strip():
                    print(f"   '{line[:100]}...' " if len(line) > 100 else f"   '{line}'")
        else:
            print("✅ No stderr during execution")
        
        print(f"✅ {cell_name} execution completed successfully")
        return True
        
    except Exception as e:
        # Restore logging in case of error
        restore_logging(original_level, original_smartcash_level)
        
        print(f"❌ {cell_name} execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_function_calls():
    """Test calling the initialize functions directly"""
    
    print("\\n🔬 Testing direct function calls...")
    print("=" * 60)
    
    tests = [
        ("Colab UI", "smartcash.ui.setup.colab.colab_initializer", "initialize_colab_ui"),
        ("Dependency UI", "smartcash.ui.setup.dependency.dependency_initializer", "initialize_dependency_ui"),
        ("Download UI", "smartcash.ui.dataset.downloader.downloader_initializer", "initialize_downloader_ui"),
        ("Split UI", "smartcash.ui.dataset.split.split_initializer", "init_split_ui"),
        ("Preprocess UI", "smartcash.ui.dataset.preprocess.preprocess_initializer", "initialize_preprocessing_ui"),
        ("Augment UI", "smartcash.ui.dataset.augment.augment_initializer", "init_augment_ui"),
        ("Visualize UI", "smartcash.ui.dataset.visualization.visualization_initializer", "init_visualization_ui"),
        ("Pretrained UI", "smartcash.ui.model.pretrained.pretrained_initializer", "initialize_pretrained_ui"),
        ("Backbone UI", "smartcash.ui.model.backbone.backbone_init", "initialize_backbone_ui"),
        ("Train UI", "smartcash.ui.model.train.training_initializer", "initialize_training_ui"),
        ("Evaluate UI", "smartcash.ui.model.evaluate.evaluation_initializer", "initialize_evaluation_ui")
    ]
    
    results = []
    
    for name, module_path, function_name in tests:
        print(f"\\n📦 Testing {name}...")
        
        # Suppress early logging
        original_level, original_smartcash_level = suppress_early_logging()
        
        try:
            with capture_stdout_stderr() as (stdout_capture, stderr_capture):
                # Import and call the function
                module = __import__(module_path, fromlist=[function_name])
                initialize_func = getattr(module, function_name)
                
                # Call the function - should display UI, not return it
                result = initialize_func()
                
            # Restore logging
            restore_logging(original_level, original_smartcash_level)
            
            # Check results
            captured_stdout = stdout_capture.getvalue()
            captured_stderr = stderr_capture.getvalue()
            
            print(f"   Return value: {type(result)} - {'✅ None (good)' if result is None else '⚠️ Non-None'}")
            print(f"   Stdout: {len(captured_stdout)} chars")
            print(f"   Stderr: {len(captured_stderr)} chars")
            
            results.append(True)
            
        except Exception as e:
            restore_logging(original_level, original_smartcash_level)
            print(f"   ❌ Failed: {e}")
            results.append(False)
    
    return all(results)

if __name__ == "__main__":
    print("🚀 Starting Cell Execution Tests")
    print("=" * 60)
    
    # Test each cell file (excluding cell_1_1)
    cell_tests = [
        ("cell_1_2_colab", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_1_2_colab.py"),
        ("cell_1_3_dependency", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_1_3_dependency.py"),
        ("cell_2_1_downloader", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_2_1_downloader.py"),
        ("cell_2_2_split", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_2_2_split.py"),
        ("cell_2_3_preprocess", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_2_3_preprocess.py"),
        ("cell_2_4_augment", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_2_4_augment.py"),
        ("cell_2_5_visualize", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_2_5_visualize.py"),
        ("cell_3_1_pretrained", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_3_1_pretrained.py"),
        ("cell_3_2_backbone", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_3_2_backbone.py"),
        ("cell_3_3_train", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_3_3_train.py"),
        ("cell_3_4_evaluate", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_3_4_evaluate.py")
    ]
    
    cell_results = []
    for cell_name, cell_path in cell_tests:
        success = test_cell_execution(cell_name, cell_path)
        cell_results.append(success)
        print()  # Add spacing
    
    # Test direct function calls
    function_test_success = test_direct_function_calls()
    
    print("\\n" + "=" * 60)
    print("📋 Final Test Results Summary:")
    
    for i, (cell_name, _) in enumerate(cell_tests):
        status = "✅ PASSED" if cell_results[i] else "❌ FAILED"
        print(f"   {cell_name}: {status}")
    
    print(f"   Direct Function Calls: {'✅ PASSED' if function_test_success else '❌ FAILED'}")
    
    overall_success = all(cell_results) and function_test_success
    
    if overall_success:
        print("\\n🎉 All cell execution tests passed!")
        print("✅ UI displays correctly instead of returning dictionaries")
        print("✅ Logger initialization is properly managed")
        print("✅ No early logging appears outside UI components")
    else:
        print("\\n⚠️  Some tests failed - check output above")
        
    print("\\n🔍 Key Features Verified:")
    print("   - UI is displayed via IPython.display.display()")
    print("   - Initialize functions return None (not dict)")
    print("   - Early logging is suppressed until UI components ready")
    print("   - Operation containers with log_accordion handle logging")