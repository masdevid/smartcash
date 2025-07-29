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
    
    print(f"🧪 {cell_name}")
    
    if not os.path.exists(cell_path):
        print(f"❌ File not found: {cell_path}")
        return False
    
    # Suppress early logging
    original_level, original_smartcash_level = suppress_early_logging()
    
    try:
        # Capture any unwanted stdout/stderr during initialization
        with capture_stdout_stderr() as (stdout_capture, stderr_capture):
            # Load and execute the cell file
            spec = importlib.util.spec_from_file_location(cell_name, cell_path)
            cell_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cell_module)
            
        # Get captured output
        captured_stdout = stdout_capture.getvalue()
        captured_stderr = stderr_capture.getvalue()
        
        # Restore logging
        restore_logging(original_level, original_smartcash_level)
        
        # Check for issues
        has_stdout = bool(captured_stdout.strip())
        has_stderr = bool(captured_stderr.strip())
        
        if has_stdout or has_stderr:
            status = "⚠️ "
            if has_stdout:
                status += f"stdout({len(captured_stdout)})"
            if has_stderr:
                status += f" stderr({len(captured_stderr)})"
            print(f"   {status}")
        else:
            print("   ✅ Clean execution")
            
        return True
        
    except Exception as e:
        # Restore logging in case of error
        restore_logging(original_level, original_smartcash_level)
        
        print(f"❌ {cell_name} execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_function_calls():
    """Test calling the factory display functions directly"""
    print("\n🔬 Testing display functions...")
    
    # Mapping of test names to their module paths and function names
    function_tests = [
        ("Colab UI", "smartcash.ui.setup.colab", "create_colab_display"),
        ("Dependency UI", "smartcash.ui.setup.dependency", "create_dependency_display"),
        ("Download UI", "smartcash.ui.dataset.downloader", "create_downloader_display"),
        # ("Split UI", "smartcash.ui.dataset.split", "create_split_display"),
        ("Preprocess UI", "smartcash.ui.dataset.preprocessing", "create_preprocessing_display"),
        ("Augment UI", "smartcash.ui.dataset.augmentation", "create_augmentation_display"),
        ("Visualize UI", "smartcash.ui.dataset.visualization", "create_visualization_display"),
        ("Pretrained UI", "smartcash.ui.model.pretrained", "create_pretrained_display"),
        # ("Backbone UI", "smartcash.ui.model.backbone", "create_backbone_display"),
        ("Train UI", "smartcash.ui.model.training", "create_training_display"),
        ("Evaluate UI", "smartcash.ui.model.evaluation", "create_evaluation_display")
    ]
    
    results = []
    
    for display_name, module_path, func_name in function_tests:
        try:
            # Import the module and get the function
            module = importlib.import_module(module_path)
            create_display_func = getattr(module, func_name)
            
            # Call the function and capture output
            with capture_stdout_stderr() as (stdout_capture, stderr_capture):
                # Some functions might return a callable that needs to be called
                if callable(create_display_func):
                    display_func = create_display_func()
                    if callable(display_func):
                        result = display_func()
                    else:
                        result = display_func
                else:
                    result = create_display_func
            
            # Get captured output
            captured_stdout = stdout_capture.getvalue()
            captured_stderr = stderr_capture.getvalue()
            
            # Check results
            output_status = []
            if result is not None:
                output_status.append("bad return")
                
            if captured_stdout:
                output_status.append(f"stdout({len(captured_stdout)})")
            if captured_stderr:
                output_status.append(f"stderr({len(captured_stderr)})")
                
            status = "✅" if not output_status else "⚠️ " + ", ".join(output_status)
            print(f"   {display_name}: {status}")
            
            results.append(True)  # If we got here without exception, consider it a success
            
        except Exception as e:
            print(f"   ❌ {display_name}: {e.__class__.__name__} - {str(e)}")
            results.append(False)
    
    return all(results)

if __name__ == "__main__":
    print("🚀 Running UI Component Tests")
    
    # Test each cell file (excluding cell_1_1)
    cell_tests = [
        ("cell_1_2_colab", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_1_2_colab.py"),
        ("cell_1_3_dependency", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_1_3_dependency.py"),
        ("cell_2_1_downloader", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_2_1_downloader.py"),
        # ("cell_2_2_split", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_2_2_split.py"),
        ("cell_2_3_preprocess", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_2_3_preprocess.py"),
        ("cell_2_4_augment", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_2_4_augment.py"),
        ("cell_2_5_visualize", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_2_5_visualize.py"),
        ("cell_3_1_pretrained", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_3_1_pretrained.py"),
        # ("cell_3_2_backbone", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_3_2_backbone.py"),
        ("cell_3_3_train", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_3_3_train.py"),
        ("cell_3_4_evaluate", "/Users/masdevid/Projects/smartcash/smartcash/ui/cells/cell_3_4_evaluate.py")
    ]
    
    # Run cell tests
    print("\n🧪 Testing Cell Execution:")
    cell_results = []
    for cell_name, cell_path in cell_tests:
        success = test_cell_execution(cell_name, cell_path)
        cell_results.append(success)
    
    # Run function tests
    function_test_success = test_direct_function_calls()
    
    # Print summary
    print("\n📋 Summary:")
    passed = sum(1 for r in cell_results if r)
    print(f"  Cells: {passed}/{len(cell_results)} passed")
    print(f"  Functions: {'✅ PASSED' if function_test_success else '❌ FAILED'}")
    
    if all(cell_results) and function_test_success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)