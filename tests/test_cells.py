#!/usr/bin/env python3
"""
Test script to check the functionality of individual cells in smartcash/ui/cells/
"""

import sys
import os
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Add the project root to Python path
sys.path.insert(0, '/Users/masdevid/Projects/smartcash')

def test_cell_import_and_execution(cell_name, cell_path):
    """
    Test a single cell by importing and executing it.
    
    Args:
        cell_name (str): Name of the cell for reporting
        cell_path (str): Python import path for the cell
        
    Returns:
        dict: Test results with status, output, and error information
    """
    result = {
        'name': cell_name,
        'status': 'UNKNOWN',
        'output': '',
        'error': '',
        'import_success': False,
        'execution_success': False
    }
    
    try:
        # Capture stdout and stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Try to import the cell module
            try:
                # Import the cell module dynamically
                module_parts = cell_path.split('.')
                module = __import__(cell_path, fromlist=[module_parts[-1]])
                result['import_success'] = True
                result['status'] = 'SUCCESS'
                result['execution_success'] = True
                
            except ImportError as e:
                result['import_success'] = False
                result['status'] = 'IMPORT_ERROR'
                result['error'] = f"Import Error: {str(e)}"
                
            except Exception as e:
                result['import_success'] = True
                result['execution_success'] = False
                result['status'] = 'EXECUTION_ERROR'
                result['error'] = f"Execution Error: {str(e)}"
        
        # Capture the output
        result['output'] = stdout_capture.getvalue()
        if stderr_capture.getvalue():
            result['error'] += f"\nStderr: {stderr_capture.getvalue()}"
            
    except Exception as e:
        result['status'] = 'CRITICAL_ERROR'
        result['error'] = f"Critical Error: {str(e)}\n{traceback.format_exc()}"
    
    return result

def main():
    """Main test function"""
    print("=" * 60)
    print("SMARTCASH CELLS FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Define cells to test
    cells_to_test = [
        ('cell_1_2_colab.py', 'smartcash.ui.cells.cell_1_2_colab'),
        ('cell_1_3_dependency.py', 'smartcash.ui.cells.cell_1_3_dependency'),
        ('cell_2_1_downloader.py', 'smartcash.ui.cells.cell_2_1_downloader'),
        ('cell_2_2_split.py', 'smartcash.ui.cells.cell_2_2_split'),
        ('cell_2_3_preprocess.py', 'smartcash.ui.cells.cell_2_3_preprocess'),
        ('cell_2_4_augment.py', 'smartcash.ui.cells.cell_2_4_augment'),
        ('cell_2_5_visualize.py', 'smartcash.ui.cells.cell_2_5_visualize'),
        ('cell_3_1_pretrained.py', 'smartcash.ui.cells.cell_3_1_pretrained'),
        ('cell_3_2_backbone.py', 'smartcash.ui.cells.cell_3_2_backbone'),
        ('cell_3_3_train.py', 'smartcash.ui.cells.cell_3_3_train'),
        ('cell_3_4_evaluate.py', 'smartcash.ui.cells.cell_3_4_evaluate'),
    ]
    
    results = []
    success_count = 0
    
    for cell_name, cell_path in cells_to_test:
        print(f"\nTesting {cell_name}...")
        result = test_cell_import_and_execution(cell_name, cell_path)
        results.append(result)
        
        # Print immediate result
        status_symbol = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_symbol} {cell_name}: {result['status']}")
        
        if result['status'] == 'SUCCESS':
            success_count += 1
        
        if result['error']:
            print(f"   Error: {result['error']}")
        
        if result['output']:
            print(f"   Output: {result['output'][:200]}...")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cells tested: {len(cells_to_test)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(cells_to_test) - success_count}")
    print(f"Success rate: {(success_count / len(cells_to_test)) * 100:.1f}%")
    
    # Detailed results
    print("\nDETAILED RESULTS:")
    print("-" * 60)
    
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Status: {result['status']}")
        print(f"  Import Success: {result['import_success']}")
        print(f"  Execution Success: {result['execution_success']}")
        
        if result['error']:
            print(f"  Error: {result['error']}")
            
        if result['output']:
            print(f"  Output: {result['output'][:100]}...")
    
    return results

if __name__ == "__main__":
    main()