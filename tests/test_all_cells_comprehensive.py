#!/usr/bin/env python3
"""
Comprehensive test script for all SmartCash UI cell files.
Tests both import functionality and UI display patterns.
"""

import sys
import os
import traceback
import logging
import json
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the project root to Python path
sys.path.insert(0, '/Users/masdevid/Projects/smartcash')

class CellTester:
    """Comprehensive cell testing framework"""
    
    def __init__(self):
        self.results = []
        self.test_timestamp = datetime.now().isoformat()
        
    def test_cell_import(self, cell_name: str, cell_module_path: str) -> Dict[str, Any]:
        """Test cell import functionality"""
        result = {
            'test_type': 'import',
            'cell_name': cell_name,
            'status': 'UNKNOWN',
            'error': None,
            'import_time': None
        }
        
        try:
            start_time = datetime.now()
            
            # Try to import the cell module
            module_parts = cell_module_path.split('.')
            module = __import__(cell_module_path, fromlist=[module_parts[-1]])
            
            end_time = datetime.now()
            result['import_time'] = (end_time - start_time).total_seconds()
            result['status'] = 'SUCCESS'
            
        except ImportError as e:
            result['status'] = 'IMPORT_ERROR'
            result['error'] = str(e)
        except Exception as e:
            result['status'] = 'EXECUTION_ERROR'
            result['error'] = str(e)
            
        return result
    
    def test_cell_ui_display(self, cell_name: str, initializer_path: str, 
                           function_name: str) -> Dict[str, Any]:
        """Test UI display pattern implementation"""
        result = {
            'test_type': 'ui_display',
            'cell_name': cell_name,
            'status': 'UNKNOWN',
            'error': None,
            'ui_displayed': False,
            'logs_suppressed': False,
            'display_function_exists': False
        }
        
        try:
            # Import the initializer module
            module_parts = initializer_path.split('.')
            module = __import__(initializer_path, fromlist=[module_parts[-1]])
            
            # Check if display function exists
            if hasattr(module, function_name):
                result['display_function_exists'] = True
                
                # Capture output to test logging suppression
                stdout_capture = StringIO()
                stderr_capture = StringIO()
                
                # Temporarily suppress logging to test the pattern
                root_logger = logging.getLogger()
                original_level = root_logger.level
                
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    try:
                        # Call the display function (should suppress early logs)
                        display_func = getattr(module, function_name)
                        display_func()  # This should display UI, not return dict
                        
                        result['ui_displayed'] = True
                        result['status'] = 'SUCCESS'
                        
                        # Check if logs were properly managed
                        stdout_content = stdout_capture.getvalue()
                        stderr_content = stderr_capture.getvalue()
                        
                        # Early logs should be minimal (suppressed)
                        if len(stdout_content) < 1000:  # Arbitrary threshold
                            result['logs_suppressed'] = True
                            
                    except Exception as e:
                        result['status'] = 'UI_ERROR'
                        result['error'] = f"UI Display Error: {str(e)}"
                    finally:
                        # Restore logging level
                        root_logger.setLevel(original_level)
            else:
                result['status'] = 'FUNCTION_NOT_FOUND'
                result['error'] = f"Function {function_name} not found in {initializer_path}"
                
        except ImportError as e:
            result['status'] = 'IMPORT_ERROR'
            result['error'] = str(e)
        except Exception as e:
            result['status'] = 'CRITICAL_ERROR'
            result['error'] = str(e)
            
        return result
    
    def test_all_cells(self) -> List[Dict[str, Any]]:
        """Test all cell files comprehensively"""
        
        # Define all cell configurations
        cell_configs = [
            {
                'name': 'cell_1_1_repo_clone',
                'cell_module': 'smartcash.ui.cells.cell_1_1_repo_clone',
                'initializer_module': 'smartcash.ui.setup.repo.repo_initializer',
                'display_function': 'initialize_repo_ui'
            },
            {
                'name': 'cell_1_2_colab',
                'cell_module': 'smartcash.ui.cells.cell_1_2_colab',
                'initializer_module': 'smartcash.ui.setup.colab.colab_initializer',
                'display_function': 'initialize_colab_ui'
            },
            {
                'name': 'cell_1_3_dependency',
                'cell_module': 'smartcash.ui.cells.cell_1_3_dependency',
                'initializer_module': 'smartcash.ui.setup.dependency.dependency_initializer',
                'display_function': 'initialize_dependency_ui'
            },
            {
                'name': 'cell_2_1_downloader',
                'cell_module': 'smartcash.ui.cells.cell_2_1_downloader',
                'initializer_module': 'smartcash.ui.dataset.downloader.downloader_initializer',
                'display_function': 'initialize_downloader_ui'
            },
            {
                'name': 'cell_2_2_split',
                'cell_module': 'smartcash.ui.cells.cell_2_2_split',
                'initializer_module': 'smartcash.ui.dataset.split.split_initializer',
                'display_function': 'initialize_split_ui'
            },
            {
                'name': 'cell_2_3_preprocess',
                'cell_module': 'smartcash.ui.cells.cell_2_3_preprocess',
                'initializer_module': 'smartcash.ui.dataset.preprocess.preprocess_initializer',
                'display_function': 'initialize_preprocess_ui'
            },
            {
                'name': 'cell_2_4_augment',
                'cell_module': 'smartcash.ui.cells.cell_2_4_augment',
                'initializer_module': 'smartcash.ui.dataset.augment.augment_initializer',
                'display_function': 'initialize_augment_ui'
            },
            {
                'name': 'cell_2_5_visualize',
                'cell_module': 'smartcash.ui.cells.cell_2_5_visualize',
                'initializer_module': 'smartcash.ui.dataset.visualization.visualization_initializer',
                'display_function': 'initialize_visualization_ui'
            },
            {
                'name': 'cell_3_1_pretrained',
                'cell_module': 'smartcash.ui.cells.cell_3_1_pretrained',
                'initializer_module': 'smartcash.ui.model.pretrained.pretrained_initializer',
                'display_function': 'initialize_pretrained_ui'
            },
            {
                'name': 'cell_3_2_backbone',
                'cell_module': 'smartcash.ui.cells.cell_3_2_backbone',
                'initializer_module': 'smartcash.ui.model.backbone.backbone_initializer',
                'display_function': 'initialize_backbone_ui'
            },
            {
                'name': 'cell_3_3_train',
                'cell_module': 'smartcash.ui.cells.cell_3_3_train',
                'initializer_module': 'smartcash.ui.model.train.training_initializer',
                'display_function': 'initialize_training_ui'
            },
            {
                'name': 'cell_3_4_evaluate',
                'cell_module': 'smartcash.ui.cells.cell_3_4_evaluate',
                'initializer_module': 'smartcash.ui.model.evaluate.evaluation_initializer',
                'display_function': 'initialize_evaluation_ui'
            }
        ]
        
        all_results = []
        
        print("🔬 COMPREHENSIVE SMARTCASH CELL TESTING")
        print("=" * 80)
        print(f"Testing {len(cell_configs)} cell modules...")
        print(f"Timestamp: {self.test_timestamp}")
        print("=" * 80)
        
        for i, config in enumerate(cell_configs, 1):
            print(f"\n[{i:2d}/{len(cell_configs)}] Testing {config['name']}...")
            print("-" * 60)
            
            # Test 1: Cell Import
            print("  📦 Testing cell import...")
            import_result = self.test_cell_import(
                config['name'], 
                config['cell_module']
            )
            
            status_symbol = "✅" if import_result['status'] == 'SUCCESS' else "❌"
            print(f"    {status_symbol} Import: {import_result['status']}")
            if import_result['error']:
                print(f"       Error: {import_result['error']}")
            
            all_results.append(import_result)
            
            # Test 2: UI Display Pattern
            print("  🎨 Testing UI display pattern...")
            ui_result = self.test_cell_ui_display(
                config['name'],
                config['initializer_module'],
                config['display_function']
            )
            
            status_symbol = "✅" if ui_result['status'] == 'SUCCESS' else "❌"
            print(f"    {status_symbol} UI Display: {ui_result['status']}")
            
            if ui_result['display_function_exists']:
                print(f"       ✅ Display function exists")
            if ui_result['ui_displayed']:
                print(f"       ✅ UI displayed correctly")
            if ui_result['logs_suppressed']:
                print(f"       ✅ Logging properly managed")
            
            if ui_result['error']:
                print(f"       Error: {ui_result['error']}")
                
            all_results.append(ui_result)
            
        self.results = all_results
        return all_results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary statistics"""
        
        import_tests = [r for r in self.results if r['test_type'] == 'import']
        ui_tests = [r for r in self.results if r['test_type'] == 'ui_display']
        
        import_success = len([r for r in import_tests if r['status'] == 'SUCCESS'])
        ui_success = len([r for r in ui_tests if r['status'] == 'SUCCESS'])
        
        summary = {
            'timestamp': self.test_timestamp,
            'total_cells': len(import_tests),
            'import_tests': {
                'total': len(import_tests),
                'successful': import_success,
                'failed': len(import_tests) - import_success,
                'success_rate': (import_success / len(import_tests)) * 100 if import_tests else 0
            },
            'ui_display_tests': {
                'total': len(ui_tests),
                'successful': ui_success,
                'failed': len(ui_tests) - ui_success,
                'success_rate': (ui_success / len(ui_tests)) * 100 if ui_tests else 0
            },
            'overall_success_rate': ((import_success + ui_success) / (len(import_tests) + len(ui_tests))) * 100 if self.results else 0
        }
        
        return summary
    
    def print_detailed_report(self):
        """Print detailed test report"""
        summary = self.generate_summary()
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"🕒 Test Timestamp: {summary['timestamp']}")
        print(f"📁 Total Cells Tested: {summary['total_cells']}")
        print()
        
        print("📦 IMPORT TESTS:")
        print(f"   ✅ Successful: {summary['import_tests']['successful']}")
        print(f"   ❌ Failed: {summary['import_tests']['failed']}")
        print(f"   📈 Success Rate: {summary['import_tests']['success_rate']:.1f}%")
        print()
        
        print("🎨 UI DISPLAY TESTS:")
        print(f"   ✅ Successful: {summary['ui_display_tests']['successful']}")
        print(f"   ❌ Failed: {summary['ui_display_tests']['failed']}")
        print(f"   📈 Success Rate: {summary['ui_display_tests']['success_rate']:.1f}%")
        print()
        
        print(f"🎯 OVERALL SUCCESS RATE: {summary['overall_success_rate']:.1f}%")
        
        # Detailed breakdown by module
        print("\n" + "=" * 80)
        print("📋 DETAILED RESULTS BY MODULE")
        print("=" * 80)
        
        # Group results by cell name
        cell_results = {}
        for result in self.results:
            cell_name = result['cell_name']
            if cell_name not in cell_results:
                cell_results[cell_name] = {}
            cell_results[cell_name][result['test_type']] = result
            
        for cell_name, tests in cell_results.items():
            print(f"\n📱 {cell_name}:")
            
            if 'import' in tests:
                import_test = tests['import']
                status = "✅ PASS" if import_test['status'] == 'SUCCESS' else "❌ FAIL"
                print(f"   📦 Import: {status}")
                if import_test['error']:
                    print(f"      Error: {import_test['error']}")
                    
            if 'ui_display' in tests:
                ui_test = tests['ui_display']
                status = "✅ PASS" if ui_test['status'] == 'SUCCESS' else "❌ FAIL"
                print(f"   🎨 UI Display: {status}")
                if ui_test['error']:
                    print(f"      Error: {ui_test['error']}")
                if ui_test['display_function_exists']:
                    print(f"      ✅ Display function found")
                if ui_test['ui_displayed']:
                    print(f"      ✅ UI displayed")
                if ui_test['logs_suppressed']:
                    print(f"      ✅ Logging managed")
    
    def save_results(self, filename: str = "cell_test_results.json"):
        """Save test results to JSON file"""
        results_data = {
            'summary': self.generate_summary(),
            'detailed_results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
            
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Main test execution function"""
    
    # Initialize tester
    tester = CellTester()
    
    try:
        # Run comprehensive tests
        results = tester.test_all_cells()
        
        # Print detailed report
        tester.print_detailed_report()
        
        # Save results
        tester.save_results()
        
        # Return summary for further processing
        return tester.generate_summary()
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in test execution:")
        print(f"   {str(e)}")
        print(f"   {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    summary = main()
    
    if summary:
        # Exit with appropriate code
        if summary['overall_success_rate'] >= 80:
            print("\n🎉 Test execution completed successfully!")
            sys.exit(0)
        elif summary['overall_success_rate'] >= 50:
            print("\n⚠️  Test execution completed with issues!")
            sys.exit(1)
        else:
            print("\n💥 Test execution failed!")
            sys.exit(2)
    else:
        print("\n💥 Critical test failure!")
        sys.exit(3)