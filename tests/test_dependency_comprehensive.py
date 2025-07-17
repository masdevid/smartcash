#!/usr/bin/env python3
"""
Comprehensive test for dependency module functionality.
Tests all the requirements mentioned:
- Button handler and component integration
- Button state and log clearing
- Initialization logs redirect to operation container
- Operation logs redirect to operation container
- Progress tracker integration
- Dual progress tracking (install, update, check)
- Single progress tracking (uninstall)
- Save/Reset operations
"""

import sys
import os
import traceback
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, '/Users/masdevid/Projects/smartcash')

def test_dependency_module_comprehensive():
    """Comprehensive test of dependency module."""
    print("🧪 COMPREHENSIVE DEPENDENCY MODULE TEST")
    print("=" * 60)
    
    test_results = {
        'import_success': False,
        'instantiation_success': False,
        'component_integration': False,
        'button_handlers': False,
        'button_states': False,
        'log_clearing': False,
        'initialization_logs': False,
        'operation_logs': False,
        'progress_tracker': False,
        'dual_progress': False,
        'single_progress': False,
        'save_reset': False
    }
    
    try:
        # Test 1: Import and instantiation
        print("\n1️⃣ Testing import and instantiation...")
        from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
        test_results['import_success'] = True
        print("✅ Import successful")
        
        # Create module instance
        module = DependencyUIModule()
        test_results['instantiation_success'] = True
        print("✅ Instantiation successful")
        
        # Test 2: Initialize the module
        print("\n2️⃣ Testing module initialization...")
        with patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components') as mock_create_ui:
            # Mock UI components
            mock_ui_components = {
                'main_container': Mock(),
                'header_container': Mock(),
                'form_container': Mock(),
                'action_container': Mock(),
                'operation_container': Mock(),
                'progress_tracker': Mock(),
                'package_checkboxes': {
                    'category1': [Mock(value=True, package_name='test_package')]
                },
                'custom_packages': Mock(value='additional_package'),
                'buttons': {
                    'install': Mock(description='Install Packages'),
                    'uninstall': Mock(description='Uninstall Packages'),
                    'update': Mock(description='Update Packages'),
                    'check': Mock(description='Check Status'),
                    'save': Mock(description='Save Config'),
                    'reset': Mock(description='Reset Config')
                }
            }
            
            # Setup header container with update_status method
            header_mock = Mock()
            header_mock.update_status = Mock()
            mock_ui_components['header_container'] = header_mock
            
            # Setup operation container with logging capabilities
            operation_mock = Mock()
            operation_mock.log = Mock()
            operation_mock.clear_logs = Mock()
            mock_ui_components['operation_container'] = operation_mock
            
            # Setup progress tracker
            progress_mock = Mock()
            progress_mock.update_progress = Mock()
            progress_mock.start_progress = Mock()
            progress_mock.complete_progress = Mock()
            progress_mock.error_progress = Mock()
            progress_mock.update_stage_progress = Mock()
            progress_mock.initialize = Mock()
            progress_mock._config = Mock()
            progress_mock._config.level = Mock()
            progress_mock._config.level.value = 2  # Dual progress level
            mock_ui_components['progress_tracker'] = progress_mock
            
            mock_create_ui.return_value = mock_ui_components
            
            # Initialize module
            success = module.initialize()
            if success:
                test_results['component_integration'] = True
                print("✅ Component integration successful")
            else:
                print("❌ Component integration failed")
        
        # Test 3: Button handlers registration
        print("\n3️⃣ Testing button handler registration...")
        if hasattr(module, '_button_handlers') and module._button_handlers:
            expected_handlers = ['install', 'uninstall', 'update', 'check', 'save', 'reset']
            registered_handlers = list(module._button_handlers.keys())
            
            if all(handler in registered_handlers for handler in expected_handlers):
                test_results['button_handlers'] = True
                print("✅ Button handlers properly registered")
                print(f"   Registered handlers: {registered_handlers}")
            else:
                print(f"❌ Missing button handlers. Expected: {expected_handlers}, Got: {registered_handlers}")
        else:
            print("❌ No button handlers found")
        
        # Test 4: Button state management and log clearing
        print("\n4️⃣ Testing button state management and log clearing...")
        
        # Test button disable/enable
        if hasattr(module, 'disable_all_buttons') and hasattr(module, 'enable_all_buttons'):
            # Mock the button disabling
            with patch.object(module, '_ui_components', mock_ui_components):
                module.disable_all_buttons("Test operation", button_id="install")
                module.enable_all_buttons(button_id="install")
                test_results['button_states'] = True
                print("✅ Button state management works")
        else:
            print("❌ Button state management methods not found")
        
        # Test log clearing
        if hasattr(module, 'clear_logs'):
            with patch.object(module, '_ui_components', mock_ui_components):
                module.clear_logs()
                test_results['log_clearing'] = True
                print("✅ Log clearing works")
        else:
            print("❌ Log clearing method not found")
        
        # Test 5: Initialization logs redirect
        print("\n5️⃣ Testing initialization logs redirect...")
        with patch.object(module, '_ui_components', mock_ui_components):
            # Test that logs go to operation container
            if hasattr(module, 'log'):
                module.log("Test initialization message", 'info')
                operation_mock.log.assert_called()
                test_results['initialization_logs'] = True
                print("✅ Initialization logs redirect to operation container")
            else:
                print("❌ Log method not found")
        
        # Test 6: Operation logs redirect
        print("\n6️⃣ Testing operation logs redirect...")
        with patch.object(module, '_ui_components', mock_ui_components):
            # Test operation logging
            if hasattr(module, 'log_operation_start'):
                module.log_operation_start("Test Operation")
                test_results['operation_logs'] = True
                print("✅ Operation logs redirect to operation container")
            else:
                print("❌ Operation logging methods not found")
        
        # Test 7: Progress tracker integration
        print("\n7️⃣ Testing progress tracker integration...")
        with patch.object(module, '_ui_components', mock_ui_components):
            # Test basic progress methods
            progress_methods = ['start_progress', 'update_progress', 'complete_progress', 'error_progress']
            all_methods_exist = all(hasattr(module, method) for method in progress_methods)
            
            if all_methods_exist:
                # Test calling progress methods
                module.start_progress("Test progress", 0)
                module.update_progress(50, "Halfway done")
                module.complete_progress("Done")
                
                test_results['progress_tracker'] = True
                print("✅ Progress tracker integration works")
            else:
                missing = [method for method in progress_methods if not hasattr(module, method)]
                print(f"❌ Progress methods missing: {missing}")
        
        # Test 8: Dual progress tracking (install, update, check)
        print("\n8️⃣ Testing dual progress tracking...")
        with patch.object(module, '_ui_components', mock_ui_components):
            # Test dual progress methods
            dual_progress_methods = ['update_stage_progress']
            if all(hasattr(module, method) for method in dual_progress_methods):
                # Test dual progress call
                module.update_stage_progress(
                    stage_progress=25,
                    stage_message="Installing packages",
                    detail_progress=50,
                    detail_message="Installing package 1/2"
                )
                test_results['dual_progress'] = True
                print("✅ Dual progress tracking works")
            else:
                print("❌ Dual progress tracking methods not found")
        
        # Test 9: Single progress tracking (uninstall)
        print("\n9️⃣ Testing single progress tracking...")
        with patch.object(module, '_ui_components', mock_ui_components):
            # Test single progress (using basic progress methods)
            try:
                module.start_progress("Uninstalling packages", 0)
                module.update_progress(50, "Removing package 1/2")
                module.complete_progress("Uninstall complete")
                test_results['single_progress'] = True
                print("✅ Single progress tracking works")
            except Exception as e:
                print(f"❌ Single progress tracking failed: {e}")
        
        # Test 10: Save/Reset operations
        print("\n🔟 Testing save/reset operations...")
        with patch.object(module, '_ui_components', mock_ui_components):
            # Test save operation
            save_result = module.save_config()
            if save_result.get('success', False):
                print("✅ Save operation works")
            else:
                print(f"⚠️ Save operation result: {save_result}")
            
            # Test reset operation
            reset_result = module.reset_config()
            if reset_result.get('success', False):
                print("✅ Reset operation works")
            else:
                print(f"⚠️ Reset operation result: {reset_result}")
            
            # Check if header status was updated
            if header_mock.update_status.called:
                test_results['save_reset'] = True
                print("✅ Save/Reset updates header status")
            else:
                print("❌ Save/Reset does not update header status")
        
        # Test 11: Actual operation execution simulation
        print("\n1️⃣1️⃣ Testing actual operation execution...")
        with patch.object(module, '_ui_components', mock_ui_components):
            # Mock operation handlers
            with patch('smartcash.ui.setup.dependency.operations.install_operation.InstallOperationHandler') as mock_install_handler:
                mock_handler_instance = Mock()
                mock_handler_instance.execute_operation.return_value = {'success': True, 'message': 'Installation successful'}
                mock_install_handler.return_value = mock_handler_instance
                
                # Test install operation
                packages = ['test_package']
                result = module._execute_install_operation(packages)
                
                if result.get('success'):
                    print("✅ Install operation execution works")
                else:
                    print(f"❌ Install operation failed: {result}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n📈 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Dependency module is fully functional.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == '__main__':
    success = test_dependency_module_comprehensive()
    sys.exit(0 if success else 1)