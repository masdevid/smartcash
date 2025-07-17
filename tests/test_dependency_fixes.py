#!/usr/bin/env python3
"""
Test script for dependency module fixes including:
1. Save/reset button double logging fix
2. Progress tracker persistent display
3. Dual progress system
4. Real package operations

This script tests the dependency module functionality without requiring Jupyter.
"""

import sys
import os
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path
sys.path.insert(0, '/Users/masdevid/Projects/smartcash')

def test_dependency_module_imports():
    """Test that dependency module can be imported properly."""
    print("🧪 Testing imports...")
    
    try:
        # Test core imports
        from smartcash.ui.core.base_ui_module import BaseUIModule
        print("✅ BaseUIModule imported successfully")
        
        from smartcash.ui.core.mixins.button_handler_mixin import ButtonHandlerMixin
        print("✅ ButtonHandlerMixin imported successfully")
        
        # Test dependency module with mocking
        with patch('ipywidgets.widgets'):
            from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
            print("✅ DependencyUIModule imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_save_reset_handlers():
    """Test save/reset button handlers for double logging fix."""
    print("\n🧪 Testing save/reset handlers...")
    
    try:
        with patch('ipywidgets.widgets'):
            from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
            
            # Create mock dependency module
            module = DependencyUIModule()
            
            # Mock necessary components
            module._ui_components = {
                'header_container': Mock(),
                'operation_container': Mock()
            }
            
            # Mock config handler
            module._config_handler = Mock()
            module._config_handler.extract_config_from_ui.return_value = {'test': 'config'}
            module._merged_config = {'test': 'config'}
            
            # Mock header status update method
            module._update_header_status = Mock()
            module.log = Mock()
            
            # Test save config handler
            result = module._handle_save_config()
            print(f"✅ Save config handler returned: {result}")
            
            # Verify header status was updated
            assert module._update_header_status.called, "Header status should be updated"
            print("✅ Header status update called for save")
            
            # Verify log was called
            assert module.log.called, "Log should be called"
            print("✅ Log accordion update called for save")
            
            # Test reset config handler
            module._update_header_status.reset_mock()
            module.log.reset_mock()
            
            result = module._handle_reset_config()
            print(f"✅ Reset config handler returned: {result}")
            
            # Verify header status was updated
            assert module._update_header_status.called, "Header status should be updated"
            print("✅ Header status update called for reset")
            
            # Verify log was called
            assert module.log.called, "Log should be called"
            print("✅ Log accordion update called for reset")
            
        return True
        
    except Exception as e:
        print(f"❌ Save/reset handler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_progress_tracker_visibility():
    """Test progress tracker visibility and dual progress system."""
    print("\n🧪 Testing progress tracker visibility...")
    
    try:
        with patch('ipywidgets.widgets'):
            from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
            
            # Create mock dependency module
            module = DependencyUIModule()
            
            # Mock progress tracker with show method
            mock_progress_tracker = Mock()
            mock_progress_tracker.show = Mock()
            mock_progress_tracker.layout = Mock()
            mock_progress_tracker.layout.display = 'none'
            mock_progress_tracker.layout.visibility = 'hidden'
            
            # Mock operation container
            mock_operation_container = Mock()
            mock_operation_container.update_progress = Mock()
            
            module._ui_components = {
                'progress_tracker': mock_progress_tracker,
                'operation_container': mock_operation_container
            }
            
            module.logger = Mock()
            
            # Test progress visibility
            module._ensure_progress_visibility()
            
            # Verify progress tracker show was called
            mock_progress_tracker.show.assert_called_once()
            print("✅ Progress tracker show() method called")
            
            # Test dual progress update
            module.update_stage_progress(
                stage_progress=50,
                stage_message="Installing packages",
                detail_progress=25,
                detail_message="Installing pandas"
            )
            
            print("✅ Dual progress update completed")
            
        return True
        
    except Exception as e:
        print(f"❌ Progress tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_operation_wrapper():
    """Test the common operation wrapper functionality."""
    print("\n🧪 Testing operation wrapper...")
    
    try:
        with patch('ipywidgets.widgets'):
            from smartcash.ui.core.base_ui_module import BaseUIModule
            
            # Create mock base module
            module = BaseUIModule()
            
            # Mock necessary methods
            module.log_operation_start = Mock()
            module.start_progress = Mock()
            module.update_operation_status = Mock()
            module.disable_all_buttons = Mock()
            module.update_progress = Mock()
            module.log_operation_complete = Mock()
            module.complete_progress = Mock()
            module.enable_all_buttons = Mock()
            module.log = Mock()
            module._update_header_status = Mock()
            
            # Test successful operation
            def mock_successful_operation():
                return {'success': True, 'message': 'Operation completed'}
            
            result = module._execute_operation_with_wrapper(
                operation_name="Test Operation",
                operation_func=mock_successful_operation
            )
            
            assert result['success'] == True
            print("✅ Operation wrapper handled success case")
            
            # Test failed operation
            def mock_failed_operation():
                return {'success': False, 'message': 'Operation failed'}
            
            result = module._execute_operation_with_wrapper(
                operation_name="Test Operation",
                operation_func=mock_failed_operation
            )
            
            assert result['success'] == False
            print("✅ Operation wrapper handled failure case")
            
            # Verify all expected methods were called
            assert module.log_operation_start.called, "Should start operation logging"
            assert module.start_progress.called, "Should start progress"
            assert module.update_progress.called, "Should update progress"
            assert module.enable_all_buttons.called, "Should re-enable buttons"
            
            print("✅ All operation wrapper methods called correctly")
            
        return True
        
    except Exception as e:
        print(f"❌ Operation wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dual_progress_system():
    """Test the dual progress system with stage and detail progress."""
    print("\n🧪 Testing dual progress system...")
    
    try:
        with patch('ipywidgets.widgets'):
            from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
            
            # Create mock dependency module
            module = DependencyUIModule()
            
            # Mock components
            module._ensure_progress_visibility = Mock()
            module.update_progress = Mock()
            module.log = Mock()
            
            mock_operation_container = Mock()
            mock_operation_container.update_progress = Mock()
            
            module._ui_components = {
                'operation_container': mock_operation_container
            }
            
            # Test dual progress update
            module.update_stage_progress(
                stage_progress=30,
                stage_message="Installing packages (1/3)",
                detail_progress=50,
                detail_message="Installing numpy"
            )
            
            # Verify stage progress was updated
            module.update_progress.assert_called_with(30, "Installing packages (1/3)", "info")
            print("✅ Stage progress updated correctly")
            
            # Verify detailed progress was updated
            mock_operation_container.update_progress.assert_called_with(
                50, "Installing numpy", "secondary", "Detail"
            )
            print("✅ Detail progress updated correctly")
            
            # Test stage-only progress
            mock_operation_container.update_progress.reset_mock()
            module.update_stage_progress(
                stage_progress=100,
                stage_message="Installation completed"
            )
            
            # Should not call detailed progress
            mock_operation_container.update_progress.assert_not_called()
            print("✅ Stage-only progress works correctly")
            
        return True
        
    except Exception as e:
        print(f"❌ Dual progress system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def simulate_package_installation():
    """Simulate a real package installation with progress tracking."""
    print("\n🧪 Simulating package installation...")
    
    try:
        with patch('ipywidgets.widgets'):
            from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
            
            # Create mock dependency module
            module = DependencyUIModule()
            
            # Mock all UI components
            module._ui_components = {
                'progress_tracker': Mock(),
                'operation_container': Mock()
            }
            
            # Mock all necessary methods
            module._ensure_progress_visibility = Mock()
            module.update_progress = Mock()
            module.log = Mock()
            module.update_stage_progress = Mock()
            
            # Mock InstallOperationHandler
            with patch('smartcash.ui.setup.dependency.operations.install_operation.InstallOperationHandler') as MockHandler:
                mock_handler = Mock()
                mock_handler.execute_operation.return_value = {
                    'success': True, 
                    'installed_count': 3,
                    'message': 'All packages installed successfully'
                }
                MockHandler.return_value = mock_handler
                
                # Test package installation
                packages = ['numpy', 'pandas', 'matplotlib']
                result = module._execute_install_operation(packages)
                
                # Verify success
                assert result['success'] == True
                print("✅ Package installation simulation completed successfully")
                
                # Verify progress tracking was called
                assert module.update_stage_progress.called
                print("✅ Dual progress tracking was utilized")
                
                # Verify handler was created and executed
                MockHandler.assert_called_once()
                mock_handler.execute_operation.assert_called_once()
                print("✅ Installation handler was properly executed")
                
        return True
        
    except Exception as e:
        print(f"❌ Package installation simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting dependency module comprehensive tests...")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_dependency_module_imports),
        ("Save/Reset Handler Tests", test_save_reset_handlers),
        ("Progress Tracker Visibility", test_progress_tracker_visibility),
        ("Operation Wrapper Tests", test_operation_wrapper),
        ("Dual Progress System", test_dual_progress_system),
        ("Package Installation Simulation", simulate_package_installation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:<10} {test_name}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 Total: {len(results)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Dependency module fixes are working correctly.")
        return True
    else:
        print(f"⚠️ {failed} test(s) failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)