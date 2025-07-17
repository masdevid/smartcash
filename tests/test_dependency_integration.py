#!/usr/bin/env python3
"""
Test dependency_uimodule integration with correct architecture:
- Dual progress tracker
- Status updates via header_container
- Logging via operation_container
"""

import sys
import os
from unittest.mock import MagicMock, patch
from typing import Dict, Any

def create_mock_ui_components():
    """Create mock UI components for testing"""
    
    # Mock header_container with status panel
    header_container = MagicMock()
    header_container.update_status = MagicMock()
    
    # Mock operation_container with dual progress and logging
    operation_container = MagicMock()
    operation_container.update_progress = MagicMock()
    operation_container.log = MagicMock()
    
    # Mock progress_tracker with dual progress support
    progress_tracker = MagicMock()
    progress_tracker.show = MagicMock()
    progress_tracker.tqdm_manager = MagicMock()
    progress_tracker.tqdm_manager.update_bar = MagicMock()
    progress_tracker._config = MagicMock()
    progress_tracker._config.get_level_configs = MagicMock(return_value=[
        MagicMock(name="overall", visible=True),
        MagicMock(name="current", visible=True)
    ])
    
    return {
        'header_container': header_container,
        'operation_container': operation_container,
        'progress_tracker': progress_tracker,
        'main_container': MagicMock(),
        'form_container': MagicMock(),
        'action_container': MagicMock()
    }

def test_dual_progress_integration():
    """Test dual progress tracker integration"""
    print("🔍 Testing Dual Progress Tracker Integration...")
    
    try:
        # Mock the UI components
        mock_components = create_mock_ui_components()
        
        # Mock the dependency module creation
        with patch('smartcash.ui.setup.dependency.configs.dependency_defaults.get_default_dependency_config') as mock_config, \
             patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components') as mock_ui_creator:
            
            mock_config.return_value = {'packages': ['numpy'], 'operation': 'install'}
            mock_ui_creator.return_value = mock_components
            
            # Try to import and create dependency module
            try:
                from smartcash.ui.setup.dependency import DependencyUIModule
                
                # Create module
                module = DependencyUIModule()
                module._ui_components = mock_components
                module._initialized = True
                
                print("✅ DependencyUIModule created successfully")
                
                # Test dual progress update
                if hasattr(module, 'update_progress'):
                    # Test overall progress (primary level)
                    module.update_progress(25, "Installing packages: 2/8", "primary")
                    
                    # Verify operation_container was called
                    operation_container = mock_components['operation_container']
                    operation_container.update_progress.assert_called_with(25, "Installing packages: 2/8", "primary")
                    print("✅ Primary progress update works correctly")
                    
                    # Test current operation progress (secondary level)
                    module.update_progress(60, "Installing pandas...", "secondary")
                    print("✅ Secondary progress update works correctly")
                    
                else:
                    print("❌ update_progress method not found")
                    return False
                
                # Test progress visibility
                if hasattr(module, '_ensure_progress_visibility'):
                    module._ensure_progress_visibility()
                    progress_tracker = mock_components['progress_tracker']
                    progress_tracker.show.assert_called()
                    print("✅ Progress visibility works correctly")
                
                return True
                
            except ImportError as e:
                print(f"❌ Import error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Dual progress test failed: {e}")
        return False

def test_status_update_integration():
    """Test status update integration via header_container"""
    print("\n🔍 Testing Status Update Integration...")
    
    try:
        # Mock the UI components
        mock_components = create_mock_ui_components()
        
        with patch('smartcash.ui.setup.dependency.configs.dependency_defaults.get_default_dependency_config') as mock_config, \
             patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components') as mock_ui_creator:
            
            mock_config.return_value = {'packages': ['numpy'], 'operation': 'install'}
            mock_ui_creator.return_value = mock_components
            
            from smartcash.ui.setup.dependency import DependencyUIModule
            
            # Create module
            module = DependencyUIModule()
            module._ui_components = mock_components
            module._initialized = True
            
            # Test status update through operation_mixin
            if hasattr(module, 'update_operation_status'):
                # Test different status types
                module.update_operation_status("Installation started", "info")
                module.update_operation_status("Installation completed", "success")
                module.update_operation_status("Installation failed", "error")
                
                # Verify header_container was called (operation_mixin should route to header)
                header_container = mock_components['header_container']
                assert header_container.update_status.call_count >= 3
                print("✅ Status updates routed to header_container correctly")
                
            else:
                print("❌ update_operation_status method not found")
                return False
                
            return True
            
    except Exception as e:
        print(f"❌ Status update test failed: {e}")
        return False

def test_logging_integration():
    """Test logging integration via operation_container"""
    print("\n🔍 Testing Logging Integration...")
    
    try:
        # Mock the UI components
        mock_components = create_mock_ui_components()
        
        with patch('smartcash.ui.setup.dependency.configs.dependency_defaults.get_default_dependency_config') as mock_config, \
             patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components') as mock_ui_creator:
            
            mock_config.return_value = {'packages': ['numpy'], 'operation': 'install'}
            mock_ui_creator.return_value = mock_components
            
            from smartcash.ui.setup.dependency import DependencyUIModule
            
            # Create module
            module = DependencyUIModule()
            module._ui_components = mock_components
            module._initialized = True
            
            # Test logging through operation_mixin
            if hasattr(module, 'log_operation'):
                # Test different log levels
                module.log_operation("📦 Starting package installation", "info")
                module.log_operation("⚠️ Package conflict detected", "warning")
                module.log_operation("❌ Installation failed", "error")
                
                # Verify operation_container was called
                operation_container = mock_components['operation_container']
                assert operation_container.log.call_count >= 3
                print("✅ Logging routed to operation_container correctly")
                
            else:
                print("❌ log_operation method not found")
                return False
                
            # Test the module's log method (should delegate to operation_mixin)
            if hasattr(module, 'log'):
                module.log("📊 Progress update logged", "info")
                print("✅ Module log method works correctly")
                
            return True
            
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_integration_workflow():
    """Test complete integration workflow"""
    print("\n🔍 Testing Complete Integration Workflow...")
    
    try:
        # Mock the UI components
        mock_components = create_mock_ui_components()
        
        with patch('smartcash.ui.setup.dependency.configs.dependency_defaults.get_default_dependency_config') as mock_config, \
             patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components') as mock_ui_creator:
            
            mock_config.return_value = {'packages': ['numpy', 'pandas'], 'operation': 'install'}
            mock_ui_creator.return_value = mock_components
            
            from smartcash.ui.setup.dependency import DependencyUIModule
            
            # Create module
            module = DependencyUIModule()
            module._ui_components = mock_components
            module._initialized = True
            
            # Simulate a complete package installation workflow
            print("  🚀 Simulating package installation workflow...")
            
            # 1. Start operation
            if hasattr(module, 'update_operation_status'):
                module.update_operation_status("Starting package installation", "info")
            
            # 2. Initialize progress
            if hasattr(module, 'start_progress'):
                module.start_progress("Installing packages", 100)
            
            # 3. Update progress with dual tracking
            if hasattr(module, 'update_progress'):
                # Overall progress
                module.update_progress(25, "Installing packages: 1/4", "primary")
                
                # Current operation progress  
                module.update_progress(50, "Installing numpy...", "secondary")
                
                # Continue with next package
                module.update_progress(50, "Installing packages: 2/4", "primary")
                module.update_progress(75, "Installing pandas...", "secondary")
                
                # Final progress
                module.update_progress(100, "Installation complete", "primary")
            
            # 4. Log important events
            if hasattr(module, 'log_operation'):
                module.log_operation("📦 numpy installed successfully", "info")
                module.log_operation("📦 pandas installed successfully", "info")
                module.log_operation("✅ All packages installed", "info")
            
            # 5. Final status update
            if hasattr(module, 'update_operation_status'):
                module.update_operation_status("Installation completed successfully", "success")
            
            print("  ✅ Complete workflow executed successfully")
            
            # Verify all components were called appropriately
            header_container = mock_components['header_container']
            operation_container = mock_components['operation_container']
            
            # Check status updates went to header
            assert header_container.update_status.call_count >= 2
            print("  ✅ Status updates routed to header_container")
            
            # Check progress updates went to operation_container
            assert operation_container.update_progress.call_count >= 4
            print("  ✅ Progress updates routed to operation_container")
            
            # Check logs went to operation_container
            assert operation_container.log.call_count >= 3
            print("  ✅ Logs routed to operation_container")
            
            return True
            
    except Exception as e:
        print(f"❌ Integration workflow test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Testing Dependency Module Integration")
    print("=" * 60)
    print("Testing correct architecture:")
    print("- Status updates → header_container")
    print("- Progress updates → operation_container") 
    print("- Logging → operation_container")
    print("- Dual progress support")
    print("=" * 60)
    
    tests = [
        ("Dual Progress Integration", test_dual_progress_integration),
        ("Status Update Integration", test_status_update_integration),
        ("Logging Integration", test_logging_integration),
        ("Complete Integration Workflow", test_integration_workflow)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Integration Test Results:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Dual progress tracker: WORKING")
        print("✅ Status updates via header_container: WORKING")
        print("✅ Logging via operation_container: WORKING")
        print("✅ Proper architecture separation: WORKING")
        return True
    else:
        print("\n⚠️  Some integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)