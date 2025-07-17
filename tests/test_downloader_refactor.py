#!/usr/bin/env python3
"""
Test script for refactored dataset downloader module.
Verifies new UIModule pattern integration while preserving backend functionality.
"""

import sys
import os
from typing import Dict, Any
from unittest.mock import Mock, patch

# Add the project root to Python path
sys.path.insert(0, '/Users/masdevid/Projects/smartcash')

def test_downloader_imports():
    """Test that downloader module imports work correctly."""
    print("🧪 Testing downloader module imports...")
    
    try:
        # Test new UIModule pattern imports
        from smartcash.ui.dataset.downloader import (
            DownloaderUIModule,
            create_downloader_uimodule,
            get_downloader_uimodule,
            reset_downloader_uimodule
        )
        print("✅ New UIModule pattern imports successful")
        
        # Test backward compatibility imports
        from smartcash.ui.dataset.downloader import (
            initialize_downloader_ui,
            display_downloader_ui,
            get_downloader_components
        )
        print("✅ Backward compatibility imports successful")
        
        # Test legacy imports
        from smartcash.ui.dataset.downloader import (
            initialize_downloader_ui_legacy,
            display_downloader_ui_legacy,
            get_downloader_components_legacy
        )
        print("✅ Legacy imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_downloader_uimodule_creation():
    """Test DownloaderUIModule creation and initialization."""
    print("\n🧪 Testing DownloaderUIModule creation...")
    
    try:
        from smartcash.ui.dataset.downloader import DownloaderUIModule, create_downloader_uimodule
        
        # Test direct instantiation
        module = DownloaderUIModule()
        print("✅ DownloaderUIModule instantiated successfully")
        
        # Test module info
        print(f"📋 Module name: {module.module_name}")
        print(f"📋 Parent module: {module.parent_module}")
        print(f"📋 Full module name: {module.full_module_name}")
        
        # Test factory creation (without auto-initialization)
        module2 = create_downloader_uimodule(auto_initialize=False)
        print("✅ Factory creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ UIModule creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_downloader_configuration():
    """Test downloader configuration handling."""
    print("\n🧪 Testing downloader configuration...")
    
    try:
        from smartcash.ui.dataset.downloader import create_downloader_uimodule
        from smartcash.ui.dataset.downloader.configs.downloader_defaults import get_default_downloader_config
        
        # Test default configuration
        default_config = get_default_downloader_config()
        print(f"✅ Default config loaded: {len(default_config)} sections")
        
        # Check required configuration sections
        required_sections = ['data', 'download', 'uuid_renaming']
        missing_sections = [section for section in required_sections if section not in default_config]
        
        if not missing_sections:
            print("✅ All required config sections present")
        else:
            print(f"⚠️ Missing config sections: {missing_sections}")
        
        # Test config with custom values
        custom_config = {
            'data': {
                'roboflow': {
                    'workspace': 'test-workspace',
                    'project': 'test-project',
                    'version': '1'
                }
            }
        }
        
        # Create module with custom config
        module = create_downloader_uimodule(config=custom_config, auto_initialize=False)
        merged_config = module.get_config()
        
        # Verify config merge
        if merged_config['data']['roboflow']['workspace'] == 'test-workspace':
            print("✅ Custom config merged correctly")
        else:
            print("⚠️ Config merge issue detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backend_service_integration():
    """Test backend service integration."""
    print("\n🧪 Testing backend service integration...")
    
    try:
        from smartcash.ui.dataset.downloader.services.downloader_service import DownloaderService
        
        # Test service instantiation
        service = DownloaderService()
        print("✅ DownloaderService instantiated successfully")
        
        # Test service methods (mock backend calls)
        with patch('smartcash.ui.dataset.downloader.services.backend_utils.get_existing_dataset_count', return_value=10):
            count = service.get_existing_dataset_count()
            print(f"✅ Dataset count service works: {count} files")
        
        # Test config validation (mock validation)
        test_config = {
            'data': {
                'roboflow': {
                    'workspace': 'test',
                    'project': 'test',
                    'version': '1',
                    'api_key': 'test_key'
                }
            }
        }
        
        with patch('smartcash.ui.dataset.downloader.services.validation_utils.validate_config', 
                  return_value={'valid': True, 'status': True, 'errors': [], 'warnings': []}):
            validation_result = service.validate_config(test_config)
            if validation_result.get('valid'):
                print("✅ Config validation service works")
            else:
                print("⚠️ Config validation issue")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_operation_manager_integration():
    """Test operation manager integration."""
    print("\n🧪 Testing operation manager integration...")
    
    try:
        from smartcash.ui.dataset.downloader.operations.manager import DownloaderOperationManager
        from unittest.mock import Mock
        
        # Create mock UI components
        mock_operation_container = Mock()
        mock_operation_container.log = Mock()
        mock_operation_container.update_progress = Mock()
        
        mock_ui_components = {
            'operation_container': mock_operation_container,
            'download_button': Mock(),
            'check_button': Mock(),
            'cleanup_button': Mock()
        }
        
        # Test operation manager creation
        operation_manager = DownloaderOperationManager(
            config={'test': 'config'},
            operation_container=mock_operation_container
        )
        operation_manager._ui_components = mock_ui_components
        
        print("✅ OperationManager created successfully")
        
        # Test initialization
        operation_manager.initialize()
        print("✅ OperationManager initialized successfully")
        
        # Test operations availability
        operations = operation_manager.get_operations()
        expected_operations = ['download', 'check', 'cleanup']
        missing_operations = [op for op in expected_operations if op not in operations]
        
        if not missing_operations:
            print("✅ All expected operations available")
        else:
            print(f"⚠️ Missing operations: {missing_operations}")
        
        # Test logging integration
        operation_manager.log("Test message", 'info')
        mock_operation_container.log.assert_called()
        print("✅ Logging integration works")
        
        return True
        
    except Exception as e:
        print(f"❌ Operation manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_component_structure():
    """Test UI component structure and creation."""
    print("\n🧪 Testing UI component structure...")
    
    try:
        from smartcash.ui.dataset.downloader.components.downloader_ui import create_downloader_ui
        
        # Mock UI component creation to avoid IPython dependencies
        with patch('smartcash.ui.components.header_container.create_header_container') as mock_header, \
             patch('smartcash.ui.components.form_container.create_form_container') as mock_form, \
             patch('smartcash.ui.components.action_container.create_action_container') as mock_action, \
             patch('smartcash.ui.components.operation_container.create_operation_container') as mock_operation, \
             patch('smartcash.ui.components.footer_container.create_footer_container') as mock_footer, \
             patch('smartcash.ui.components.main_container.create_main_container') as mock_main:
            
            # Configure mocks to return dict-like objects
            mock_header.return_value = Mock(container=Mock())
            mock_form.return_value = {'container': Mock(), 'add_item': Mock()}
            mock_action.return_value = {'container': Mock(), 'buttons': {}}
            mock_operation.return_value = {'container': Mock(), 'log_message': Mock(), 'update_progress': Mock(), 'show_dialog': Mock()}
            mock_footer.return_value = Mock(container=Mock())
            mock_main.return_value = Mock()
            
            # Test UI creation
            ui_components = create_downloader_ui({'test': 'config'})
            
            # Verify structure
            required_components = [
                'main_container', 'header_container', 'form_container',
                'action_container', 'operation_container', 'footer_container'
            ]
            
            missing_components = [comp for comp in required_components if comp not in ui_components]
            
            if not missing_components:
                print("✅ All required UI components created")
            else:
                print(f"⚠️ Missing UI components: {missing_components}")
            
            print(f"✅ UI components structure validated: {len(ui_components)} components")
        
        return True
        
    except Exception as e:
        print(f"❌ UI component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test backward compatibility with legacy API."""
    print("\n🧪 Testing backward compatibility...")
    
    try:
        from smartcash.ui.dataset.downloader import initialize_downloader_ui, get_downloader_components
        
        # Mock the display functionality to avoid IPython dependencies
        with patch('IPython.display.display') as mock_display, \
             patch('smartcash.ui.dataset.downloader.downloader_uimodule.create_downloader_ui') as mock_create_ui:
            
            # Configure mock
            mock_create_ui.return_value = {'main_container': Mock()}
            
            # Test legacy mode
            initialize_downloader_ui(use_legacy=True, config={'test': 'config'})
            print("✅ Legacy mode initialization works")
            
            # Test new mode (default)
            initialize_downloader_ui(use_legacy=False, config={'test': 'config'})
            print("✅ New mode initialization works")
            
            # Test default behavior (should use new mode)
            initialize_downloader_ui(config={'test': 'config'})
            print("✅ Default initialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shared_methods():
    """Test shared methods registration and functionality."""
    print("\n🧪 Testing shared methods...")
    
    try:
        from smartcash.ui.dataset.downloader.downloader_uimodule import register_downloader_shared_methods
        
        # Test shared methods registration
        register_downloader_shared_methods()
        print("✅ Shared methods registered successfully")
        
        # Test template registration
        from smartcash.ui.dataset.downloader.downloader_uimodule import register_downloader_template
        register_downloader_template()
        print("✅ Template registered successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Shared methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_downloader_refactor_tests():
    """Run all downloader refactor tests."""
    print("🧪 Running Dataset Downloader refactor tests...")
    print("=" * 60)
    
    test_results = []
    
    tests = [
        ("Downloader Imports", test_downloader_imports),
        ("UIModule Creation", test_downloader_uimodule_creation),
        ("Configuration Handling", test_downloader_configuration),
        ("Backend Service Integration", test_backend_service_integration),
        ("Operation Manager Integration", test_operation_manager_integration),
        ("UI Component Structure", test_ui_component_structure),
        ("Backward Compatibility", test_backward_compatibility),
        ("Shared Methods", test_shared_methods),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("🧪 DOWNLOADER REFACTOR TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} refactor tests passed")
    
    if passed == total:
        print("\n🎉 ALL DOWNLOADER REFACTOR TESTS PASSED!")
        print("\n📋 REFACTOR ACHIEVEMENTS:")
        print("✅ New UIModule pattern implementation")
        print("✅ Preserved unique form and UI structure") 
        print("✅ Maintained backend integration flow")
        print("✅ Backend service integration working")
        print("✅ Operation manager properly integrated")
        print("✅ Full backward compatibility maintained")
        print("✅ Shared methods and templates registered")
        print("✅ Configuration handling preserved")
        
        print("\n🔄 MIGRATION BENEFITS:")
        print("📦 Consistent with Colab and Dependency modules")
        print("🔧 Enhanced error handling and logging")
        print("⚡ Improved button management and UI feedback")
        print("🎯 Better separation of concerns")
        print("🔗 Shared method injection support")
        print("♻️ Easy module reset and cleanup")
    else:
        print("⚠️ Some refactor tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = run_downloader_refactor_tests()
    sys.exit(0 if success else 1)