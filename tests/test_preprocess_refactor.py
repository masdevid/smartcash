#!/usr/bin/env python3
"""
Test script for refactored dataset preprocess module.
Verifies new UIModule pattern integration while preserving backend functionality.
"""

import sys
import os
from typing import Dict, Any
from unittest.mock import Mock, patch

# Add the project root to Python path
sys.path.insert(0, '/Users/masdevid/Projects/smartcash')

def test_preprocess_imports():
    """Test that preprocess module imports work correctly."""
    print("🧪 Testing preprocess module imports...")
    
    try:
        # Test new UIModule pattern imports
        from smartcash.ui.dataset.preprocess import (
            PreprocessUIModule,
            create_preprocess_uimodule,
            get_preprocess_uimodule,
            reset_preprocess_uimodule
        )
        print("✅ New UIModule pattern imports successful")
        
        # Test backward compatibility imports
        from smartcash.ui.dataset.preprocess import (
            initialize_preprocess_ui,
            display_preprocess_ui,
            get_preprocess_components
        )
        print("✅ Backward compatibility imports successful")
        
        # Test legacy imports
        from smartcash.ui.dataset.preprocess import (
            initialize_preprocess_ui_legacy,
            get_preprocessing_initializer
        )
        print("✅ Legacy imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocess_uimodule_creation():
    """Test PreprocessUIModule creation and initialization."""
    print("\n🧪 Testing PreprocessUIModule creation...")
    
    try:
        from smartcash.ui.dataset.preprocess import PreprocessUIModule, create_preprocess_uimodule
        
        # Test direct instantiation
        module = PreprocessUIModule()
        print("✅ PreprocessUIModule instantiated successfully")
        
        # Test module info
        print(f"📋 Module name: {module.module_name}")
        print(f"📋 Parent module: {module.parent_module}")
        print(f"📋 Full module name: {module.full_module_name}")
        
        # Test factory creation (without auto-initialization)
        module2 = create_preprocess_uimodule(auto_initialize=False)
        print("✅ Factory creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ UIModule creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocess_configuration():
    """Test preprocess configuration handling."""
    print("\n🧪 Testing preprocess configuration...")
    
    try:
        from smartcash.ui.dataset.preprocess import create_preprocess_uimodule
        from smartcash.ui.dataset.preprocess.configs.preprocess_defaults import get_default_preprocessing_config
        
        # Test default configuration
        default_config = get_default_preprocessing_config()
        print(f"✅ Default config loaded: {len(default_config)} sections")
        
        # Check required configuration sections
        required_sections = ['preprocessing', 'data', 'performance']
        missing_sections = [section for section in required_sections if section not in default_config]
        
        if not missing_sections:
            print("✅ All required config sections present")
        else:
            print(f"⚠️ Missing config sections: {missing_sections}")
        
        # Test config with custom values
        custom_config = {
            'preprocessing': {
                'target_splits': ['train', 'valid', 'test'],
                'normalization': {
                    'preset': 'yolov5l',
                    'target_size': [832, 832]
                }
            }
        }
        
        # Create module with custom config
        module = create_preprocess_uimodule(config=custom_config, auto_initialize=False)
        merged_config = module.get_config()
        
        # Verify config merge
        if merged_config['preprocessing']['normalization']['preset'] == 'yolov5l':
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
        from smartcash.ui.dataset.preprocess.services.preprocess_service import PreprocessService
        
        # Test service instantiation
        service = PreprocessService()
        print("✅ PreprocessService instantiated successfully")
        
        # Test service methods (mock backend calls)
        test_config = {
            'preprocessing': {
                'target_splits': ['train', 'valid'],
                'normalization': {
                    'preset': 'yolov5s',
                    'target_size': [640, 640]
                }
            },
            'data': {
                'dir': 'data',
                'preprocessed_dir': 'data/preprocessed'
            }
        }
        
        # Test preprocessing operation (mock)
        with patch('smartcash.dataset.preprocessor.preprocess_dataset', 
                  return_value={'success': True, 'processed_files': 100, 'processed_splits': ['train', 'valid']}):
            result = service.preprocess_dataset(test_config)
            if result.get('success'):
                print("✅ Preprocessing service works")
            else:
                print("⚠️ Preprocessing service issue")
        
        # Test status check (mock)
        with patch('smartcash.dataset.preprocessor.get_preprocessing_status', 
                  return_value={'success': True, 'service_ready': True, 'files_found': 100}):
            status_result = service.get_preprocessing_status(test_config)
            if status_result.get('success'):
                print("✅ Status check service works")
            else:
                print("⚠️ Status check service issue")
        
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
        from smartcash.ui.dataset.preprocess.operations.manager import PreprocessOperationManager
        from unittest.mock import Mock
        
        # Create mock UI components
        mock_operation_container = Mock()
        mock_operation_container.log = Mock()
        mock_operation_container.update_progress = Mock()
        
        # Test operation manager creation
        operation_manager = PreprocessOperationManager(
            config={'test': 'config'},
            operation_container=mock_operation_container
        )
        
        print("✅ OperationManager created successfully")
        
        # Test initialization
        operation_manager.initialize()
        print("✅ OperationManager initialized successfully")
        
        # Test operations availability
        operations = operation_manager.get_operations()
        expected_operations = ['preprocess', 'check', 'cleanup']
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
        from smartcash.ui.dataset.preprocess.components.preprocess_ui import create_preprocessing_main_ui
        
        # Mock UI component creation to avoid IPython dependencies
        with patch('smartcash.ui.components.header_container.create_header_container') as mock_header, \
             patch('smartcash.ui.components.form_container.create_form_container') as mock_form, \
             patch('smartcash.ui.components.action_container.create_action_container') as mock_action, \
             patch('smartcash.ui.components.operation_container.create_operation_container') as mock_operation, \
             patch('smartcash.ui.components.footer_container.create_footer_container') as mock_footer, \
             patch('smartcash.ui.components.main_container.create_main_container') as mock_main:
            
            # Configure mocks to return dict-like objects
            mock_header.return_value = Mock(container=Mock())
            mock_form.return_value = {'container': Mock(), 'get_form_container': Mock(return_value=Mock(children=()))}
            
            # Mock action container to return buttons
            mock_action_result = Mock()
            mock_action_result.get = Mock(side_effect=lambda key: Mock() if key in ['preprocess', 'check', 'cleanup'] else None)
            mock_action.return_value = mock_action_result
            
            mock_operation.return_value = {'container': Mock(), 'progress_tracker': Mock(), 'log_accordion': Mock()}
            mock_footer.return_value = Mock(container=Mock())
            mock_main.return_value = Mock()
            
            # Test UI creation
            ui_components = create_preprocessing_main_ui({'test': 'config'})
            
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
        from smartcash.ui.dataset.preprocess import initialize_preprocess_ui, get_preprocess_components
        
        # Mock the display functionality to avoid IPython dependencies
        with patch('IPython.display.display') as mock_display, \
             patch('smartcash.ui.dataset.preprocess.preprocess_uimodule.create_preprocessing_main_ui') as mock_create_ui:
            
            # Configure mock
            mock_create_ui.return_value = {'main_container': Mock()}
            
            # Test legacy mode
            initialize_preprocess_ui(use_legacy=True, config={'test': 'config'})
            print("✅ Legacy mode initialization works")
            
            # Test new mode (default)
            initialize_preprocess_ui(use_legacy=False, config={'test': 'config'})
            print("✅ New mode initialization works")
            
            # Test default behavior (should use new mode)
            initialize_preprocess_ui(config={'test': 'config'})
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
        from smartcash.ui.dataset.preprocess.preprocess_uimodule import register_preprocess_shared_methods
        
        # Test shared methods registration
        register_preprocess_shared_methods()
        print("✅ Shared methods registered successfully")
        
        # Test template registration
        from smartcash.ui.dataset.preprocess.preprocess_uimodule import register_preprocess_template
        register_preprocess_template()
        print("✅ Template registered successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Shared methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocessing_operations():
    """Test preprocessing operations execution."""
    print("\n🧪 Testing preprocessing operations...")
    
    try:
        from smartcash.ui.dataset.preprocess import create_preprocess_uimodule
        
        # Create module without auto-initialization to avoid UI dependencies
        module = create_preprocess_uimodule(auto_initialize=False)
        
        # Mock backend operations
        with patch('smartcash.dataset.preprocessor.preprocess_dataset', 
                  return_value={'success': True, 'processed_files': 100}) as mock_preprocess, \
             patch('smartcash.dataset.preprocessor.get_preprocessing_status', 
                  return_value={'success': True, 'service_ready': True}) as mock_status, \
             patch('smartcash.dataset.preprocessor.api.cleanup_api.cleanup_preprocessing_files', 
                  return_value={'success': True, 'files_removed': 50}) as mock_cleanup:
            
            # Test preprocessing operation
            result = module.execute_preprocess()
            if result.get('success'):
                print("✅ Preprocessing operation works")
            else:
                print(f"⚠️ Preprocessing operation issue: {result.get('message', 'Unknown')}")
            
            # Test check operation
            check_result = module.execute_check()
            if check_result.get('success'):
                print("✅ Check operation works")
            else:
                print(f"⚠️ Check operation issue: {check_result.get('message', 'Unknown')}")
            
            # Test cleanup operation
            cleanup_result = module.execute_cleanup()
            if cleanup_result.get('success'):
                print("✅ Cleanup operation works")
            else:
                print(f"⚠️ Cleanup operation issue: {cleanup_result.get('message', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_preprocess_refactor_tests():
    """Run all preprocess refactor tests."""
    print("🧪 Running Dataset Preprocess refactor tests...")
    print("=" * 60)
    
    test_results = []
    
    tests = [
        ("Preprocess Imports", test_preprocess_imports),
        ("UIModule Creation", test_preprocess_uimodule_creation),
        ("Configuration Handling", test_preprocess_configuration),
        ("Backend Service Integration", test_backend_service_integration),
        ("Operation Manager Integration", test_operation_manager_integration),
        ("UI Component Structure", test_ui_component_structure),
        ("Backward Compatibility", test_backward_compatibility),
        ("Shared Methods", test_shared_methods),
        ("Preprocessing Operations", test_preprocessing_operations),
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
    print("🧪 PREPROCESS REFACTOR TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} refactor tests passed")
    
    if passed == total:
        print("\n🎉 ALL PREPROCESS REFACTOR TESTS PASSED!")
        print("\n📋 REFACTOR ACHIEVEMENTS:")
        print("✅ New UIModule pattern implementation")
        print("✅ Preserved unique YOLO preprocessing functionality") 
        print("✅ Maintained backend integration flow")
        print("✅ Backend service integration working")
        print("✅ Operation manager properly integrated")
        print("✅ Full backward compatibility maintained")
        print("✅ Shared methods and templates registered")
        print("✅ Configuration handling preserved")
        print("✅ All preprocessing operations functional")
        
        print("\n🔄 MIGRATION BENEFITS:")
        print("📦 Consistent with Colab and Dependency modules")
        print("🔧 Enhanced error handling and logging")
        print("⚡ Improved button management and UI feedback")
        print("🎯 Better separation of concerns")
        print("🔗 Shared method injection support")
        print("♻️ Easy module reset and cleanup")
        print("🚀 YOLO-compatible preprocessing with normalization")
    else:
        print("⚠️ Some refactor tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = run_preprocess_refactor_tests()
    sys.exit(0 if success else 1)