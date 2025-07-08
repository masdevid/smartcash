"""
Simple test for augment module to verify basic functionality
"""

import sys
import os

# Add the smartcash directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

def test_augment_imports():
    """Test that augment module can be imported successfully."""
    try:
        # Test constants import
        from smartcash.ui.dataset.augment.constants import (
            AugmentationOperation, AugmentationTypes, UI_CONFIG, DEFAULT_AUGMENTATION_PARAMS
        )
        print("✅ Constants imported successfully")
        
        # Test config handler import
        from smartcash.ui.dataset.augment.configs.augment_config_handler import AugmentConfigHandler
        print("✅ Config handler imported successfully")
        
        # Test config handler functionality
        config_handler = AugmentConfigHandler()
        default_config = config_handler.get_default_config()
        print(f"✅ Config handler works - default config has {len(default_config)} sections")
        
        # Test configuration validation
        test_config = {
            'data': {'dir': 'test_data'},
            'augmentation': {
                'num_variations': 2,
                'target_count': 100,
                'intensity': 0.5,
                'balance_classes': True,
                'target_split': 'train',
                'types': ['combined']
            }
        }
        
        is_valid, errors = config_handler.validate_config(test_config)
        print(f"✅ Config validation works - valid: {is_valid}, errors: {len(errors)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_augment_components():
    """Test that augment components can be created."""
    try:
        from unittest.mock import patch, MagicMock
        
        # Mock ipywidgets
        with patch('smartcash.ui.dataset.augment.components.basic_options.widgets') as mock_widgets:
            mock_widgets.IntSlider.return_value = MagicMock()
            mock_widgets.FloatSlider.return_value = MagicMock()
            mock_widgets.Dropdown.return_value = MagicMock()
            mock_widgets.Checkbox.return_value = MagicMock()
            mock_widgets.Text.return_value = MagicMock()
            mock_widgets.VBox.return_value = MagicMock()
            mock_widgets.HTML.return_value = MagicMock()
            mock_widgets.Layout.return_value = MagicMock()
            
            from smartcash.ui.dataset.augment.components.basic_options import create_basic_options_widget
            
            result = create_basic_options_widget()
            print(f"✅ Basic options widget created with {len(result['widgets'])} widgets")
            
            return True
            
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_augment_operations():
    """Test that augment operations can be created."""
    try:
        from smartcash.ui.dataset.augment.operations.augment_operation import AugmentOperation
        from smartcash.ui.dataset.augment.operations.check_operation import CheckOperation
        
        # Test operation creation
        mock_ui_components = {
            'update_methods': {
                'progress': lambda p, ph: None,
                'activity': lambda msg: None,
                'operation_metrics': lambda t, p, s: None,
                'dataset_stats': lambda o, g, c: None
            }
        }
        
        operation = AugmentOperation(mock_ui_components)
        check_operation = CheckOperation(mock_ui_components)
        
        print("✅ Operations created successfully")
        
        # Test status
        status = operation.get_status()
        print(f"✅ Operation status works - progress: {status['progress']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all simple tests."""
    print("🧪 Running augment module simple tests...")
    
    tests = [
        ("Import Test", test_augment_imports),
        ("Components Test", test_augment_components), 
        ("Operations Test", test_augment_operations)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Augment module is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)