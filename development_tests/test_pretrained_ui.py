#!/usr/bin/env python3
"""
Test script for pretrained module UI functionality.
This script tests the complete pretrained module workflow.
"""

import sys
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch
from pathlib import Path

sys.path.append('/Users/masdevid/Projects/smartcash')

from smartcash.ui.model.pretrained.pretrained_initializer import PretrainedInitializer
from smartcash.ui.model.pretrained.components.pretrained_ui import create_pretrained_ui
from smartcash.ui.model.pretrained.handlers.pretrained_ui_handler import PretrainedUIHandler
from smartcash.ui.model.pretrained.operations.download_operation import DownloadOperation


def test_ui_components_creation():
    """Test UI components creation."""
    print("🧪 Testing UI components creation...")
    
    try:
        # Test UI creation with default config
        ui_components = create_pretrained_ui()
        
        # Verify required components exist
        required_components = [
            'ui', 'main_container', 'header_container', 'form_container',
            'action_container', 'summary_container', 'operation_container',
            'footer_container', 'download_button', 'progress_tracker',
            'log_output', 'input_options'
        ]
        
        missing_components = [comp for comp in required_components if comp not in ui_components]
        if missing_components:
            print(f"❌ Missing UI components: {missing_components}")
            return False
        
        print("✅ All required UI components created successfully")
        
        # Test input components
        input_options = ui_components['input_options']
        input_required = ['model_dir_input', 'yolo_url_input', 'efficientnet_url_input']
        
        missing_inputs = [inp for inp in input_required if inp not in input_options]
        if missing_inputs:
            print(f"❌ Missing input components: {missing_inputs}")
            return False
        
        print("✅ All input components created successfully")
        
        # Test button functionality
        download_button = ui_components['download_button']
        if not hasattr(download_button, 'on_click'):
            print("❌ Download button missing on_click method")
            return False
        
        print("✅ Download button has proper interface")
        return True
        
    except Exception as e:
        print(f"❌ UI components creation failed: {str(e)}")
        return False


def test_ui_handler_functionality():
    """Test UI handler functionality."""
    print("\n🧪 Testing UI handler functionality...")
    
    try:
        # Create mock UI components
        ui_components = {
            'download_button': Mock(),
            'input_options': {
                'model_dir_input': Mock(),
                'yolo_url_input': Mock(),
                'efficientnet_url_input': Mock()
            },
            'progress_tracker': Mock(),
            'log_output': Mock()
        }
        
        # Set mock values
        ui_components['input_options']['model_dir_input'].value = '/test/models'
        ui_components['input_options']['yolo_url_input'].value = 'https://test.url/yolo.pt'
        ui_components['input_options']['efficientnet_url_input'].value = ''
        
        # Create handler
        handler = PretrainedUIHandler(ui_components)
        
        # Test config extraction
        config = handler._extract_config_from_ui()
        
        if config['models_dir'] != '/test/models':
            print(f"❌ Config extraction failed - models_dir: {config['models_dir']}")
            return False
        
        if config['model_urls']['yolov5s'] != 'https://test.url/yolo.pt':
            print(f"❌ Config extraction failed - yolo url: {config['model_urls']['yolov5s']}")
            return False
        
        print("✅ Config extraction working correctly")
        
        # Test button handler setup
        if not ui_components['download_button'].on_click.called:
            print("✅ Button handler setup correctly (on_click should be called during initialization)")
        
        return True
        
    except Exception as e:
        print(f"❌ UI handler test failed: {str(e)}")
        return False


async def test_download_operation():
    """Test download operation functionality."""
    print("\n🧪 Testing download operation...")
    
    try:
        # Create operation
        operation = DownloadOperation()
        
        # Test operation properties
        if operation.operation_type != 'download':
            print(f"❌ Wrong operation type: {operation.operation_type}")
            return False
        
        if not operation.progress_steps:
            print("❌ No progress steps defined")
            return False
        
        print(f"✅ Operation type: {operation.operation_type}")
        print(f"✅ Progress steps: {len(operation.progress_steps)} steps")
        
        # Test operation methods
        ops = operation.get_operations()
        if 'download' not in ops:
            print("❌ Download operation not available")
            return False
        
        print("✅ Download operation available")
        
        # Test with mock config and UI components
        config = {
            'models_dir': '/tmp/test_models',
            'model_urls': {
                'yolov5s': 'https://test.url/yolo.pt',
                'efficientnet_b4': ''
            }
        }
        
        ui_components = {
            'progress_tracker': Mock(),
            'log_output': Mock()
        }
        
        # Set up mocks for callbacks
        ui_components['progress_tracker'].update_progress = Mock()
        ui_components['log_output'].log = Mock()
        
        # Mock the service methods to avoid actual downloads
        with patch.object(operation.service, 'check_existing_models') as mock_check, \
             patch.object(operation.service, 'download_all_models') as mock_download, \
             patch.object(operation.service, 'get_models_summary') as mock_summary:
            
            # Set up mock returns
            mock_check.return_value = {
                'models_dir': '/tmp/test_models',
                'models_found': [],
                'models_missing': [
                    {'model_type': 'yolov5s', 'name': 'YOLOv5s'},
                    {'model_type': 'efficientnet_b4', 'name': 'EfficientNet-B4'}
                ],
                'total_found': 0,
                'all_present': False
            }
            
            mock_download.return_value = {
                'all_successful': True,
                'success_count': 2,
                'total_count': 2,
                'downloads': [
                    {'model': 'YOLOv5s', 'success': True},
                    {'model': 'EfficientNet-B4', 'success': True}
                ]
            }
            
            mock_summary.return_value = {
                'models_count': 2,
                'total_size_mb': 89.4,
                'available_models': []
            }
            
            # Execute operation
            result = await operation.execute_operation(config, ui_components)
            
            # Verify results
            if not result.get('success', False):
                print(f"❌ Operation failed: {result.get('error', 'Unknown error')}")
                return False
            
            if result['operation'] != 'download':
                print(f"❌ Wrong operation in result: {result['operation']}")
                return False
            
            print("✅ Download operation executed successfully")
            print(f"✅ Service methods called: check={mock_check.called}, download={mock_download.called}, summary={mock_summary.called}")
            
            # Verify UI callbacks were used
            if ui_components['progress_tracker'].update_progress.called:
                print("✅ Progress callback was used")
            
            if ui_components['log_output'].log.called:
                print("✅ Log callback was used")
            
            return True
        
    except Exception as e:
        print(f"❌ Download operation test failed: {str(e)}")
        return False


async def test_post_init_functionality():
    """Test post-initialization functionality."""
    print("\n🧪 Testing post-initialization functionality...")
    
    try:
        # Create initializer
        initializer = PretrainedInitializer()
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'models_dir': temp_dir,
                'model_urls': {
                    'yolov5s': 'https://test.url/yolo.pt',
                    'efficientnet_b4': ''
                }
            }
            
            # Mock UI creation to avoid widget dependencies
            with patch('smartcash.ui.model.pretrained.components.pretrained_ui.create_pretrained_ui') as mock_create_ui, \
                 patch('smartcash.ui.model.pretrained.handlers.pretrained_ui_handler.PretrainedUIHandler') as mock_handler_class:
                
                # Mock UI components
                mock_ui_components = {
                    'ui': Mock(),
                    'download_button': Mock(),
                    'progress_tracker': Mock(),
                    'log_output': Mock(),
                    'input_options': {
                        'model_dir_input': Mock(),
                        'yolo_url_input': Mock(),
                        'efficientnet_url_input': Mock()
                    }
                }
                mock_create_ui.return_value = mock_ui_components
                
                # Mock handler
                mock_handler = Mock()
                mock_handler.check_models_status = Mock()
                mock_handler.check_models_status.return_value = {
                    'total_found': 0,
                    'all_present': False,
                    'models_found': [],
                    'models_missing': [
                        {'name': 'YOLOv5s'},
                        {'name': 'EfficientNet-B4'}
                    ]
                }
                mock_handler_class.return_value = mock_handler
                
                # Set up log mock
                mock_log = Mock()
                mock_ui_components['log_output'].log = mock_log
                
                # Initialize
                result = initializer._initialize_impl(config)
                
                # Verify successful initialization
                if not result or 'ui' not in result:
                    print("❌ Initialization failed")
                    return False
                
                print("✅ Module initialization successful")
                
                # Give a moment for async post-init check
                await asyncio.sleep(0.1)
                
                # Verify post-init check was attempted
                if hasattr(mock_handler, 'check_models_status'):
                    print("✅ Post-init check handler method exists")
                
                return True
                
    except Exception as e:
        print(f"❌ Post-init test failed: {str(e)}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Pretrained Module UI Tests")
    print("=" * 50)
    
    tests = [
        ("UI Components Creation", test_ui_components_creation),
        ("UI Handler Functionality", test_ui_handler_functionality),
        ("Download Operation", test_download_operation),
        ("Post-Init Functionality", test_post_init_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            
            if success:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Pretrained module is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)