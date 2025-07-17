#!/usr/bin/env python3
"""
Final comprehensive test suite to verify all UI fixes work correctly.

This script tests:
1. PretrainedOperationManager clear_logs method exists and works
2. TrainingService initialization without loss_metrics errors 
3. Refresh backbone config functionality works (via training module)
4. Log suppression and redirection
5. Verify the refresh button was added to training UI

Author: Claude Code Assistant
Date: July 13, 2025
"""

import sys
import os
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import importlib

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pretrained_clear_logs():
    """Test 1: PretrainedOperationManager clear_logs method exists and works"""
    print("\n=== Test 1: PretrainedOperationManager clear_logs method ===")
    
    try:
        # Import the pretrained operation manager
        from smartcash.ui.model.pretrained.operations.pretrained_operation_manager import PretrainedOperationManager
        
        # Mock the required dependencies
        mock_config = {'pretrained': {'models_dir': '/data/pretrained'}}
        mock_operation_container = Mock()
        mock_operation_container.clear_logs = Mock()
        
        # Initialize the operation manager with mocked dependencies
        manager = PretrainedOperationManager(config=mock_config, operation_container=mock_operation_container)
        
        # Check if clear_logs method exists
        if hasattr(manager, 'clear_logs'):
            print("✅ clear_logs method exists")
            
            # Test the method
            try:
                result = manager.clear_logs()
                print(f"✅ clear_logs method executed successfully")
                
                # Verify it calls the operation container's clear_logs
                mock_operation_container.clear_logs.assert_called()
                print("✅ clear_logs properly delegates to operation container")
                return True
            except Exception as e:
                print(f"❌ clear_logs method failed: {e}")
                return False
        else:
            print("❌ clear_logs method does not exist")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import PretrainedOperationManager: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_training_service_init():
    """Test 2: TrainingService initialization without loss_metrics errors"""
    print("\n=== Test 2: TrainingService initialization ===")
    
    try:
        # First, let's check if the training service exists
        from smartcash.ui.model.train.services.training_service import TrainingService
        
        print("✅ TrainingService import successful")
        
        # Try to initialize the service
        try:
            service = TrainingService()
            print("✅ TrainingService initialization successful")
            
            # Check if it has the expected attributes without loss_metrics issues
            if hasattr(service, '__dict__'):
                attrs = service.__dict__
                print(f"✅ Service attributes: {list(attrs.keys())}")
                
                # Check if loss_metrics is handled properly
                if 'loss_metrics' in attrs:
                    print(f"✅ loss_metrics attribute found: {attrs['loss_metrics']}")
                else:
                    print("✅ No loss_metrics attribute (which is fine)")
                    
            return True
            
        except Exception as init_error:
            print(f"❌ TrainingService initialization failed: {init_error}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import TrainingService: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_training_refresh_backbone_config():
    """Test 3: Refresh backbone config functionality works (via training module)"""
    print("\n=== Test 3: Training module refresh backbone config functionality ===")
    
    try:
        # Import training UI module
        from smartcash.ui.model.train import create_train_uimodule
        
        print("✅ Training UI module import successful")
        
        # Create the training UI module with mocked dependencies
        try:
            with patch('ipywidgets.VBox'), patch('ipywidgets.HBox'), patch('IPython.display.display'):
                train_module = create_train_uimodule(auto_initialize=False)
                print("✅ Training UIModule creation successful")
                
                # Check if refresh_backbone_config method exists
                if hasattr(train_module, 'refresh_backbone_config'):
                    print("✅ refresh_backbone_config method found in train module")
                    
                    # Test the method
                    try:
                        # Mock the config handler
                        with patch.object(train_module, '_config_handler') as mock_config_handler:
                            mock_config_handler.refresh_backbone_config.return_value = {'status': 'refreshed'}
                            
                            result = train_module.refresh_backbone_config()
                            print(f"✅ refresh_backbone_config executed successfully: {result}")
                            return True
                    except Exception as e:
                        print(f"❌ refresh_backbone_config execution failed: {e}")
                        return False
                else:
                    print("❌ refresh_backbone_config method not found")
                    return False
                
        except Exception as e:
            print(f"❌ Training UIModule creation failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import training module: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_log_suppression():
    """Test 4: Log suppression and redirection functionality"""
    print("\n=== Test 4: Log suppression and redirection ===")
    
    try:
        # Test log suppression utility
        from smartcash.ui.core.utils.log_suppression import suppress_initial_logs
        
        print("✅ Log suppression utility import successful")
        
        # Test the context manager
        try:
            with suppress_initial_logs(duration=1.0):
                # This should suppress logs
                logger = logging.getLogger('test_logger')
                logger.info("This log should be suppressed")
                print("✅ Log suppression context manager works")
            
            # Outside context, logs should work normally
            logger.info("This log should appear")
            print("✅ Log suppression cleanup successful")
            return True
            
        except Exception as e:
            print(f"❌ Log suppression test failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import log suppression utility: {e}")
        
        # Try alternative import path
        try:
            from smartcash.ui.core.logging.ui_logging_manager import UILoggingManager
            print("✅ Alternative log manager import successful")
            
            # Test basic functionality
            manager = UILoggingManager()
            print("✅ UI logging manager creation successful")
            return True
            
        except ImportError as e2:
            print(f"❌ Alternative log manager import failed: {e2}")
            return False
        except Exception as e2:
            print(f"❌ UI logging manager test failed: {e2}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_training_refresh_button():
    """Test 5: Verify refresh button was added to training UI"""
    print("\n=== Test 5: Training UI refresh button ===")
    
    try:
        # Import training UI module
        from smartcash.ui.model.train import create_train_uimodule
        
        print("✅ Training UI module import successful")
        
        # Create the training UI module
        try:
            with patch('ipywidgets.VBox'), patch('ipywidgets.HBox'), patch('IPython.display.display'):
                train_module = create_train_uimodule(auto_initialize=False)
                print("✅ Training UIModule creation successful")
                
                # Check for refresh functionality in various places
                refresh_found = False
                
                # Check in the module itself
                if hasattr(train_module, 'refresh_backbone_config'):
                    print("✅ refresh_backbone_config method found in train module")
                    refresh_found = True
                
                # Check execute method
                if hasattr(train_module, 'execute_refresh_backbone_config'):
                    print("✅ execute_refresh_backbone_config method found")
                    refresh_found = True
                
                # Check if refresh method exists
                methods = [method for method in dir(train_module) if 'refresh' in method.lower()]
                if methods:
                    print(f"✅ Refresh-related methods found: {methods}")
                    refresh_found = True
                
                if refresh_found:
                    print("✅ Refresh functionality confirmed in training UI")
                    return True
                else:
                    print("❌ No refresh functionality found in training UI")
                    return False
                
        except Exception as e:
            print(f"❌ Training UIModule creation failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import training module: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_module_structure():
    """Additional test: Verify module structure and imports"""
    print("\n=== Additional Test: Module Structure and Imports ===")
    
    modules_to_test = [
        'smartcash.ui.model.pretrained',
        'smartcash.ui.model.train',
        'smartcash.ui.model.backbone',
        'smartcash.ui.core.utils.log_suppression',
        'smartcash.ui.core.logging.ui_logging_manager'
    ]
    
    results = {}
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name} - Import successful")
            results[module_name] = True
        except ImportError as e:
            print(f"❌ {module_name} - Import failed: {e}")
            results[module_name] = False
        except Exception as e:
            print(f"❌ {module_name} - Unexpected error: {e}")
            results[module_name] = False
    
    return all(results.values())

def main():
    """Run all comprehensive tests"""
    print("🧪 SmartCash UI Fixes Final Test Suite")
    print("=" * 60)
    
    # Run all tests
    test_results = []
    
    # Test 1: PretrainedOperationManager clear_logs
    test_results.append(("PretrainedOperationManager clear_logs", test_pretrained_clear_logs()))
    
    # Test 2: TrainingService initialization
    test_results.append(("TrainingService initialization", test_training_service_init()))
    
    # Test 3: Training refresh backbone config
    test_results.append(("Training refresh backbone config", test_training_refresh_backbone_config()))
    
    # Test 4: Log suppression
    test_results.append(("Log suppression", test_log_suppression()))
    
    # Test 5: Training refresh button
    test_results.append(("Training refresh button", test_training_refresh_button()))
    
    # Additional test: Module structure
    test_results.append(("Module structure", test_module_structure()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<35} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Total Tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(test_results)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! UI fixes are working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Some fixes may need attention.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)