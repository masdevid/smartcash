#!/usr/bin/env python3
"""
Comprehensive Test Suite for Colab UI Module

This test suite validates all aspects of the colab UI module including:
- UI component creation and structure
- Button functionality and phase management
- Form widgets and configuration
- Integration with handlers and operations
- Configuration validation and edge cases
"""

import sys
import os
import traceback
from typing import Dict, Any, Optional
import ipywidgets as widgets

# Add project root to path
sys.path.insert(0, '/Users/masdevid/Projects/smartcash')

def test_ui_creation():
    """Test basic UI component creation and structure."""
    print("🧪 Testing UI Creation...")
    
    try:
        from smartcash.ui.setup.colab.components.colab_ui import create_colab_ui
        
        # Test with default config
        ui_components = create_colab_ui()
        
        # Check required keys are present
        required_keys = [
            'ui', 'header_container', 'form_container', 'action_container',
            'operation_container', 'footer_container', 'summary_container',
            'primary_button', 'form_widgets', 'module_name', 'parent_module',
            'ui_initialized', 'config', 'version'
        ]
        
        missing_keys = [key for key in required_keys if key not in ui_components]
        if missing_keys:
            print(f"❌ Missing required keys: {missing_keys}")
            return False
        
        # Check that main UI is a widget
        if not isinstance(ui_components['ui'], widgets.Widget):
            print(f"❌ Main UI is not a widget: {type(ui_components['ui'])}")
            return False
        
        # Check that containers are widgets
        container_keys = ['header_container', 'form_container', 'action_container', 
                         'operation_container', 'footer_container', 'summary_container']
        for key in container_keys:
            if key in ui_components and ui_components[key] is not None:
                if not isinstance(ui_components[key], widgets.Widget):
                    print(f"❌ {key} is not a widget: {type(ui_components[key])}")
                    return False
        
        print("✅ UI Creation Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ UI Creation Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def test_button_functionality():
    """Test primary button and its functionality."""
    print("🧪 Testing Button Functionality...")
    
    try:
        from smartcash.ui.setup.colab.components.colab_ui import create_colab_ui
        
        ui_components = create_colab_ui()
        
        # Check primary button exists
        if 'primary_button' not in ui_components or ui_components['primary_button'] is None:
            print("❌ Primary button not found")
            return False
        
        primary_button = ui_components['primary_button']
        
        # Check button properties
        if not hasattr(primary_button, 'description'):
            print("❌ Primary button missing description property")
            return False
        
        if '🚀' not in primary_button.description:
            print(f"❌ Primary button description incorrect: {primary_button.description}")
            return False
        
        # Check setup button alias exists
        if 'setup_button' not in ui_components:
            print("❌ Setup button alias not found")
            return False
        
        # Check save/reset buttons
        if 'save_button' not in ui_components or 'reset_button' not in ui_components:
            print("❌ Save/Reset buttons not found")
            return False
        
        print("✅ Button Functionality Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Button Functionality Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def test_form_widgets():
    """Test form widgets and configuration."""
    print("🧪 Testing Form Widgets...")
    
    try:
        from smartcash.ui.setup.colab.components.colab_ui import create_colab_ui
        
        # Test with custom config
        custom_config = {
            'auto_detect': False,
            'drive_path': '/custom/drive',
            'project_name': 'TestProject',
            'show_summary': True
        }
        
        ui_components = create_colab_ui(config=custom_config)
        
        # Check form widgets exist
        if 'form_widgets' not in ui_components:
            print("❌ Form widgets not found")
            return False
        
        form_widgets = ui_components['form_widgets']
        
        # Check required form widget keys
        required_widget_keys = ['ui', 'auto_detect', 'drive_path', 'project_name', 'env_info_panel']
        missing_widget_keys = [key for key in required_widget_keys if key not in form_widgets]
        if missing_widget_keys:
            print(f"❌ Missing form widget keys: {missing_widget_keys}")
            return False
        
        # Check widget values match config
        if form_widgets['auto_detect'].value != custom_config['auto_detect']:
            print(f"❌ Auto detect value mismatch: {form_widgets['auto_detect'].value} != {custom_config['auto_detect']}")
            return False
        
        if form_widgets['drive_path'].value != custom_config['drive_path']:
            print(f"❌ Drive path value mismatch: {form_widgets['drive_path'].value} != {custom_config['drive_path']}")
            return False
        
        if form_widgets['project_name'].value != custom_config['project_name']:
            print(f"❌ Project name value mismatch: {form_widgets['project_name'].value} != {custom_config['project_name']}")
            return False
        
        print("✅ Form Widgets Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Form Widgets Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def test_configuration_validation():
    """Test configuration validation and helper functions."""
    print("🧪 Testing Configuration Validation...")
    
    try:
        from smartcash.ui.setup.colab.components.colab_ui import (
            validate_colab_config, get_colab_default_config, update_colab_config
        )
        
        # Test default config
        default_config = get_colab_default_config()
        if not isinstance(default_config, dict):
            print("❌ Default config is not a dictionary")
            return False
        
        required_default_keys = ['auto_detect', 'drive_path', 'project_name', 'show_summary']
        missing_default_keys = [key for key in required_default_keys if key not in default_config]
        if missing_default_keys:
            print(f"❌ Missing default config keys: {missing_default_keys}")
            return False
        
        # Test validation with valid config
        valid_config = {
            'project_name': 'ValidProject',
            'environment_type': 'colab'
        }
        
        try:
            validated = validate_colab_config(valid_config)
            if not isinstance(validated, dict):
                print("❌ Validation did not return a dictionary")
                return False
        except Exception as e:
            print(f"❌ Valid config validation failed: {str(e)}")
            return False
        
        # Test validation with invalid config (empty project name)
        invalid_config = {
            'project_name': '',
            'environment_type': 'colab'
        }
        
        try:
            validate_colab_config(invalid_config)
            print("❌ Invalid config validation should have failed")
            return False
        except ValueError:
            # Expected behavior
            pass
        
        print("✅ Configuration Validation Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration Validation Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def test_integration():
    """Test full integration with cell execution."""
    print("🧪 Testing Full Integration...")
    
    try:
        # Test cell execution
        from smartcash.ui.setup.colab.colab_initializer import initialize_colab_ui
        
        # This should run without errors
        result = initialize_colab_ui()
        
        # The function might return None but should not raise exceptions
        print("✅ Integration Test Passed - Cell execution successful")
        return True
        
    except Exception as e:
        print(f"❌ Integration Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def test_template_compliance():
    """Test compliance with UI module template."""
    print("🧪 Testing Template Compliance...")
    
    try:
        # Run validation script
        import subprocess
        result = subprocess.run([
            'python', 'validate_ui_module.py', 
            'smartcash/ui/setup/colab/components/colab_ui.py'
        ], capture_output=True, text=True, cwd='/Users/masdevid/Projects/smartcash')
        
        if result.returncode != 0:
            print(f"❌ Template compliance validation failed")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            return False
        
        if "Score: 100.0%" not in result.stdout:
            print(f"❌ Template compliance score not 100%")
            print(f"Output: {result.stdout}")
            return False
        
        print("✅ Template Compliance Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Template Compliance Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def test_constants_and_imports():
    """Test constants and imports."""
    print("🧪 Testing Constants and Imports...")
    
    try:
        from smartcash.ui.setup.colab.constants import UI_CONFIG, BUTTON_CONFIG, VALIDATION_RULES
        
        # Check UI_CONFIG structure
        required_ui_keys = ['title', 'subtitle', 'icon', 'module_name', 'parent_module', 'version']
        missing_ui_keys = [key for key in required_ui_keys if key not in UI_CONFIG]
        if missing_ui_keys:
            print(f"❌ Missing UI_CONFIG keys: {missing_ui_keys}")
            return False
        
        # Check BUTTON_CONFIG structure
        if 'primary' not in BUTTON_CONFIG:
            print("❌ Primary button config not found")
            return False
        
        primary_config = BUTTON_CONFIG['primary']
        required_button_keys = ['text', 'style', 'tooltip', 'order']
        missing_button_keys = [key for key in required_button_keys if key not in primary_config]
        if missing_button_keys:
            print(f"❌ Missing primary button config keys: {missing_button_keys}")
            return False
        
        if primary_config['style'] != 'primary':
            print(f"❌ Primary button style incorrect: {primary_config['style']}")
            return False
        
        # Check VALIDATION_RULES
        if not isinstance(VALIDATION_RULES, dict):
            print("❌ VALIDATION_RULES is not a dictionary")
            return False
        
        print("✅ Constants and Imports Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Constants and Imports Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def run_comprehensive_test():
    """Run all comprehensive tests."""
    print("🚀 Starting Comprehensive Colab UI Test Suite")
    print("=" * 60)
    
    tests = [
        ("Constants and Imports", test_constants_and_imports),
        ("UI Creation", test_ui_creation),
        ("Button Functionality", test_button_functionality),
        ("Form Widgets", test_form_widgets),
        ("Configuration Validation", test_configuration_validation),
        ("Template Compliance", test_template_compliance),
        ("Full Integration", test_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} Test: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} Test: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} Test: FAILED with exception: {str(e)}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Colab UI module is fully functional.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)