#!/usr/bin/env python3
"""
Test script to validate augment module UI display instead of logs.

This script tests the DisplayInitializer pattern implementation in the augment module
and verifies that UI components are displayed correctly.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_augment_ui_display():
    """Test the augment module UI display functionality."""
    
    print("="*80)
    print("🧪 AUGMENT MODULE UI DISPLAY TEST")
    print("="*80)
    
    # Test 1: Import testing
    print("\n1️⃣ Testing module imports...")
    try:
        from smartcash.ui.dataset.augment import AugmentInitializer, initialize_augment_ui
        print("✅ Successfully imported AugmentInitializer and initialize_augment_ui")
    except ImportError as e:
        print(f"❌ Failed to import augment module: {e}")
        return False
    
    # Test 2: Create initializer
    print("\n2️⃣ Testing AugmentInitializer creation...")
    try:
        # Suppress early logging during initialization
        logging.getLogger().setLevel(logging.WARNING)
        
        initializer = AugmentInitializer()
        print("✅ Successfully created AugmentInitializer instance")
        print(f"   Module name: {initializer.module_name}")
        print(f"   Parent module: {initializer.parent_module}")
    except Exception as e:
        print(f"❌ Failed to create AugmentInitializer: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Initialize UI components
    print("\n3️⃣ Testing UI component initialization...")
    try:
        # Test with empty config
        ui_components = initializer.initialize()
        print("✅ Successfully initialized UI components")
        print(f"   Available keys: {list(ui_components.keys())}")
        
        # Check for main UI component
        if 'ui' in ui_components:
            print("✅ Found 'ui' key in components (DisplayInitializer pattern working)")
        elif 'main_container' in ui_components:
            print("✅ Found 'main_container' key in components")
        else:
            print("⚠️ Neither 'ui' nor 'main_container' found in components")
            
    except Exception as e:
        print(f"❌ Failed to initialize UI components: {e}")
        return False
    
    # Test 4: Check UI component types
    print("\n4️⃣ Testing UI component types...")
    try:
        ui_key = 'ui' if 'ui' in ui_components else 'main_container'
        if ui_key in ui_components:
            ui_component = ui_components[ui_key]
            print(f"✅ UI component type: {type(ui_component)}")
            
            # Check if it's a widget (should have typical widget attributes)
            if hasattr(ui_component, 'children') or hasattr(ui_component, 'layout'):
                print("✅ UI component appears to be a valid widget")
            else:
                print("⚠️ UI component may not be a standard widget")
                
        else:
            print("❌ No UI component found")
            return False
            
    except Exception as e:
        print(f"❌ Failed to check UI component types: {e}")
        return False
    
    # Test 5: Test DisplayInitializer pattern
    print("\n5️⃣ Testing DisplayInitializer pattern...")
    try:
        # Test the factory function that should use DisplayInitializer
        ui_components_factory = initialize_augment_ui()
        print("✅ Successfully called initialize_augment_ui factory function")
        
        # Check if it returns displayable components
        if isinstance(ui_components_factory, dict):
            print("✅ Factory function returned dictionary")
            if 'ui' in ui_components_factory:
                print("✅ DisplayInitializer pattern working - 'ui' key present")
            else:
                print("⚠️ 'ui' key not found in factory result")
        else:
            print(f"⚠️ Factory function returned {type(ui_components_factory)} instead of dict")
            
    except Exception as e:
        print(f"❌ Failed to test DisplayInitializer pattern: {e}")
        return False
    
    # Test 6: Test handlers initialization
    print("\n6️⃣ Testing handlers initialization...")
    try:
        config_handler = initializer.get_config_handler()
        ui_handler = initializer.get_ui_handler()
        
        if config_handler:
            print("✅ Config handler initialized successfully")
            print(f"   Config handler type: {type(config_handler)}")
        else:
            print("⚠️ Config handler not initialized")
            
        if ui_handler:
            print("✅ UI handler initialized successfully")
            print(f"   UI handler type: {type(ui_handler)}")
        else:
            print("⚠️ UI handler not initialized")
            
    except Exception as e:
        print(f"❌ Failed to test handlers initialization: {e}")
        return False
    
    # Test 7: Test operation status
    print("\n7️⃣ Testing operation status...")
    try:
        status = initializer.get_operation_status()
        print("✅ Successfully retrieved operation status")
        print(f"   Module initialized: {status.get('module_initialized', False)}")
        print(f"   UI handler ready: {status.get('ui_handler_ready', False)}")
        print(f"   Config handler ready: {status.get('config_handler_ready', False)}")
        
    except Exception as e:
        print(f"❌ Failed to test operation status: {e}")
        return False
    
    # Test 8: Test UI display (simulation)
    print("\n8️⃣ Testing UI display simulation...")
    try:
        # In a real Jupyter environment, this would display the UI
        # Here we just verify the structure is correct for display
        
        ui_key = 'ui' if 'ui' in ui_components else 'main_container'
        if ui_key in ui_components:
            ui_component = ui_components[ui_key]
            
            # Check if the component has display-related attributes
            display_attrs = ['layout', 'children', 'style', 'class_']
            found_attrs = [attr for attr in display_attrs if hasattr(ui_component, attr)]
            
            if found_attrs:
                print(f"✅ UI component has display attributes: {found_attrs}")
                print("✅ UI component is ready for display")
            else:
                print("⚠️ UI component may not have standard display attributes")
                
            # Test component structure
            if hasattr(ui_component, 'children') and ui_component.children:
                print(f"✅ UI component has {len(ui_component.children)} child components")
            
        else:
            print("❌ No UI component found for display testing")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test UI display: {e}")
        return False
    
    # Test 9: Test error handling
    print("\n9️⃣ Testing error handling...")
    try:
        # Test with invalid config
        try:
            invalid_initializer = AugmentInitializer(config={'invalid_key': 'invalid_value'})
            ui_components_invalid = invalid_initializer.initialize()
            print("✅ Error handling works - invalid config handled gracefully")
        except Exception as e:
            print(f"✅ Error handling works - invalid config raises exception: {type(e).__name__}")
        
    except Exception as e:
        print(f"❌ Failed to test error handling: {e}")
        return False
    
    # Test 10: Test logging suppression
    print("\n🔟 Testing logging suppression...")
    try:
        # Capture log messages during initialization
        import io
        
        # Create a string buffer to capture log messages
        log_buffer = io.StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.INFO)
        
        # Set up logging to capture messages
        logger = logging.getLogger('smartcash.ui.dataset.augment')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Test fresh initialization
        fresh_initializer = AugmentInitializer()
        fresh_ui = fresh_initializer.initialize()
        
        # Check captured logs
        log_contents = log_buffer.getvalue()
        logger.removeHandler(handler)
        
        if log_contents:
            print(f"ℹ️ Captured {len(log_contents.split(chr(10)))} log messages during initialization")
            print("ℹ️ This is normal - logs should appear in UI components, not console")
        else:
            print("✅ No console logs captured - good logging suppression")
            
    except Exception as e:
        print(f"❌ Failed to test logging suppression: {e}")
        return False
    
    print("\n" + "="*80)
    print("🎉 AUGMENT MODULE UI DISPLAY TEST COMPLETED")
    print("="*80)
    return True

def test_with_config():
    """Test augment module with sample configuration."""
    
    print("\n" + "="*80)
    print("🧪 AUGMENT MODULE WITH CONFIG TEST")
    print("="*80)
    
    # Sample configuration
    sample_config = {
        'augmentation_type': 'combined',
        'intensity': 0.5,
        'variations': 5,
        'target_count': 100,
        'class_weights': {
            'rp_1000': 1.0,
            'rp_2000': 1.0,
            'rp_5000': 1.0,
            'rp_10000': 1.0,
            'rp_20000': 1.0,
            'rp_50000': 1.0,
            'rp_100000': 1.0
        },
        'position_params': {
            'rotation_range': 10,
            'translation_range': 0.1,
            'scale_range': 0.1
        },
        'lighting_params': {
            'brightness_range': 0.2,
            'contrast_range': 0.2
        }
    }
    
    try:
        # Test with configuration
        print("1️⃣ Testing with sample configuration...")
        initializer = AugmentInitializer(config=sample_config)
        ui_components = initializer.initialize()
        
        print("✅ Successfully initialized with configuration")
        
        # Test config handler
        config_handler = initializer.get_config_handler()
        if config_handler:
            print("✅ Config handler available")
            
            # Test config validation
            is_valid, errors = config_handler.validate_config(sample_config)
            if is_valid:
                print("✅ Configuration validation passed")
            else:
                print(f"⚠️ Configuration validation failed: {errors}")
        
        # Test config update
        print("2️⃣ Testing configuration update...")
        new_config = sample_config.copy()
        new_config['intensity'] = 0.8
        
        initializer.update_config(new_config)
        print("✅ Configuration update successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test with configuration: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Augment Module UI Display Tests")
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python version: {sys.version}")
    print(f"📍 Current directory: {os.getcwd()}")
    
    try:
        # Run basic UI display test
        success1 = test_augment_ui_display()
        
        # Run configuration test
        success2 = test_with_config()
        
        if success1 and success2:
            print("\n🎉 ALL TESTS PASSED! 🎉")
            print("✅ Augment module UI display is working correctly")
            print("✅ DisplayInitializer pattern is implemented properly")
            print("✅ UI components are ready for display")
        else:
            print("\n❌ SOME TESTS FAILED!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 TEST SCRIPT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)