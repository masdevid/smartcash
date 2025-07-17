#!/usr/bin/env python3
"""
Test script to verify button click events are working
"""

import sys
import os

# Add the project root to Python path
project_root = '/Users/masdevid/Projects/smartcash'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_button_click_events():
    """Test that button click events are properly bound and working."""
    print("🧪 Testing Button Click Events...")
    
    try:
        from smartcash.ui.setup.colab.colab_uimodule import ColabUIModule
        from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
        
        # Test Colab module
        print("\n1. Testing Colab button clicks...")
        colab_module = ColabUIModule()
        colab_module.initialize()
        
        # Get the colab_setup button
        action_container = colab_module._ui_components.get('action_container')
        colab_setup_btn = action_container['buttons'].get('colab_setup')
        
        if colab_setup_btn:
            print("✅ Found colab_setup button")
            
            # Check if on_click handlers are registered
            if hasattr(colab_setup_btn, '_click_handlers') and colab_setup_btn._click_handlers:
                handler_count = len(colab_setup_btn._click_handlers.callbacks)
                print(f"✅ Button has {handler_count} click handler(s) registered")
                
                # Try to simulate a click
                print("🔄 Simulating button click...")
                try:
                    # This would normally be called by the widget framework
                    for callback in colab_setup_btn._click_handlers.callbacks:
                        result = callback(colab_setup_btn)
                        print("✅ Click handler executed successfully")
                        break
                except Exception as e:
                    print(f"❌ Click handler failed: {e}")
            else:
                print("❌ No click handlers found on button")
        else:
            print("❌ colab_setup button not found")
        
        # Test Dependency module
        print("\n2. Testing Dependency button clicks...")
        dep_module = DependencyUIModule()
        dep_module.initialize()
        
        # Get the install button
        action_container = dep_module._ui_components.get('action_container')
        install_btn = action_container['buttons'].get('install')
        
        if install_btn:
            print("✅ Found install button")
            
            # Check if on_click handlers are registered
            if hasattr(install_btn, '_click_handlers') and install_btn._click_handlers:
                handler_count = len(install_btn._click_handlers.callbacks)
                print(f"✅ Button has {handler_count} click handler(s) registered")
                
                # Try to simulate a click
                print("🔄 Simulating button click...")
                try:
                    # This would normally be called by the widget framework
                    for callback in install_btn._click_handlers.callbacks:
                        result = callback(install_btn)
                        print("✅ Click handler executed successfully")
                        break
                except Exception as e:
                    print(f"❌ Click handler failed: {e}")
            else:
                print("❌ No click handlers found on button")
        else:
            print("❌ install button not found")
            
        print("\n🎉 Button click event testing completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_button_click_events()