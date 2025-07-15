#!/usr/bin/env python3
"""
Debug script to investigate button handler registration issues
"""

import sys
import os

# Add the project root to Python path
project_root = '/Users/masdevid/Projects/smartcash'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def debug_colab_button_handlers():
    """Debug Colab module button handlers."""
    print("🔍 Debugging Colab Button Handlers...")
    
    try:
        from smartcash.ui.setup.colab.colab_uimodule import ColabUIModule
        
        # Create and initialize module
        colab_module = ColabUIModule()
        print(f"✅ Module created: {colab_module}")
        
        # Check button handlers before initialization
        print(f"\n📋 Button handlers before init: {getattr(colab_module, '_button_handlers', {})}")
        
        # Initialize module
        success = colab_module.initialize()
        print(f"✅ Initialization {'successful' if success else 'failed'}")
        
        # Check button handlers after initialization
        print(f"\n📋 Button handlers after init: {colab_module._button_handlers}")
        
        # Check UI components
        if hasattr(colab_module, '_ui_components') and colab_module._ui_components:
            action_container = colab_module._ui_components.get('action_container')
            print(f"\n🎛️ Action container: {action_container}")
            
            if action_container:
                buttons = action_container.get('buttons', {})
                print(f"🔘 Available buttons: {list(buttons.keys())}")
                
                # Check specific button
                colab_setup_btn = buttons.get('colab_setup')
                print(f"🔘 colab_setup button: {colab_setup_btn}")
                
                if colab_setup_btn:
                    print(f"🔘 Button type: {type(colab_setup_btn)}")
                    print(f"🔘 Has on_click: {hasattr(colab_setup_btn, 'on_click')}")
                    
                    # Check if button has any click handlers
                    if hasattr(colab_setup_btn, '_click_handlers'):
                        print(f"🔘 Click handlers: {getattr(colab_setup_btn, '_click_handlers', [])}")
        
        # Test manual button handler registration
        print(f"\n🧪 Testing manual button click...")
        if 'colab_setup' in colab_module._button_handlers:
            handler = colab_module._button_handlers['colab_setup']
            print(f"🔧 Handler found: {handler}")
            
            # Try calling the handler directly
            try:
                result = handler()
                print(f"✅ Handler executed successfully: {result}")
            except Exception as e:
                print(f"❌ Handler execution failed: {e}")
        else:
            print("❌ No handler found for 'colab_setup'")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def debug_dependency_button_handlers():
    """Debug Dependency module button handlers."""
    print("\n🔍 Debugging Dependency Button Handlers...")
    
    try:
        from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
        
        # Create and initialize module
        dep_module = DependencyUIModule()
        print(f"✅ Module created: {dep_module}")
        
        # Check button handlers before initialization
        print(f"\n📋 Button handlers before init: {getattr(dep_module, '_button_handlers', {})}")
        
        # Initialize module
        success = dep_module.initialize()
        print(f"✅ Initialization {'successful' if success else 'failed'}")
        
        # Check button handlers after initialization
        print(f"\n📋 Button handlers after init: {dep_module._button_handlers}")
        
        # Check UI components
        if hasattr(dep_module, '_ui_components') and dep_module._ui_components:
            action_container = dep_module._ui_components.get('action_container')
            print(f"\n🎛️ Action container: {action_container}")
            
            if action_container:
                buttons = action_container.get('buttons', {})
                print(f"🔘 Available buttons: {list(buttons.keys())}")
                
                # Check each button
                for btn_id, btn_widget in buttons.items():
                    print(f"🔘 Button '{btn_id}': {btn_widget}")
                    if btn_widget and hasattr(btn_widget, 'on_click'):
                        print(f"   ✅ Has on_click method")
                    else:
                        print(f"   ❌ Missing on_click method")
        
        # Test a specific handler
        print(f"\n🧪 Testing manual button click for 'install'...")
        if 'install' in dep_module._button_handlers:
            handler = dep_module._button_handlers['install']
            print(f"🔧 Handler found: {handler}")
        else:
            print("❌ No handler found for 'install'")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_colab_button_handlers()
    debug_dependency_button_handlers()