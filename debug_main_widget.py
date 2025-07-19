#!/usr/bin/env python3
"""
Debug script to check main widget availability in failing modules
"""

import sys
import os
sys.path.insert(0, '/Users/masdevid/Projects/smartcash')

def debug_module(module_name, factory_class, create_method):
    """Debug a specific module's main widget availability"""
    print(f"\n🔍 Debugging {module_name}:")
    print("=" * 50)
    
    try:
        # Create module instance
        module = factory_class.create_colab_module() if 'colab' in module_name.lower() else getattr(factory_class, create_method)()
        
        print(f"✅ Module created: {type(module)}")
        
        # Check if it has get_ui_components method
        if hasattr(module, 'get_ui_components'):
            components = module.get_ui_components()
            if 'error' in components:
                print(f"❌ Error in get_ui_components: {components['error']}")
                return
            
            print(f"📦 Available component keys: {list(components.keys())}")
            
            # Check for main widget keys
            for key in ['main_container', 'ui', 'container']:
                if key in components:
                    widget = components[key]
                    print(f"   {key}: {type(widget)} - {'✅ Available' if widget is not None else '❌ None'}")
                else:
                    print(f"   {key}: ❌ Missing")
            
            # Check get_main_widget method
            if hasattr(module, 'get_main_widget'):
                main_widget = module.get_main_widget()
                print(f"🎯 get_main_widget() result: {type(main_widget) if main_widget else 'None'}")
            else:
                print(f"❌ No get_main_widget method")
        else:
            print(f"❌ No get_ui_components method")
            
    except Exception as e:
        print(f"❌ Failed to create module: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test failing modules
    print("🚀 Debugging Main Widget Issues")
    
    # Colab
    try:
        from smartcash.ui.setup.colab.colab_ui_factory import ColabUIFactory
        debug_module("Colab", ColabUIFactory, "create_colab_module")
    except Exception as e:
        print(f"Failed to test Colab: {e}")
    
    # Split
    try:
        from smartcash.ui.dataset.split.split_ui_factory import SplitUIFactory
        debug_module("Split", SplitUIFactory, "create_split_module")
    except Exception as e:
        print(f"Failed to test Split: {e}")
    
    # Backbone
    try:
        from smartcash.ui.model.backbone.backbone_ui_factory import BackboneUIFactory
        debug_module("Backbone", BackboneUIFactory, "create_backbone_module")
    except Exception as e:
        print(f"Failed to test Backbone: {e}")