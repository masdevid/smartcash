#!/usr/bin/env python3
"""
Final test for save/reset button functionality
"""

import sys
import os

# Add the project root to Python path
project_root = '/Users/masdevid/Projects/smartcash'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_save_reset_buttons():
    """Test save/reset button functionality."""
    print("🧪 Testing Save/Reset Button Functionality...")
    
    try:
        # Test Dependency module
        print("\n1. Testing Dependency module...")
        from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
        
        dep_module = DependencyUIModule()
        dep_module.initialize()
        
        # Test save button
        print("   Testing save button...")
        save_result = dep_module._handle_save_config()
        print(f"   Save result: {save_result.get('success', False)}")
        
        # Test reset button  
        print("   Testing reset button...")
        reset_result = dep_module._handle_reset_config()
        print(f"   Reset result: {reset_result.get('success', False)}")
        
        # Test Colab module
        print("\n2. Testing Colab module...")
        from smartcash.ui.setup.colab.colab_uimodule import ColabUIModule
        
        colab_module = ColabUIModule()
        colab_module.initialize()
        
        # Test save button
        print("   Testing save button...")
        save_result = colab_module._handle_save_config()
        print(f"   Save result: {save_result.get('success', False)}")
        
        # Test reset button
        print("   Testing reset button...")  
        reset_result = colab_module._handle_reset_config()
        print(f"   Reset result: {reset_result.get('success', False)}")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_save_reset_buttons()