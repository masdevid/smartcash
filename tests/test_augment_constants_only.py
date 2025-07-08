#!/usr/bin/env python3
"""
Minimal test for augment constants only
"""

import sys
import os

# Add the smartcash directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_constants():
    """Test just the constants import."""
    try:
        from smartcash.ui.dataset.augment.constants import (
            AugmentationOperation, AugmentationTypes, UI_CONFIG, DEFAULT_AUGMENTATION_PARAMS
        )
        print("✅ Constants imported successfully")
        print(f"   - UI_CONFIG module: {UI_CONFIG['module_name']}")
        print(f"   - Default variations: {DEFAULT_AUGMENTATION_PARAMS['num_variations']}")
        print(f"   - Available operations: {[op.value for op in AugmentationOperation]}")
        return True
    except Exception as e:
        print(f"❌ Constants import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_handler():
    """Test config handler import and basic functionality."""
    try:
        from smartcash.ui.dataset.augment.configs.augment_config_handler import AugmentConfigHandler
        print("✅ Config handler imported successfully")
        
        # Test config handler creation
        handler = AugmentConfigHandler()
        print("✅ Config handler created successfully")
        
        # Test getting default config
        config = handler.config
        print(f"✅ Default config retrieved with {len(config)} sections")
        
        return True
    except Exception as e:
        print(f"❌ Config handler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing augment module constants...")
    
    success1 = test_constants()
    print()
    success2 = test_config_handler()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)