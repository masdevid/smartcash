#!/usr/bin/env python3
"""
Test script to verify training module logging appears in log_accordion.
"""

def test_training_logs():
    """Test that training module logs appear in log_accordion."""
    try:
        print("🧪 Testing training module logging to log_accordion...")
        
        # Add project to path
        import sys
        sys.path.append('/Users/masdevid/Projects/smartcash')
        
        from smartcash.ui.model.train.train_uimodule import TrainUIModule
        
        # Create module
        print("🔧 Creating training module...")
        train_module = TrainUIModule()
        
        # Initialize module
        print("🔧 Initializing training module...")
        train_module.initialize()
        
        if not train_module.is_ready():
            print("❌ Training module failed to initialize")
            return False
        
        print("✅ Training module initialized successfully")
        
        # Test logging through operation manager
        print("📝 Testing operation manager logging...")
        if train_module._operation_manager:
            train_module._operation_manager.log("🧪 Test message from operation manager", 'info')
            train_module._operation_manager.log("✅ Operation manager logging test", 'success')
            train_module._operation_manager.log("⚠️ Warning test message", 'warning')
            print("✅ Operation manager logging test completed")
        else:
            print("❌ Operation manager not available")
            return False
        
        # Test logging through UIModule
        print("📝 Testing UIModule logging...")
        train_module.log("🧪 Test message from UIModule", 'info') 
        train_module.log("✅ UIModule logging test", 'success')
        print("✅ UIModule logging test completed")
        
        # Test button handler logging
        print("📝 Testing button handler logging...")
        train_module._handle_validate_model()
        train_module._handle_refresh_backbone_config()
        print("✅ Button handler logging test completed")
        
        # Check operation container
        operation_container = train_module._ui_components.get('operation_container')
        if operation_container:
            print("✅ Operation container found")
            
            # Check if it has log_message method
            if isinstance(operation_container, dict):
                if 'log_message' in operation_container:
                    print("✅ log_message method found in operation container dict")
                elif 'log' in operation_container:
                    print("✅ log method found in operation container dict")
                else:
                    print("❌ No logging method found in operation container dict")
                    print(f"Available keys: {list(operation_container.keys())}")
            elif hasattr(operation_container, 'log_message'):
                print("✅ log_message method found in operation container object")
            else:
                print("❌ No log_message method found in operation container object")
                print(f"Available attributes: {[attr for attr in dir(operation_container) if not attr.startswith('_')]}")
        else:
            print("❌ Operation container not found")
            return False
        
        print("🎉 Training module logging test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_logs()
    print(f"\n🎯 Test {'PASSED' if success else 'FAILED'}")