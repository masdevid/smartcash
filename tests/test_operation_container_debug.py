#!/usr/bin/env python3
"""
Debug operation container integration issues.
"""

from smartcash.ui.model.pretrained.pretrained_uimodule import create_pretrained_uimodule

def debug_operation_container():
    """Debug operation container structure and methods."""
    print("🔍 DEBUGGING OPERATION CONTAINER INTEGRATION")
    print("=" * 60)
    
    # Create pretrained module
    module = create_pretrained_uimodule(auto_initialize=True)
    
    # Get components
    ui_components = module.get_ui_components()
    operation_container = ui_components.get('operation')
    operation_manager = module._operation_manager
    
    print("\\n1. Operation container structure:")
    print(f"Type: {type(operation_container)}")
    if isinstance(operation_container, dict):
        print(f"Keys: {list(operation_container.keys())}")
        
        # Check log_message method
        log_message = operation_container.get('log_message')
        print(f"log_message type: {type(log_message)}")
        print(f"log_message callable: {callable(log_message)}")
    
    print("\\n2. Operation manager setup:")
    print(f"Operation manager type: {type(operation_manager)}")
    print(f"Has _operation_container: {hasattr(operation_manager, '_operation_container')}")
    
    if hasattr(operation_manager, '_operation_container'):
        op_container = operation_manager._operation_container
        print(f"_operation_container type: {type(op_container)}")
        print(f"_operation_container is operation_container: {op_container is operation_container}")
        
        if isinstance(op_container, dict):
            print(f"_operation_container keys: {list(op_container.keys())}")
            
            # Test log_message method
            log_message = op_container.get('log_message')
            print(f"_operation_container log_message callable: {callable(log_message)}")
    
    print("\\n3. Testing operation manager log method directly:")
    
    # Test the log method
    if operation_manager:
        print("Calling operation_manager.log()...")
        try:
            operation_manager.log("🧪 Debug test log", 'info')
            print("✅ operation_manager.log() succeeded")
        except Exception as e:
            print(f"❌ operation_manager.log() failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\\n4. Testing log redirection manually:")
    if operation_container:
        log_accordion = operation_container.get('log_accordion')
        if log_accordion:
            print(f"Log accordion entries before: {len(log_accordion.log_entries)}")
            
            # Test direct log_message call
            try:
                operation_container['log_message']("🧪 Manual log_message test", 'info')
                print(f"Log accordion entries after log_message: {len(log_accordion.log_entries)}")
            except Exception as e:
                print(f"❌ log_message failed: {e}")
            
            # Test operation manager log call
            try:
                if hasattr(operation_manager, '_operation_container'):
                    op_container = operation_manager._operation_container
                    if 'log_message' in op_container:
                        op_container['log_message']("🧪 Operation manager container log test", 'info')
                        print(f"Log accordion entries after op mgr log: {len(log_accordion.log_entries)}")
            except Exception as e:
                print(f"❌ Operation manager container log failed: {e}")
    
    print("\\n✅ Debug completed")

if __name__ == "__main__":
    debug_operation_container()