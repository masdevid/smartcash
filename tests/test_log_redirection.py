#!/usr/bin/env python3
"""
Test log redirection from operation managers to operation container log accordion.
"""

from smartcash.ui.model.pretrained.pretrained_uimodule import create_pretrained_uimodule
from smartcash.ui.components.log_accordion import LogLevel

def test_log_redirection():
    """Test that operation manager logs are redirected to operation container."""
    print("🔍 TESTING LOG REDIRECTION FROM OPERATION MANAGER")
    print("=" * 60)
    
    # Create pretrained module
    module = create_pretrained_uimodule(auto_initialize=True)
    
    # Get UI components
    ui_components = module.get_ui_components()
    operation_container = ui_components.get('operation')
    
    if operation_container:
        log_accordion = operation_container.get('log_accordion')
        if log_accordion:
            print(f"📊 Initial log entries: {len(log_accordion.log_entries)}")
            
            # Test direct logging to operation container
            print("\\nTesting direct logging to operation container...")
            operation_container['log_message']("🔧 Direct log test", LogLevel.INFO)
            print(f"📊 After direct log: {len(log_accordion.log_entries)}")
            
            # Test logging through operation manager
            print("\\nTesting logging through operation manager...")
            operation_manager = module._operation_manager
            if operation_manager:
                print(f"Operation manager available: {operation_manager is not None}")
                print(f"Operation manager has operation container: {hasattr(operation_manager, '_operation_container')}")
                
                # Test operation manager log method
                operation_manager.log("🚀 Operation manager test log", 'info')
                print(f"📊 After operation manager log: {len(log_accordion.log_entries)}")
                
                # Check if operation manager is using the right operation container
                if hasattr(operation_manager, '_operation_container'):
                    print(f"Operation manager container type: {type(operation_manager._operation_container)}")
                    print(f"Operation container type: {type(operation_container)}")
                    print(f"Containers match: {operation_manager._operation_container is operation_container}")
            
            print("\\nTesting operation execution with logging...")
            # Execute an operation that should log messages
            result = module.execute_refresh()
            print(f"📊 After operation execution: {len(log_accordion.log_entries)}")
            print(f"Operation result: {result}")
            
            # Show all log entries
            print("\\n📋 All log entries:")
            for i, entry in enumerate(log_accordion.log_entries):
                print(f"  {i+1}. [{entry.level.name}] {entry.message} (namespace: {entry.namespace})")
            
            # Show filtered entries
            filtered = log_accordion._get_filtered_entries()
            print(f"\\n📋 Filtered entries ({len(filtered)}):")
            for i, entry in enumerate(filtered):
                print(f"  {i+1}. [{entry.level.name}] {entry.message} (namespace: {entry.namespace})")
        else:
            print("❌ No log accordion found in operation container")
    else:
        print("❌ No operation container found in UI components")
    
    print("\\n✅ Log redirection test completed")

if __name__ == "__main__":
    test_log_redirection()