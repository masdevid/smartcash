#!/usr/bin/env python3
"""
Test to understand logger redirection behavior.
"""

from smartcash.ui.model.pretrained.pretrained_uimodule import create_pretrained_uimodule
from smartcash.ui.components.log_accordion import LogLevel

def test_logger_behavior():
    """Test different logger methods to understand redirection."""
    print("🔍 TESTING LOGGER BEHAVIOR AND REDIRECTION")
    print("=" * 60)
    
    # Create pretrained module
    module = create_pretrained_uimodule(auto_initialize=True)
    
    # Get components
    ui_components = module.get_ui_components()
    operation_container = ui_components.get('operation')
    operation_manager = module._operation_manager
    
    if operation_container and operation_manager:
        log_accordion = operation_container.get('log_accordion')
        
        print(f"📊 Initial log entries: {len(log_accordion.log_entries)}")
        
        print("\\n1. Testing operation manager log method...")
        operation_manager.log("🔧 Operation manager log method", 'info')
        print(f"📊 After operation manager log: {len(log_accordion.log_entries)}")
        
        print("\\n2. Testing operation manager logger.info...")
        operation_manager.logger.info("🔧 Operation manager logger.info")
        print(f"📊 After logger.info: {len(log_accordion.log_entries)}")
        
        print("\\n3. Testing operation container log_message...")
        operation_container['log_message']("🔧 Direct operation container log", LogLevel.INFO)
        print(f"📊 After operation container log: {len(log_accordion.log_entries)}")
        
        print("\\n4. Testing log accordion direct log...")
        log_accordion.log("🔧 Direct log accordion", LogLevel.INFO, namespace="test")
        print(f"📊 After log accordion log: {len(log_accordion.log_entries)}")
        
        print("\\n5. Testing operation execution logging...")
        print("Before operation:")
        print(f"  - Log entries: {len(log_accordion.log_entries)}")
        
        # Execute operation and monitor logs
        print("Executing refresh operation...")
        result = module.execute_refresh()
        
        print("After operation:")
        print(f"  - Log entries: {len(log_accordion.log_entries)}")
        print(f"  - Result: {result}")
        
        print("\\n📋 All log entries after tests:")
        for i, entry in enumerate(log_accordion.log_entries):
            print(f"  {i+1}. [{entry.level.name}] {entry.message} (namespace: {entry.namespace})")
    
    print("\\n✅ Logger behavior test completed")

if __name__ == "__main__":
    test_logger_behavior()