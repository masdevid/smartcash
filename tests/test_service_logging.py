#!/usr/bin/env python3
"""
Test service logging callback flow.
"""

from smartcash.ui.model.pretrained.pretrained_uimodule import create_pretrained_uimodule

def test_service_logging_flow():
    """Test that service logs are properly routed through operation manager."""
    print("🔍 TESTING SERVICE LOGGING CALLBACK FLOW")
    print("=" * 60)
    
    # Create pretrained module
    module = create_pretrained_uimodule(auto_initialize=True)
    
    # Get components
    ui_components = module.get_ui_components()
    operation_container = ui_components.get('operation')
    operation_manager = module._operation_manager
    log_accordion = operation_container.get('log_accordion')
    
    print(f"📊 Initial log entries: {len(log_accordion.log_entries)}")
    
    print("\\n1. Testing operation manager log callback:")
    
    # Test the log callback directly
    def test_callback(message, level='info'):
        print(f"Callback received: {message} (level: {level})")
        operation_manager.log(message, level)
    
    test_callback("🧪 Test callback message", 'info')
    print(f"📊 After test callback: {len(log_accordion.log_entries)}")
    
    print("\\n2. Testing service operation with logging:")
    
    # Let's manually test the service with the callback
    service = operation_manager._service
    if service:
        print("Testing service check_existing_models with callback...")
        
        # Create a mock log callback that captures messages
        captured_logs = []
        
        def capturing_log_callback(message, level='info'):
            captured_logs.append((message, level))
            print(f"Service log: {message}")
            # Also call the operation manager log
            operation_manager.log(message, level)
        
        # Test the service method directly
        try:
            # This will be an async call, so we need to handle it properly
            result = operation_manager._safe_callback(
                service.check_existing_models(
                    '/data/pretrained',
                    log_callback=capturing_log_callback
                )
            )
            print(f"Service result type: {type(result)}")
            print(f"Service captured {len(captured_logs)} log messages")
            print(f"📊 After service call: {len(log_accordion.log_entries)}")
            
        except Exception as e:
            print(f"Service call failed: {e}")
    
    print("\\n3. Testing full operation execution:")
    print("Before refresh operation:")
    print(f"  - Log entries: {len(log_accordion.log_entries)}")
    
    # Execute the refresh operation
    result = module.execute_refresh()
    
    print("After refresh operation:")
    print(f"  - Log entries: {len(log_accordion.log_entries)}")
    print(f"  - Operation result: {result.get('success', False)}")
    
    print("\\n📋 All log entries:")
    for i, entry in enumerate(log_accordion.log_entries):
        print(f"  {i+1}. [{entry.level.name}] {entry.message}")
    
    print("\\n✅ Service logging flow test completed")

if __name__ == "__main__":
    test_service_logging_flow()