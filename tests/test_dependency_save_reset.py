#!/usr/bin/env python3
"""
Test script for dependency module save/reset operations.

This script tests real save/reset operations in the dependency module to identify:
1. Whether save/reset creates logs in operation container
2. Whether save/reset updates status in header container 
3. Whether button states are correct during operations
"""

import sys
import os
sys.path.insert(0, '/Users/masdevid/Projects/smartcash')

from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
import time

def test_dependency_save_reset():
    """Test real save/reset operations in dependency module."""
    print("🧪 Testing dependency module save/reset operations...")
    
    # Create and initialize module
    module = DependencyUIModule()
    success = module.initialize()
    
    if not success:
        print("❌ Failed to initialize dependency module")
        return
    
    print("✅ Dependency module initialized successfully")
    
    # Get UI components
    ui_components = module.get_ui_components()
    if not ui_components:
        print("❌ Failed to get UI components")
        return
    
    print(f"📋 Available UI components: {list(ui_components.keys())}")
    
    # Test operation container logging capabilities
    operation_container = ui_components.get('operation_container')
    if operation_container:
        print(f"📊 Operation container type: {type(operation_container)}")
        if isinstance(operation_container, dict):
            print(f"📊 Operation container methods: {list(operation_container.keys())}")
        else:
            print(f"📊 Operation container methods: {dir(operation_container)}")
    
    # Test header container status update capabilities  
    header_container = ui_components.get('header_container')
    if header_container:
        print(f"📊 Header container type: {type(header_container)}")
        if hasattr(header_container, 'update_status'):
            print("✅ Header container has update_status method")
        else:
            print("❌ Header container missing update_status method")
    
    # Test save operation
    print("\n🔍 Testing SAVE operation:")
    save_button = ui_components.get('save_button')
    if save_button:
        print(f"📌 Save button found: {type(save_button)}")
        print(f"📌 Save button enabled: {not save_button.disabled}")
        
        # Clear any existing logs first
        if operation_container and isinstance(operation_container, dict) and 'clear_logs' in operation_container:
            operation_container['clear_logs']()
            print("🧹 Cleared existing logs")
        
        # Monitor button state before save
        print(f"📌 Button state before save - enabled: {not save_button.disabled}")
        
        # Trigger save operation by simulating click
        print("🔄 Triggering save operation...")
        save_button.click()
        
        # Wait a moment for async operations
        time.sleep(1)
        
        # Monitor button state after save
        print(f"📌 Button state after save - enabled: {not save_button.disabled}")
        
        # Check if operation container received logs
        if operation_container and isinstance(operation_container, dict):
            log_accordion = operation_container.get('log_accordion')
            if log_accordion and hasattr(log_accordion, '_log_entries'):
                log_entries = log_accordion._log_entries
                print(f"📋 Operation container log entries: {len(log_entries)}")
                for i, entry in enumerate(log_entries[-5:]):  # Show last 5 entries
                    print(f"  {i+1}. {entry.get('message', 'No message')}")
            else:
                print("❌ No log entries found in operation container")
        
        # Check header container status
        if header_container and hasattr(header_container, '_status_widget'):
            status_value = header_container._status_widget.value if hasattr(header_container._status_widget, 'value') else "No status"
            print(f"📊 Header status after save: {status_value}")
        
    else:
        print("❌ Save button not found")
    
    # Test reset operation
    print("\n🔍 Testing RESET operation:")
    reset_button = ui_components.get('reset_button')
    if reset_button:
        print(f"📌 Reset button found: {type(reset_button)}")
        print(f"📌 Reset button enabled: {not reset_button.disabled}")
        
        # Clear any existing logs first
        if operation_container and isinstance(operation_container, dict) and 'clear_logs' in operation_container:
            operation_container['clear_logs']()
            print("🧹 Cleared existing logs")
        
        # Monitor button state before reset
        print(f"📌 Button state before reset - enabled: {not reset_button.disabled}")
        
        # Trigger reset operation by simulating click
        print("🔄 Triggering reset operation...")
        reset_button.click()
        
        # Wait a moment for async operations
        time.sleep(1)
        
        # Monitor button state after reset
        print(f"📌 Button state after reset - enabled: {not reset_button.disabled}")
        
        # Check if operation container received logs
        if operation_container and isinstance(operation_container, dict):
            log_accordion = operation_container.get('log_accordion')
            if log_accordion and hasattr(log_accordion, '_log_entries'):
                log_entries = log_accordion._log_entries
                print(f"📋 Operation container log entries: {len(log_entries)}")
                for i, entry in enumerate(log_entries[-5:]):  # Show last 5 entries
                    print(f"  {i+1}. {entry.get('message', 'No message')}")
            else:
                print("❌ No log entries found in operation container")
        
        # Check header container status
        if header_container and hasattr(header_container, '_status_widget'):
            status_value = header_container._status_widget.value if hasattr(header_container._status_widget, 'value') else "No status"
            print(f"📊 Header status after reset: {status_value}")
        
    else:
        print("❌ Reset button not found")
    
    # Test with a main operation to compare button behavior
    print("\n🔍 Testing main operation for comparison:")
    main_button = ui_components.get('primary_button') or ui_components.get('check_button')
    if main_button:
        print(f"📌 Main button found: {type(main_button)}")
        print(f"📌 Main button enabled before operation: {not main_button.disabled}")
        
        # Clear any existing logs first
        if operation_container and isinstance(operation_container, dict) and 'clear_logs' in operation_container:
            operation_container['clear_logs']()
            print("🧹 Cleared existing logs")
        
        # Trigger main operation
        print("🔄 Triggering main operation...")
        main_button.click()
        
        # Check button state during operation (should be disabled)
        print(f"📌 Main button enabled during operation: {not main_button.disabled}")
        
        # Wait for operation to complete
        time.sleep(3)
        
        # Check button state after operation (should be enabled again)
        print(f"📌 Main button enabled after operation: {not main_button.disabled}")
        
        # Check logs from main operation
        if operation_container and isinstance(operation_container, dict):
            log_accordion = operation_container.get('log_accordion')
            if log_accordion and hasattr(log_accordion, '_log_entries'):
                log_entries = log_accordion._log_entries
                print(f"📋 Operation container log entries from main operation: {len(log_entries)}")
                for i, entry in enumerate(log_entries[-10:]):  # Show last 10 entries
                    print(f"  {i+1}. {entry.get('message', 'No message')}")
    
    print("\n✅ Save/reset testing completed")

if __name__ == "__main__":
    test_dependency_save_reset()