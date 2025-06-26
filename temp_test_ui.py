"""Temporary test file to debug UI component structure."""
import sys
import pytest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from smartcash.ui.components.log_accordion import create_log_accordion

def test_real_log_accordion():
    """Test the real create_log_accordion function."""
    # Call the real function
    result = create_log_accordion(module_name='Test')
    
    # Debug print
    print("\n=== Real create_log_accordion Result ===")
    print(f"Returned keys: {list(result.keys())}")
    
    # Check the structure
    assert 'log_accordion' in result, "log_accordion key missing"
    assert 'log_output' in result, "log_output key missing"
    assert 'entries_container' in result, "entries_container key missing"
    
    # Get the widgets
    log_accordion = result['log_accordion']
    log_output = result['log_output']
    entries_container = result['entries_container']
    
    # Debug print widget types
    print(f"\n=== Widget Types ===")
    print(f"log_accordion: {type(log_accordion).__name__}")
    print(f"log_output: {type(log_output).__name__}")
    print(f"entries_container: {type(entries_container).__name__}")
    
    # Debug print widget attributes
    print("\n=== log_accordion attributes ===")
    print(f"Children: {[type(c).__name__ for c in log_accordion.children] if hasattr(log_accordion, 'children') else 'No children'}")
    
    print("\n=== log_output attributes ===")
    print(f"Has append_log: {hasattr(log_output, 'append_log')}")
    print(f"Has clear_logs: {hasattr(log_output, 'clear_logs')}")
    
    # Check widget types
    assert isinstance(log_accordion, widgets.Accordion), "log_accordion should be an Accordion"
    assert hasattr(log_output, 'append_log'), "log_output should have append_log method"
    assert hasattr(log_output, 'clear_logs'), "log_output should have clear_logs method"
    assert isinstance(entries_container, widgets.VBox), "entries_container should be a VBox"
    
    print("\n=== All assertions passed! ===")

if __name__ == "__main__":
    test_real_log_accordion()
