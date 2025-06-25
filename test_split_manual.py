"""
Manual test script for SplitConfigHandler.
Run this directly in your environment to test the split configuration functionality.
"""

def test_split_config_handler():
    """Manual test for SplitConfigHandler functionality."""
    print("=== Testing SplitConfigHandler ===")
    
    # Import inside the function to handle any import errors gracefully
    try:
        from smartcash.ui.dataset.split.split_init import SplitConfigHandler
        from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config
        
        print("\n1. Creating handler with default config...")
        handler = SplitConfigHandler()
        print(f"Initial config: {handler.config}")
        
        print("\n2. Testing update_config...")
        new_config = {"data": {"split_ratios": {"train": 0.8, "valid": 0.1, "test": 0.1}}}
        handler.update_config(new_config)
        print(f"After update: {handler.config['data']['split_ratios']}")
        
        print("\n3. Testing reset_to_defaults...")
        handler.reset_to_defaults()
        print(f"After reset: {handler.config['data']['split_ratios']}")
        
        print("\n4. Testing update_from_ui...")
        # Create mock UI components
        mock_ui = {
            'train_slider': type('Slider', (), {'value': 0.7}),
            'valid_slider': type('Slider', (), {'value': 0.2}),
            'test_slider': type('Slider', (), {'value': 0.1})
        }
        
        # Import the extractor to ensure it's available
        from smartcash.ui.dataset.split.handlers.config_extractor import extract_split_config
        config = extract_split_config(mock_ui)
        print(f"Extracted config from UI: {config['data']['split_ratios']}")
        
        # Update handler from UI
        handler.update_from_ui(mock_ui)
        print(f"After UI update: {handler.config['data']['split_ratios']}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_split_config_handler()
