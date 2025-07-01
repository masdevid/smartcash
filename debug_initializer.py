#!/usr/bin/env python3
"""Debug script for ConfigCellInitializer."""

import sys
import os
import logging
from unittest.mock import MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Import after setting up logging
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer
from smartcash.ui.handlers.error_handler import create_error_response
import ipywidgets as widgets

class TestHandler:
    def __init__(self):
        self.config = {"test_key": "test_value"}
    
    def get_config(self):
        return self.config

class TestConfigCellInitializer(ConfigCellInitializer):
    def create_handler(self):
        logger.info("Creating test handler")
        return TestHandler()
    
    def create_ui_components(self, config):
        logger.info("Creating UI components")
        container = widgets.VBox()
        return {
            'container': container,
            'test_component': widgets.Label(value='Test Component')
        }
    
    def setup_event_handlers(self):
        logger.info("Setting up event handlers")
    
    def initialize_cleanup(self):
        logger.info("Initializing cleanup")

def test_initializer():
    logger.info("Starting test_initializer")
    
    # Create initializer
    initializer = TestConfigCellInitializer(
        module_name='test_module',
        config_filename='test_config.yaml'
    )
    
    logger.info("Initializer created")
    
    # Call initialize
    try:
        logger.info("Calling initialize()")
        result = initializer.initialize()
        logger.info(f"initialize() returned: {result}")
        logger.info(f"Type of result: {type(result)}")
        logger.info(f"Is instance of Widget: {isinstance(result, widgets.Widget)}")
        
        if hasattr(result, 'keys'):
            logger.info("Result has keys() method")
            logger.info(f"Keys: {list(result.keys())}")
            
            if 'container' in result and isinstance(result['container'], widgets.Widget):
                logger.info("Found 'container' key with Widget value")
                return result['container']
        
        return result
        
    except Exception as e:
        logger.error(f"Error in initialize(): {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("=== Starting debug script ===")
    try:
        result = test_initializer()
        logger.info("=== Test completed successfully ===")
        logger.info(f"Final result: {result}")
        logger.info(f"Final result type: {type(result)}")
        
        # Try to display the result if in IPython
        try:
            from IPython.display import display
            logger.info("Displaying result...")
            display(result)
        except ImportError:
            logger.warning("IPython not available, skipping display")
        
    except Exception as e:
        logger.error("=== Test failed ===", exc_info=True)
        sys.exit(1)
