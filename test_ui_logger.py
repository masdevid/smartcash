"""
File: test_ui_logger.py
Deskripsi: Test sederhana untuk UILogger yang telah diperbaiki
"""

import sys
import logging
from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display, HTML

# Import UILogger secara langsung untuk menghindari masalah import
from smartcash.ui.utils.ui_logger import UILogger

class DummyOutput:
    """Dummy Output widget untuk test di luar notebook"""
    def __init__(self):
        pass
    
    def clear_output(self, wait=False):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def create_test_ui_components() -> Dict[str, Any]:
    """Buat UI components dummy untuk test"""
    dummy_output = DummyOutput()
    return {
        'status': dummy_output,
        'log_output': dummy_output
    }

def test_ui_logger():
    """Test dasar untuk UILogger"""
    print("=== Test UILogger ===")
    
    # Buat UI components
    ui_components = create_test_ui_components()
    
    # Buat UILogger instance
    logger = UILogger(ui_components, name="test_logger", log_level=logging.INFO)
    
    # Test log messages
    print("Test log info:")
    logger.info("Ini adalah pesan info")
    
    print("Test log warning:")
    logger.warning("Ini adalah pesan warning")
    
    print("Test log error:")
    logger.error("Ini adalah pesan error")
    
    print("Test log critical:")
    logger.critical("Ini adalah pesan critical")
    
    print("Test log success:")
    logger.success("Ini adalah pesan success")
    
    # Test log pesan kosong (seharusnya tidak muncul)
    print("Test log pesan kosong (tidak seharusnya ada output):")
    logger.info("")
    logger.info("   ")
    logger.info("\n")
    
    print("=== Test selesai ===")

if __name__ == "__main__":
    test_ui_logger() 