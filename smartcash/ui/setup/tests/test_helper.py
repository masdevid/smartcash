"""
File: smartcash/ui/setup/tests/test_helper.py
Deskripsi: Helper untuk pengujian dengan implementasi untuk meredam warning
"""

import warnings
import functools
import unittest

def ignore_layout_warnings(test_func):
    """
    Decorator untuk meredam warning terkait Layout dari ipywidgets/traitlets.
    
    Args:
        test_func: Fungsi test yang akan dijalankan dengan warning diredam
        
    Returns:
        Fungsi wrapper yang meredam warning
    """
    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            # Filter warning dari traitlets terkait Layout
            warnings.filterwarnings(
                "ignore", 
                message="Passing unrecognized arguments to super\\(Layout\\).__init__",
                category=DeprecationWarning
            )
            # Jalankan fungsi test
            return test_func(*args, **kwargs)
    return wrapper

class WarningTestCase(unittest.TestCase):
    """Test case dasar dengan fitur untuk meredam warning."""
    
    @classmethod
    def setUpClass(cls):
        """Setup untuk class test dengan meredam warning."""
        super().setUpClass()
        # Redam warning dari traitlets terkait Layout
        warnings.filterwarnings(
            "ignore", 
            message="Passing unrecognized arguments to super\\(Layout\\).__init__",
            category=DeprecationWarning
        )
