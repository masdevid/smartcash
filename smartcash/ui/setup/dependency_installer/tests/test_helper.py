"""
File: smartcash/ui/setup/dependency_installer/tests/test_helper.py
Deskripsi: Helper untuk unit testing
"""

import unittest
import warnings
import functools

class WarningTestCase(unittest.TestCase):
    """Test case yang menangani warning"""
    
    def setUp(self):
        """Setup untuk test case"""
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=ResourceWarning)
        warnings.filterwarnings('ignore', message='.*configuration.*')

def ignore_layout_warnings(func):
    """Decorator untuk mengabaikan layout warnings"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=ResourceWarning)
            warnings.filterwarnings('ignore', message='.*configuration.*')
            return func(*args, **kwargs)
    return wrapper

def ignore_config_warnings(func):
    """Decorator untuk mengabaikan configuration warnings"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*configuration.*')
            return func(*args, **kwargs)
    return wrapper
